import torch
import torch.nn as nn
import logging

class CumulativeMLP(nn.Module):
    def __init__(self, in_dim=6, out_dim=384, dropout=0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        B, N, _ = x.size()
        return self.mlp(x.view(-1, 6)).view(B, N, -1)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_teeth, max_stages, num_layers=1, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_stages, embed_dim))  # Stage embeddings
        self.tooth_pos_embed = nn.Parameter(torch.zeros(1, num_teeth, embed_dim))  # Tooth embeddings
        
        # Cumulative MLP
        self.cumulative_mlp = CumulativeMLP(in_dim=6, out_dim=embed_dim, dropout=0.4)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.4,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-4
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Prediction heads
        self.ratio_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_stages)  # Predict max_stages ratios
        )
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=1e-2)

    def forward(self, memory, cumulative_transforms, num_stages=None):
        logger = logging.getLogger('TrainLogger')
        B = memory.size(0)
        device = memory.device

        # Validate input shapes
        expected_memory_shape = (B, self.num_teeth, self.embed_dim)
        expected_cumulative_shape = (B, self.num_teeth, 6)
        if memory.shape != expected_memory_shape:
            logger.error(f"Invalid memory shape: got {memory.shape}, expected {expected_memory_shape}")
            raise RuntimeError(f"Memory shape mismatch: got {memory.shape}, expected {expected_memory_shape}")
        if cumulative_transforms.shape != expected_cumulative_shape:
            logger.error(f"Invalid cumulative_transforms shape: got {cumulative_transforms.shape}, expected {expected_cumulative_shape}")
            raise RuntimeError(f"Cumulative_transforms shape mismatch: got {cumulative_transforms.shape}, expected {expected_cumulative_shape}")

        # Handle NaNs
        memory = torch.nan_to_num(memory, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_transforms = torch.nan_to_num(cumulative_transforms, nan=0.0, posinf=1.0, neginf=-1.0)

        # Fuse memory with cumulative transforms
        memory = memory + self.cumulative_mlp(cumulative_transforms)  # [B, num_teeth, embed_dim]

        # Memory key padding mask
        memory_key_padding_mask = torch.zeros(B, self.num_teeth, dtype=torch.bool, device=device)

        # Initialize outputs
        ratios_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        directions_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)

        # Prepare target embeddings
        tooth_pos = self.tooth_pos_embed.expand(B, -1, -1)  # [B, num_teeth, embed_dim]
        pos_embeds = self.pos_embed.expand(B, -1, -1).unsqueeze(2)  # [B, max_stages, 1, embed_dim]
        tgt = tooth_pos.unsqueeze(1) + pos_embeds  # [B, max_stages, num_teeth, embed_dim]
        tgt = self.pre_norm(tgt)  # [B, max_stages, num_teeth, embed_dim]

        # Reshape for decoder
        tgt = tgt.view(B, self.max_stages * self.num_teeth, self.embed_dim)  # [B, max_stages * num_teeth, embed_dim]
        memory = memory.repeat_interleave(self.max_stages, dim=1)  # [B, max_stages * num_teeth, embed_dim]
        memory_key_padding_mask = memory_key_padding_mask.repeat(1, self.max_stages)  # [B, max_stages * num_teeth]

        # Decoder forward pass
        output = self.decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)  # [B, max_stages * num_teeth, embed_dim]
        output = self.final_norm(output)  # [B, max_stages * num_teeth, embed_dim]
        output = output.view(B, self.max_stages, self.num_teeth, -1)  # [B, max_stages, num_teeth, embed_dim]

        # Predict ratios
        ratios = self.ratio_head(output.view(-1, self.embed_dim)).view(B, self.max_stages, self.num_teeth, 6)  # [B, max_stages, num_teeth, 6]
        ratios_sequence = torch.softmax(ratios, dim=1)  # Softmax across stages

        # Predict directions
        directions = self.direction_head(output.view(-1, self.embed_dim)).view(B, self.max_stages, self.num_teeth, 6)  # [B, max_stages, num_teeth, 6]
        directions_sequence = directions

        # Enforce zero stage-wise transforms where cumulative transform is zero
        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)  # [B, 1, num_teeth, 6]
        ratios_sequence = ratios_sequence * (1 - cumulative_zero_mask)
        directions_sequence = directions_sequence * (1 - cumulative_zero_mask)

        # Apply stage mask
        stage_mask = torch.ones(B, self.max_stages, 1, 1, device=device)  # [B, max_stages, 1, 1]
        if num_stages is not None:
            for i in range(B):
                stage = num_stages[i]
                assert isinstance(stage, (int, torch.Tensor)) and 0 <= stage <= self.max_stages, f"Invalid num_stages[{i}]: {stage}"
                stage_mask[i, stage:] = 0.0
        ratios_sequence = ratios_sequence * stage_mask
        directions_sequence = directions_sequence * stage_mask

        logger.debug(f"Ratios sequence mean: {ratios_sequence.mean().item():.4f}")
        logger.debug(f"Directions sequence mean: {directions_sequence.mean().item():.4f}")

        return [ratios_sequence, directions_sequence]