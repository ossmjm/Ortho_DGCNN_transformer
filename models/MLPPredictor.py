import torch
import torch.nn as nn
import logging
import math

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

class MLPPredictor(nn.Module):
    def __init__(self, embed_dim=384, num_teeth=14, max_stages=25, num_heads=4, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        self.num_params = 6
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_stages, embed_dim))  # Stage embeddings
        self.tooth_pos_embed = nn.Parameter(torch.zeros(1, num_teeth, embed_dim))  # Tooth embeddings
        
        # Initialize tooth positional embeddings with sinusoidal encoding
        self._init_tooth_pos_embed()
        
        # Cumulative MLP
        self.cumulative_mlp = CumulativeMLP(in_dim=6, out_dim=embed_dim, dropout=dropout)
        
        # Shared deep MLP with bottleneck architecture
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, eps=1e-3),
            nn.Dropout(0.3),  # Reduced dropout
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, eps=1e-3),
            nn.Dropout(0.3),
            nn.Linear(64, embed_dim)
        )
        
        # Stage embedding MLP with consistent intermediate size
        self.stage_embed_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout
            nn.Linear(64, embed_dim)
        )
        
        # Per-tooth attention and prediction heads
        self.tooth_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_teeth)
        ])
        self.tooth_ratio_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64, eps=1e-3),  # Added batch norm
                nn.Dropout(0.3),  # Reduced dropout
                nn.Linear(64, max_stages * self.num_params)  # Ratios for 6 parameters
            ) for _ in range(num_teeth)
        ])
        self.tooth_direction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64, eps=1e-3),  # Added batch norm
                nn.Dropout(0.3),  # Reduced dropout
                nn.Linear(64, self.num_params)  # Directions for 6 parameters
            ) for _ in range(num_teeth)
        ])
        
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-4)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-4)
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-4)  # Added for pre-attention normalization
        
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

    def _init_tooth_pos_embed(self):
        fdi_indices = torch.tensor(
            [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47][:self.num_teeth],
            dtype=torch.float32
        )
        pos = fdi_indices.unsqueeze(1)  # [num_teeth, 1]
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim))
        pe = torch.zeros(self.num_teeth, self.embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.tooth_pos_embed.data = pe.unsqueeze(0)  # [1, num_teeth, embed_dim]

    def forward(self, memory, cumulative_transforms, num_stages=None):
        logger = logging.getLogger('TrainLogger')
        B = memory.size(0)
        device = memory.device

        # Validate input shapes
        expected_memory_shape = (B, self.num_teeth, self.embed_dim)
        expected_cumulative_shape = (B, self.num_teeth, self.num_params)
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

        # Apply tooth positional embeddings and shared MLP with residual connection
        tooth_pos = self.tooth_pos_embed.expand(B, -1, -1)  # [B, num_teeth, embed_dim]
        memory = self.norm1(memory + tooth_pos)  # [B, num_teeth, embed_dim]
        memory_residual = memory
        memory = self.shared_mlp(memory.view(-1, self.embed_dim)).view(B, self.num_teeth, self.embed_dim)
        memory = memory + memory_residual  # Residual connection
        memory = self.norm2(memory)  # [B, num_teeth, embed_dim]

        # Initialize outputs
        ratios_sequence = torch.zeros(B, self.max_stages, self.num_teeth, self.num_params, device=device)
        directions_sequence = torch.zeros(B, self.max_stages, self.num_teeth, self.num_params, device=device)

        # Process stage embeddings
        pos_embeds = self.pos_embed.expand(B, -1, -1)  # [B, max_stages, embed_dim]
        pos_embeds = self.stage_embed_mlp(pos_embeds)  # [B, max_stages, embed_dim]

        # Per-tooth processing
        for t in range(self.num_teeth):
            # Extract tooth-specific features
            tooth_features = memory[:, t:t+1, :]  # [B, 1, embed_dim]
            tooth_features = tooth_features.unsqueeze(1).expand(-1, self.max_stages, -1, -1)  # [B, max_stages, 1, embed_dim]
            tooth_features = tooth_features + pos_embeds.unsqueeze(2)  # [B, max_stages, 1, embed_dim]
            tooth_features = self.norm2(tooth_features)  # [B, max_stages, 1, embed_dim]

            # Apply pre-attention normalization
            tooth_features = self.attention_norm(tooth_features.view(B, self.max_stages, self.embed_dim))  # [B, max_stages, embed_dim]

            # Apply tooth-specific attention over stages
            tooth_features, _ = self.tooth_attention[t](
                tooth_features,
                tooth_features,
                tooth_features
            )  # [B, max_stages, embed_dim]

            # Predict ratios with residual connection
            ratios_residual = tooth_features
            ratios = self.tooth_ratio_heads[t](tooth_features)  # [B, max_stages, max_stages * num_params]
            ratios = ratios.view(B, self.max_stages, self.num_params, self.max_stages)  # [B, max_stages, num_params, max_stages]
            ratios = torch.softmax(ratios, dim=1)  # Softmax across stages (dim=1)
            ratios_sequence[:, :, t, :] = ratios[:, :, :, 0]  # [B, max_stages, num_params]

            # Predict directions with residual connection
            directions = self.tooth_direction_heads[t](tooth_features)  # [B, max_stages, num_params]
            directions_sequence[:, :, t, :] = torch.sigmoid(directions)  # [B, max_stages, num_params]

        # Enforce zero stage-wise transforms where cumulative transform is zero
        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)  # [B, 1, num_teeth, num_params]
        ratios_sequence = ratios_sequence * (1 - cumulative_zero_mask)
        directions_sequence = directions_sequence * (1 - cumulative_zero_mask)

        # # Apply stage mask
        # stage_mask = torch.ones(B, self.max_stages, 1, 1, device=device)  # [B, max_stages, 1, 1]
        # if num_stages is not None:
        #     for i in range(B):
        #         stage = num_stages[i]
        #         assert isinstance(stage, (int, torch.Tensor)) and 0 <= stage <= self.max_stages, f"Invalid num_stages[{i}]: {stage}"
        #         stage_mask[i, stage:] = 0.0
        # ratios_sequence = ratios_sequence * stage_mask
        # directions_sequence = directions_sequence * stage_mask

        logger.debug(f"Ratios sequence mean: {ratios_sequence.mean().item():.4f}")
        logger.debug(f"Directions sequence mean: {directions_sequence.mean().item():.4f}")

        return [ratios_sequence, directions_sequence]