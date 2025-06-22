import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

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

class PerToothTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, num_teeth=14, max_stages=25, num_layers=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_stages, embed_dim))
        self.ratio_embed = nn.Linear(6, embed_dim)  # Embedding for teacher-forced ratios
        self.direction_embed = nn.Linear(6, embed_dim)  # Embedding for teacher-forced directions
        
        # Normalization layers
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.interaction_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # Cross-tooth attention
        self.cross_tooth_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.2, batch_first=True)
        
        # Previous targets attention
        self.prev_targets_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.2, batch_first=True)
        
        # Cumulative MLP to match cumulative_transforms to memory size
        self.cumulative_mlp = CumulativeMLP(in_dim=6, out_dim=embed_dim, dropout=0.4)
        
        # Per-tooth transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-5
        )
        self.tooth_decoders = nn.ModuleList([
            nn.TransformerDecoder(decoder_layer, num_layers=num_layers) for _ in range(num_teeth)
        ])

        # MLP heads from TransformerDecoder
        self.ratio_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, max_stages * 6)  # Predict ratios for all stages and parameters
        )
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, max_stages * 6)  # Predict directions logits for all stages
        )

        self.tooth_pos_embed = nn.Parameter(torch.zeros(1, num_teeth, embed_dim))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for attn in [self.cross_tooth_attention, self.prev_targets_attention]:
            nn.init.xavier_uniform_(attn.out_proj.weight, gain=1.0)
            if attn.out_proj.bias is not None:
                nn.init.zeros_(attn.out_proj.bias)

    def _get_teacher_forcing_params(self, epoch, total_epochs, stage_idx, num_stages, val_loss=None, base_tf_prob=0.9):
        logger = logging.getLogger('TrainLogger')
        progress = epoch / total_epochs
        tf_prob = max(0.3, base_tf_prob - 0.6 * progress)
        logger.debug(f"Teacher forcing prob: epoch={epoch}, stage_idx={stage_idx}, tf_prob={tf_prob:.4f}")
        return tf_prob

    def forward(self, memory, cumulative_transforms, num_stages=None, targets=None, directions=None, training=False, epoch=0, total_epochs=100, val_loss=None):
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
        if targets is not None:
            expected_targets_shape = (B, self.max_stages, self.num_teeth, 6)
            if targets.shape != expected_targets_shape:
                logger.error(f"Invalid targets shape: got {targets.shape}, expected {expected_targets_shape}")
                raise RuntimeError(f"Targets shape mismatch: got {targets.shape}, expected {expected_targets_shape}")
        if directions is not None:
            expected_directions_shape = (B, self.max_stages, self.num_teeth, 6)
            if directions.shape != expected_directions_shape:
                logger.error(f"Invalid directions shape: got {directions.shape}, expected {expected_directions_shape}")
                raise RuntimeError(f"Directions shape mismatch: got {directions.shape}, expected {expected_directions_shape}")

        # Handle NaNs
        memory = torch.nan_to_num(memory, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_transforms = torch.nan_to_num(cumulative_transforms, nan=0.0, posinf=1.0, neginf=-1.0)
        if targets is not None:
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=-1.0)
        if directions is not None:
            directions = torch.nan_to_num(directions, nan=0.0, posinf=1.0, neginf=-1.0)

        # Memory key padding mask
        memory_key_padding_mask = torch.zeros(B, self.num_teeth, dtype=torch.bool, device=device)

        # Process cumulative_transforms to match memory size
        cumulative_embed = self.cumulative_mlp(cumulative_transforms)  # [B, num_teeth, embed_dim]
        cumulative_embed = torch.nan_to_num(cumulative_embed, nan=0.0, posinf=1.0, neginf=-1.0)

        # Merge memory and cumulative_embed
        memory = memory + cumulative_embed  # [B, num_teeth, embed_dim]

        # Apply cross-tooth attention
        memory_attn, _ = self.cross_tooth_attention(memory, memory, memory, key_padding_mask=memory_key_padding_mask)
        memory = memory + memory_attn

        # Initialize outputs
        ratios_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        directions_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)

        # Process each stage
        prev_ratios_seq = []
        prev_directions_seq = []
        stage_outputs = []
        for stage_idx in range(self.max_stages):
            tf_prob = self._get_teacher_forcing_params(epoch, total_epochs, stage_idx, num_stages, val_loss)
            effective_tf_prob = tf_prob if training else 0.0
            use_tf = training and stage_idx > 0 and torch.rand(1).item() < effective_tf_prob

            # Prepare target embeddings
            start_idx = max(0, stage_idx - 1)
            if use_tf and targets is not None and directions is not None:
                ratios_prev = targets[:, start_idx:stage_idx, :, :]  # [B, num_stages_to_use, num_teeth, 6]
                directions_prev = directions[:, start_idx:stage_idx, :, :]  # [B, num_stages_to_use, num_teeth, 6]
                if ratios_prev.shape[1] > 0:
                    embedded_ratios = self.ratio_embed(ratios_prev)  # [B, num_stages_to_use, num_teeth, embed_dim]
                    embedded_directions = self.direction_embed(directions_prev)  # [B, num_stages_to_use, num_teeth, embed_dim]
                    embedded_prev = embedded_ratios + embedded_directions  # [B, num_stages_to_use, num_teeth, embed_dim]
                    embedded_prev = embedded_prev.mean(dim=1)  # Aggregate across stages: [B, num_teeth, embed_dim]
                    query = self.pos_embed[:, stage_idx, :].unsqueeze(1).expand(B, 1, self.num_teeth, -1)
                    embedded_prev, _ = self.prev_targets_attention(query.view(B, self.num_teeth, -1), embedded_prev, embedded_prev)  # [B, num_teeth, embed_dim]
                else:
                    embedded_prev = self.ratio_embed(torch.zeros(B, self.num_teeth, 6, device=device))  # [B, num_teeth, embed_dim]
            else:
                if prev_ratios_seq:
                    ratios_prev = torch.stack(prev_ratios_seq, dim=1)[:, start_idx:stage_idx, :, :]  # [B, num_stages_to_use, num_teeth, 6]
                    directions_prev = torch.stack(prev_directions_seq, dim=1)[:, start_idx:stage_idx, :, :]  # [B, num_stages_to_use, num_teeth, 6]
                    if ratios_prev.shape[1] > 0:
                        embedded_ratios = self.ratio_embed(ratios_prev)  # [B, num_stages_to_use, num_teeth, embed_dim]
                        embedded_directions = self.direction_embed(directions_prev)  # [B, num_stages_to_use, num_teeth, embed_dim]
                        embedded_prev = embedded_ratios + embedded_directions  # [B, num_stages_to_use, num_teeth, embed_dim]
                        embedded_prev = embedded_prev.mean(dim=1)  # Aggregate across stages: [B, num_teeth, embed_dim]
                        query = self.pos_embed[:, stage_idx, :].unsqueeze(1).expand(B, 1, self.num_teeth, -1)
                        embedded_prev, _ = self.prev_targets_attention(query.view(B, self.num_teeth, -1), embedded_prev, embedded_prev)  # [B, num_teeth, embed_dim]
                    else:
                        embedded_prev = self.ratio_embed(torch.zeros(B, self.num_teeth, 6, device=device))  # [B, num_teeth, embed_dim]
                else:
                    embedded_prev = self.ratio_embed(torch.zeros(B, self.num_teeth, 6, device=device))  # [B, num_teeth, embed_dim]

            # Combine embeddings
            pos_embed = self.pos_embed[:, stage_idx, :].unsqueeze(1).expand(-1, self.num_teeth, -1)
            tooth_pos = self.tooth_pos_embed.expand(B, -1, -1)
            tgt = embedded_prev + pos_embed + tooth_pos  # [B, num_teeth, embed_dim]
            tgt = self.pre_norm(tgt)

            # Apply cross-tooth attention
            tooth_interaction, _ = self.cross_tooth_attention(tgt, tgt, tgt)  # [B, num_teeth, embed_dim]
            tgt = self.interaction_norm(tgt + tooth_interaction)  # Residual connection

            # Decoder forward pass for each tooth
            output = torch.zeros(B, self.num_teeth, self.embed_dim, device=device)
            for tooth_idx in range(self.num_teeth):
                tooth_tgt = tgt[:, tooth_idx:tooth_idx+1, :]  # [B, 1, embed_dim]
                tooth_memory = memory[:, tooth_idx:tooth_idx+1, :]  # [B, 1, embed_dim]
                tooth_output = self.tooth_decoders[tooth_idx](tooth_tgt, tooth_memory, memory_key_padding_mask=memory_key_padding_mask[:, tooth_idx:tooth_idx+1])  # [B, 1, embed_dim]
                output[:, tooth_idx, :] = tooth_output.squeeze(1)

            output = self.final_norm(output)
            stage_outputs.append(output)

            # Update previous sequences for teacher forcing
            if use_tf and stage_idx < self.max_stages - 1:
                if targets is not None:
                    prev_ratios_seq.append(targets[:, stage_idx, :, :])
                if directions is not None:
                    prev_directions_seq.append(directions[:, stage_idx, :, :])

        # Stack outputs across stages
        stage_outputs = torch.stack(stage_outputs, dim=1)  # [B, max_stages, num_teeth, embed_dim]

        # Predict ratios for all stages
        ratios = self.ratio_head(stage_outputs.view(-1, self.embed_dim))  # [B * max_stages * num_teeth, max_stages * 6]
        ratios = ratios.view(B, self.max_stages, self.num_teeth, self.max_stages, 6)[:, :, :, 0, :]  # [B, max_stages, num_teeth, 6]
        ratios_sequence = torch.softmax(ratios, dim=1)  # Softmax across stages per tooth/parameter

        # Predict directions for all stages
        directions = self.direction_head(stage_outputs.view(-1, self.embed_dim))  # [B * max_stages * num_teeth, max_stages * 6]
        directions = directions.view(B, self.max_stages, self.num_teeth, self.max_stages, 6)[:, :, :, 0, :]  # [B, max_stages, num_teeth, 6]
        directions_sequence = directions  # Logits, no sigmoid applied

        # Update previous sequences with predictions
        prev_ratios_seq = [r.detach() for r in torch.unbind(ratios_sequence, dim=1)]
        prev_directions_seq = [d.detach() for d in torch.unbind(directions_sequence, dim=1)]

        # Enforce zero stage-wise transforms where cumulative transform is zero
        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)  # [B, 1, num_teeth, 6]
        ratios_sequence = ratios_sequence * (1 - cumulative_zero_mask)
        directions_sequence = directions_sequence * (1 - cumulative_zero_mask)

        logger.debug(f"Ratios sequence mean: {ratios_sequence.mean().item():.4f}")
        logger.debug(f"Directions sequence mean: {directions_sequence.mean().item():.4f}")

        return [ratios_sequence, directions_sequence]