import torch
import torch.nn as nn
import logging

class PerToothTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, num_teeth=14, max_stages=25, num_layers=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        
        # Positional embeddings for stages
        self.pos_embed = nn.Parameter(torch.zeros(1, max_stages, embed_dim))
        self.target_embed = nn.Linear(6, embed_dim)
        self.cumulative_embed = nn.Linear(6, embed_dim)
        
        # Cross-tooth attention to aggregate memory
        self.cross_tooth_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3, batch_first=True)
        
        # Per-tooth transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.3,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-4
        )
        self.tooth_decoders = nn.ModuleList([
            nn.TransformerDecoder(decoder_layer, num_layers=num_layers) for _ in range(num_teeth)
        ])
        # Enhanced MLP for transformation prediction
        self.activity = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Linear(128, 100),
            nn.BatchNorm1d(100, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 80),
            nn.Linear(80, 50),
            nn.BatchNorm1d(50, eps=1e-3),
            nn.ReLU()
        )

        # Shared prediction heads
        self.activity_head = nn.Sequential(
            nn.Linear(50, 1),  
            nn.Sigmoid()
        )
                
        self.param_activity_head = nn.Sequential(
            nn.Linear(50, 6),  
            nn.Sigmoid()
        )

        self.out_layer = nn.Linear(embed_dim, 6)
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Align with OrthoDGCNNModel's initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=0.01)
    
    def forward(self, memory, cumulative_transforms, num_stages=None, targets=None, use_teacher_forcing=False, training=False):
        logger = logging.getLogger('TrainLogger')
        B = memory.size(0)
        device = memory.device

        # Validate input shapes
        expected_memory_shape = (B, self.num_teeth, self.embed_dim)
        expected_cumulative_shape = (B, self.num_teeth, 6)
        if memory.shape != expected_memory_shape:
            logger.error(f"Invalid memory shape: got {memory.shape}, expected {expected_memory_shape}")
            raise RuntimeError(f"Memory shape mismatch")
        if cumulative_transforms.shape != expected_cumulative_shape:
            logger.error(f"Invalid cumulative_transforms shape: got {cumulative_transforms.shape}, expected {expected_cumulative_shape}")
            raise RuntimeError(f"Cumulative_transforms shape mismatch")

        # Handle NaNs
        memory = torch.nan_to_num(memory, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_transforms = torch.nan_to_num(cumulative_transforms, nan=0.0, posinf=1.0, neginf=-1.0)

        # Cross-tooth attention to enrich memory
        memory_key_padding_mask = torch.zeros(B, self.num_teeth, dtype=torch.bool, device=device)
        memory_attn, _ = self.cross_tooth_attention(memory, memory, memory, key_padding_mask=memory_key_padding_mask)
        memory = memory + memory_attn  # Residual connection

        # Cumulative embedding
        cumulative_embed = self.cumulative_embed(cumulative_transforms)  # [B, num_teeth, embed_dim]
        cumulative_embed = torch.nan_to_num(cumulative_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_embed = cumulative_embed.unsqueeze(1).expand(-1, self.max_stages, -1, -1)  # [B, max_stages, num_teeth, embed_dim]

        # Initialize outputs
        output_all = torch.zeros(B, self.max_stages, self.num_teeth, self.embed_dim, device=device)
        transforms_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, device=device)
        param_activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        param_activity_masks = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)  # New tensor to store masks

        # Process each tooth independently
        for tooth_idx in range(self.num_teeth):
            # Initialize previous transformation (zero for stage 0)
            prev_transform = torch.zeros(B, 6, device=device)  # [B, 6]

            for stage_idx in range(self.max_stages):
                # Prepare target embeddings
                if use_teacher_forcing and targets is not None and training and stage_idx > 0:
                    expected_targets_shape = (B, self.max_stages, self.num_teeth, 6)
                    if targets.shape != expected_targets_shape:
                        logger.error(f"Invalid targets shape: got {targets.shape}, expected {expected_targets_shape}")
                        raise RuntimeError(f"Targets shape mismatch")
                    # Use ground-truth transformation from previous stage
                    targets_prev = targets[:, stage_idx - 1, tooth_idx, :]  # [B, 6]
                    targets_prev = torch.nan_to_num(targets_prev, nan=0.0, posinf=1.0, neginf=-1.0)
                    embedded_target = self.target_embed(targets_prev)  # [B, embed_dim]
                else:
                    # Use model's previous prediction (or zero for stage 0)
                    embedded_target = self.target_embed(prev_transform)  # [B, embed_dim]

                # Add positional and cumulative embeddings
                pos_embed = self.pos_embed[:, stage_idx, :]  # [1, embed_dim]
                cum_embed = cumulative_embed[:, stage_idx, tooth_idx, :]  # [B, embed_dim]
                tgt = embedded_target + pos_embed + cum_embed  # [B, embed_dim]
                tgt = self.pre_norm(tgt.unsqueeze(1))  # [B, 1, embed_dim]

                # Prepare memory for this tooth (use all teeth's memory)
                tooth_memory = memory  # [B, num_teeth, embed_dim]

                # No causal mask needed since processing one stage at a time
                output = self.tooth_decoders[tooth_idx](tgt, tooth_memory, memory_key_padding_mask=memory_key_padding_mask)
                output = self.final_norm(output.squeeze(1))  # [B, embed_dim]

                # Store output
                output_all[:, stage_idx, tooth_idx, :] = output
                activity_mlp = self.activity(output)
                # Predict activity
                activity_logit = self.activity_head(activity_mlp).squeeze(-1)  # [B]
                activity_mask = (activity_logit > 0.5).float()  # [B]
                activity_logits[:, stage_idx, tooth_idx] = activity_logit

                # Predict parameter activity
                param_activity_logit = self.param_activity_head(activity_mlp)  # [B, 6]
                param_activity_mask = (param_activity_logit > 0.5).float() * activity_mask.unsqueeze(-1)  # [B, 6]
                param_activity_logits[:, stage_idx, tooth_idx, :] = param_activity_logit
                param_activity_masks[:, stage_idx, tooth_idx, :] = param_activity_mask  # Store the mask

                # Predict transformations
                transforms = self.out_layer(output)  # [B, 6]
                transforms = transforms * param_activity_mask  # Zero out inactive parameters
                transforms_sequence[:, stage_idx, tooth_idx, :] = transforms

                # Update previous transformation for next stage
                prev_transform = transforms.detach()  # Use model's prediction
                if use_teacher_forcing and targets is not None and training and stage_idx < self.max_stages - 1:
                    # Optionally use ground-truth for next iteration with probability
                    targets_next = targets[:, stage_idx, tooth_idx, :]  # [B, 6]
                    prev_transform = targets_next  # Override with ground-truth

        # Enforce zero stage-wise transforms if cumulative transform is zero
        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)  # [B, 1, num_teeth, 6]
        transforms_sequence = transforms_sequence * (1 - cumulative_zero_mask)

        # Adjust transforms to match cumulative transforms
        stage_mask = torch.ones(B, self.max_stages, 1, 1, device=device)
        if num_stages is not None:
            for i in range(B):
                assert isinstance(num_stages[i], (int, torch.Tensor)) and 0 <= num_stages[i] <= self.max_stages
                stage_mask[i, num_stages[i]:] = 0.0
        
        pred_sum = (transforms_sequence * stage_mask).sum(dim=1)  # [B, num_teeth, 6]
        residual = cumulative_transforms - pred_sum  # [B, num_teeth, 6]
        active_stages = stage_mask.sum(dim=1, keepdim=True).clamp(min=1e-4)  # [B, 1, 1, 1]
        residual_expanded = residual.unsqueeze(1).expand(-1, self.max_stages, -1, -1)  # [B, max_stages, num_teeth, 6]
        adjustment = (residual_expanded / active_stages) * stage_mask * param_activity_masks  # Use accumulated masks

        transforms_sequence = transforms_sequence + adjustment

        logger.debug(f"Activity mask mean: {activity_mask.mean().item():.4f}")
        logger.debug(f"Param activity mask mean: {param_activity_masks.mean().item():.4f}")
        logger.debug(f"Transforms sequence min: {transforms_sequence.min().item():.4f}, max: {transforms_sequence.max().item():.4f}")
        logger.debug(f"Consistency error: {((transforms_sequence * stage_mask).sum(dim=1) - cumulative_transforms).abs().mean().item():.4f}")

        return [transforms_sequence, activity_logits, param_activity_logits]