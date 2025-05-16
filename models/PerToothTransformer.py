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
        # New embeddings for activity and parameter activity
        self.activity_target_embed = nn.Linear(1, embed_dim)  # Activity is scalar (binary)
        self.param_activity_target_embed = nn.Linear(6, embed_dim)  # Param activity is vector of 6
        
        # Cross-tooth attention to aggregate memory
        self.cross_tooth_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3, batch_first=True)
        
        # Attention to aggregate previous stage embeddings
        self.prev_targets_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3, batch_first=True)
        
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=0.01)
    
    def _get_teacher_forcing_params(self, epoch, total_epochs, stage_idx):
        """Compute teacher forcing probability and number of previous stages based on epoch."""
        # Teacher forcing probability: Linear decay from 0.9 to 0.3
        progress = epoch / total_epochs
        tf_prob = max(0.3, 0.9 - 0.6 * progress)  # From 0.9 to 0.3
        
        # Curriculum for number of previous stages
        if progress < 0.1:  # First 10% of epochs
            num_stages = min(2, stage_idx)  # At least 2 stages
        elif progress < 0.3:  # 10–30%
            num_stages = min(5, stage_idx)  # Up to 5 stages
        elif progress < 0.6:  # 30–60%
            num_stages = min(10, stage_idx)  # Up to 10 stages
        else:  # 60–100%
            num_stages = stage_idx  # All previous stages
        
        return tf_prob, num_stages

    def forward(self, memory, cumulative_transforms, num_stages=None, targets=None, activity_targets=None, param_activity_targets=None, use_teacher_forcing=False, training=False, epoch=0, total_epochs=100):
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
        if targets is not None:
            expected_targets_shape = (B, self.max_stages, self.num_teeth, 6)
            if targets.shape != expected_targets_shape:
                logger.error(f"Invalid targets shape: got {targets.shape}, expected {expected_targets_shape}")
                raise RuntimeError(f"Targets shape mismatch")
        if activity_targets is not None:
            expected_activity_shape = (B, self.max_stages, self.num_teeth)
            if activity_targets.shape != expected_activity_shape:
                logger.error(f"Invalid activity_targets shape: got {activity_targets.shape}, expected {expected_activity_shape}")
                raise RuntimeError(f"Activity_targets shape mismatch")
        if param_activity_targets is not None:
            expected_param_activity_shape = (B, self.max_stages, self.num_teeth, 6)
            if param_activity_targets.shape != expected_param_activity_shape:
                logger.error(f"Invalid param_activity_targets shape: got {param_activity_targets.shape}, expected {expected_param_activity_shape}")
                raise RuntimeError(f"Param_activity_targets shape mismatch")

        # Handle NaNs
        memory = torch.nan_to_num(memory, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_transforms = torch.nan_to_num(cumulative_transforms, nan=0.0, posinf=1.0, neginf=-1.0)
        if targets is not None:
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=-1.0)
        if activity_targets is not None:
            activity_targets = torch.nan_to_num(activity_targets, nan=0.0, posinf=1.0, neginf=-1.0)
        if param_activity_targets is not None:
            param_activity_targets = torch.nan_to_num(param_activity_targets, nan=0.0, posinf=1.0, neginf=-1.0)

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
        param_activity_masks = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)

        # Initialize sequences for predicted values
        prev_transform_seq = torch.zeros(B, self.max_stages, 6, device=device)  # [B, max_stages, 6]
        prev_activity_seq = torch.zeros(B, self.max_stages, 1, device=device)  # [B, max_stages, 1]
        prev_param_activity_seq = torch.zeros(B, self.max_stages, 6, device=device)  # [B, max_stages, 6]

        # Process each tooth independently
        for tooth_idx in range(self.num_teeth):
            for stage_idx in range(self.max_stages):
                # Get teacher forcing probability and number of stages
                tf_prob, num_stages_to_use = self._get_teacher_forcing_params(epoch, total_epochs, stage_idx)
                # Combine with use_teacher_forcing (treat as probability if float)
                effective_tf_prob = min(tf_prob, use_teacher_forcing if isinstance(use_teacher_forcing, float) else 1.0)
                # Check if stage_idx is within true_num_stages for each sample
                if num_stages is not None:
                    tf_mask = (stage_idx < num_stages).float()  # [B]
                    use_tf = training and stage_idx > 0 and (torch.rand(B, device=device) < effective_tf_prob * tf_mask).any()
                else:
                    use_tf = training and stage_idx > 0 and torch.rand(1).item() < effective_tf_prob

                # Prepare target embeddings
                start_idx = max(0, stage_idx - num_stages_to_use)  # Use same start_idx for both cases
                if use_tf and targets is not None:
                    targets_prev = targets[:, start_idx:stage_idx, tooth_idx, :]  # [B, num_stages_to_use, 6]
                    embedded_target = self.target_embed(targets_prev)  # [B, num_stages_to_use, embed_dim]
                    # Aggregate using attention
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)  # [B, 1, embed_dim]
                    embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)  # [B, 1, embed_dim]
                    embedded_target = embedded_target.squeeze(1)  # [B, embed_dim]
                else:
                    # Use accumulated predicted transforms, limited to num_stages_to_use
                    targets_prev = prev_transform_seq[:, start_idx:stage_idx, :]  # [B, num_stages_to_use, 6]
                    if targets_prev.shape[1] == 0:  # Handle empty slice
                        embedded_target = self.target_embed(torch.zeros(B, 6, device=device))  # [B, embed_dim]
                    else:
                        embedded_target = self.target_embed(targets_prev)  # [B, num_stages_to_use, embed_dim]
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)  # [B, 1, embed_dim]
                        embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)  # [B, 1, embed_dim]
                        embedded_target = embedded_target.squeeze(1)  # [B, embed_dim]

                # Prepare activity target embeddings
                if use_tf and activity_targets is not None:
                    activity_prev = activity_targets[:, start_idx:stage_idx, tooth_idx].unsqueeze(-1)  # [B, num_stages_to_use, 1]
                    embedded_activity = self.activity_target_embed(activity_prev)  # [B, num_stages_to_use, embed_dim]
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)  # [B, 1, embed_dim]
                    embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)  # [B, 1, embed_dim]
                    embedded_activity = embedded_activity.squeeze(1)  # [B, embed_dim]
                else:
                    activity_prev = prev_activity_seq[:, start_idx:stage_idx, :]  # [B, num_stages_to_use, 1]
                    if activity_prev.shape[1] == 0:  # Handle empty slice
                        embedded_activity = self.activity_target_embed(torch.zeros(B, 1, device=device))  # [B, embed_dim]
                    else:
                        embedded_activity = self.activity_target_embed(activity_prev)  # [B, num_stages_to_use, embed_dim]
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)  # [B, 1, embed_dim]
                        embedded_activity = embedded_activity.squeeze(1)  # [B, embed_dim]

                # Prepare param activity target embeddings
                if use_tf and param_activity_targets is not None:
                    param_activity_prev = param_activity_targets[:, start_idx:stage_idx, tooth_idx, :]  # [B, num_stages_to_use, 6]
                    embedded_param_activity = self.param_activity_target_embed(param_activity_prev)  # [B, num_stages_to_use, embed_dim]
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)  # [B, 1, embed_dim]
                    embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)  # [B, 1, embed_dim]
                    embedded_param_activity = embedded_param_activity.squeeze(1)  # [B, embed_dim]
                else:
                    param_activity_prev = prev_param_activity_seq[:, start_idx:stage_idx, :]  # [B, num_stages_to_use, 6]
                    if param_activity_prev.shape[1] == 0:  # Handle empty slice
                        embedded_param_activity = self.param_activity_target_embed(torch.zeros(B, 6, device=device))  # [B, embed_dim]
                    else:
                        embedded_param_activity = self.param_activity_target_embed(param_activity_prev)  # [B, num_stages_to_use, embed_dim]
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)  # [B, 1, embed_dim]
                        embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)  # [B, 1, embed_dim]
                        embedded_param_activity = embedded_param_activity.squeeze(1)  # [B, embed_dim]

                # Combine embeddings
                pos_embed = self.pos_embed[:, stage_idx, :]  # [1, embed_dim]
                cum_embed = cumulative_embed[:, stage_idx, tooth_idx, :]  # [B, embed_dim]
                tgt = embedded_target + embedded_activity + embedded_param_activity + pos_embed + cum_embed  # [B, embed_dim]
                tgt = self.pre_norm(tgt.unsqueeze(1))  # [B, 1, embed_dim]

                # Prepare memory for this tooth
                tooth_memory = memory  # [B, num_teeth, embed_dim]

                # Decode
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
                param_activity_masks[:, stage_idx, tooth_idx, :] = param_activity_mask

                # Predict transformations
                transforms = self.out_layer(output)  # [B, 6]
                transforms = transforms * param_activity_mask  # Zero out inactive parameters
                transforms_sequence[:, stage_idx, tooth_idx, :] = transforms

                # Update previous sequences
                prev_transform_seq[:, stage_idx, :] = transforms.detach()
                prev_activity_seq[:, stage_idx, :] = activity_logit.detach().unsqueeze(-1)
                prev_param_activity_seq[:, stage_idx, :] = param_activity_logit.detach()
                if use_tf and stage_idx < self.max_stages - 1:
                    if targets is not None:
                        prev_transform_seq[:, stage_idx, :] = targets[:, stage_idx, tooth_idx, :]
                    if activity_targets is not None:
                        prev_activity_seq[:, stage_idx, :] = activity_targets[:, stage_idx, tooth_idx].unsqueeze(-1)
                    if param_activity_targets is not None:
                        prev_param_activity_seq[:, stage_idx, :] = param_activity_targets[:, stage_idx, tooth_idx, :]

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
        adjustment = (residual_expanded / active_stages) * stage_mask * param_activity_masks

        transforms_sequence = transforms_sequence + adjustment

        logger.debug(f"Activity mask mean: {activity_mask.mean().item():.4f}")
        logger.debug(f"Param activity mask mean: {param_activity_masks.mean().item():.4f}")
        logger.debug(f"Transforms sequence min: {transforms_sequence.min().item():.4f}, max: {transforms_sequence.max().item():.4f}")
        logger.debug(f"Consistency error: {((transforms_sequence * stage_mask).sum(dim=1) - cumulative_transforms).abs().mean().item():.4f}")

        return [transforms_sequence, activity_logits, param_activity_logits]