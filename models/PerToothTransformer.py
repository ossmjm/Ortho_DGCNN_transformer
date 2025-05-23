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
        self.activity_target_embed = nn.Linear(1, embed_dim)
        self.param_activity_target_embed = nn.Linear(6, embed_dim)
        
        # Cross-tooth attention to aggregate memory
        self.cross_tooth_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3, batch_first=True)
        
        # Attention to aggregate previous stage embeddings
        self.prev_targets_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3, batch_first=True)
        
        # Cumulative attention for stage-wise cumulative transforms
        self.cumulative_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3, batch_first=True)
        
        # Additional normalization before cumulative attention
        self.pre_cumulative_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        
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

        # Stage activity head
        self.stage_activity_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        self.activity_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3)
        )
        self.param_activity_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3)
        )
        self.transform_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 100), nn.BatchNorm1d(100), nn.ReLU(),
            nn.Linear(100, 80), nn.Linear(80, 50), nn.ReLU()
        )

        self.activity_head = nn.Sequential(
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.param_activity_head = nn.Sequential(
            nn.Linear(64, 6), nn.Sigmoid()
        )
        self.out_layer = nn.Linear(50, 6)
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # Learned stage weights for residual distribution
        self.stage_weights = nn.Parameter(torch.ones(1, max_stages, 1, 1))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=0.02)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Specific initialization for param_activity_target_embed
        nn.init.xavier_uniform_(self.param_activity_target_embed.weight, gain=0.1)
        if self.param_activity_target_embed.bias is not None:
            nn.init.zeros_(self.param_activity_target_embed.bias)
        # Specific initialization for cumulative_attention output projection
        nn.init.xavier_uniform_(self.cumulative_attention.out_proj.weight, gain=0.1)
        if self.cumulative_attention.out_proj.bias is not None:
            nn.init.zeros_(self.cumulative_attention.out_proj.bias)

    def _get_teacher_forcing_params(self, epoch, total_epochs, stage_idx, num_stages, val_loss=None, base_tf_prob=0.9):
        """Compute teacher forcing probability and number of previous stages."""
        if val_loss is not None:
            normalized_loss = min(val_loss / 0.1, 1.0)
            tf_prob = base_tf_prob * (1 - normalized_loss)
            tf_prob = max(0.3, tf_prob)
        else:
            progress = epoch / total_epochs
            tf_prob = max(0.3, base_tf_prob - 0.6 * progress)
        
        num_stages_to_use = min(stage_idx, num_stages.max().item() if num_stages is not None else stage_idx)
        progress = epoch / total_epochs
        if progress < 0.3:
            num_stages_to_use = min(num_stages_to_use, 5)
        elif progress < 0.6:
            num_stages_to_use = min(num_stages_to_use, 10)
        
        return tf_prob, num_stages_to_use

    def forward(self, memory, cumulative_transforms, num_stages=None, targets=None, activity_targets=None, param_activity_targets=None, use_teacher_forcing=False, training=False, epoch=0, total_epochs=100, val_loss=None):
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

        # Cross-tooth attention
        memory_key_padding_mask = torch.zeros(B, self.num_teeth, dtype=torch.bool, device=device)
        memory_attn, _ = self.cross_tooth_attention(memory, memory, memory, key_padding_mask=memory_key_padding_mask)
        memory = memory + memory_attn

        # Cumulative embedding
        cumulative_embed = self.cumulative_embed(cumulative_transforms)
        cumulative_embed = torch.nan_to_num(cumulative_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_embed = cumulative_embed.unsqueeze(1).expand(-1, self.max_stages, -1, -1)

        # Initialize outputs
        output_all = torch.zeros(B, self.max_stages, self.num_teeth, self.embed_dim, device=device)
        transforms_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, device=device)
        param_activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        param_activity_masks = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        stage_activity_logits = torch.zeros(B, self.max_stages, device=device)

        # Initialize sequences
        prev_transform_seq = torch.zeros(B, self.max_stages, 6, device=device)
        prev_activity_seq = torch.zeros(B, self.max_stages, 1, device=device)
        prev_param_activity_seq = torch.zeros(B, self.max_stages, 6, device=device)

        # Process each tooth
        for tooth_idx in range(self.num_teeth):
            for stage_idx in range(self.max_stages):
                # Get teacher forcing params (only relevant for training)
                tf_prob, num_stages_to_use = self._get_teacher_forcing_params(epoch, total_epochs, stage_idx, num_stages, val_loss)
                effective_tf_prob = min(tf_prob, use_teacher_forcing if isinstance(use_teacher_forcing, float) else 1.0)
                if num_stages is not None and training:
                    tf_mask = (stage_idx < num_stages).float()
                    use_tf = training and stage_idx > 0 and (torch.rand(B, device=device) < effective_tf_prob * tf_mask).any()
                else:
                    use_tf = False  # Disable teacher forcing during inference

                # Prepare target embeddings with scheduled sampling
                start_idx = max(0, stage_idx - num_stages_to_use)
                if use_tf and targets is not None:
                    targets_prev = targets[:, start_idx:stage_idx, tooth_idx, :]
                    predicted_prev = prev_transform_seq[:, start_idx:stage_idx, :]
                    mix_ratio = effective_tf_prob
                    targets_prev = mix_ratio * targets_prev + (1 - mix_ratio) * predicted_prev
                    embedded_target = self.target_embed(targets_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)
                    embedded_target = embedded_target.squeeze(1)
                else:
                    targets_prev = prev_transform_seq[:, start_idx:stage_idx, :]
                    if targets_prev.shape[1] == 0:
                        embedded_target = self.target_embed(torch.zeros(B, 6, device=device))
                    else:
                        embedded_target = self.target_embed(targets_prev)
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)
                        embedded_target = embedded_target.squeeze(1)

                # Activity target embeddings
                if use_tf and activity_targets is not None:
                    activity_prev = activity_targets[:, start_idx:stage_idx, tooth_idx].unsqueeze(-1)
                    predicted_activity_prev = prev_activity_seq[:, start_idx:stage_idx, :]
                    activity_prev = mix_ratio * activity_prev + (1 - mix_ratio) * predicted_activity_prev
                    embedded_activity = self.activity_target_embed(activity_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)
                    embedded_activity = embedded_activity.squeeze(1)
                else:
                    activity_prev = prev_activity_seq[:, start_idx:stage_idx, :]
                    if activity_prev.shape[1] == 0:
                        embedded_activity = self.activity_target_embed(torch.zeros(B, 1, device=device))
                    else:
                        embedded_activity = self.activity_target_embed(activity_prev)
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)
                        embedded_activity = embedded_activity.squeeze(1)

                # Param activity target embeddings
                if use_tf and param_activity_targets is not None:
                    param_activity_prev = param_activity_targets[:, start_idx:stage_idx, tooth_idx, :]
                    predicted_param_activity_prev = prev_param_activity_seq[:, start_idx:stage_idx, :]
                    param_activity_prev = mix_ratio * param_activity_prev + (1 - mix_ratio) * predicted_param_activity_prev
                    embedded_param_activity = self.param_activity_target_embed(param_activity_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)
                    embedded_param_activity = embedded_param_activity.squeeze(1)
                    # Log output for debugging
                    if torch.isnan(embedded_param_activity).any() or torch.isinf(embedded_param_activity).any():
                        logger.error(f"NaN/Inf in prev_targets_attention output at tooth {tooth_idx}, stage {stage_idx}")
                        raise RuntimeError("NaN/Inf detected in prev_targets_attention")
                    logger.debug(f"prev_targets_attention output: min={embedded_param_activity.min().item():.4f}, max={embedded_param_activity.max().item():.4f}")
                else:
                    param_activity_prev = prev_param_activity_seq[:, start_idx:stage_idx, :]
                    if param_activity_prev.shape[1] == 0:
                        zero_input = torch.zeros(B, 6, device=device)
                        embedded_param_activity = self.param_activity_target_embed(zero_input)
                        # Log output for debugging
                        if torch.isnan(embedded_param_activity).any() or torch.isinf(embedded_param_activity).any():
                            logger.error(f"NaN/Inf in param_activity_target_embed output at tooth {tooth_idx}, stage {stage_idx}")
                            raise RuntimeError("NaN/Inf detected in param_activity_target_embed")
                        logger.debug(f"param_activity_target_embed output: min={embedded_param_activity.min().item():.4f}, max={embedded_param_activity.max().item():.4f}")
                    else:
                        embedded_param_activity = self.param_activity_target_embed(param_activity_prev)
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)
                        embedded_param_activity = embedded_param_activity.squeeze(1)
                        # Log output for debugging
                        if torch.isnan(embedded_param_activity).any() or torch.isinf(embedded_param_activity).any():
                            logger.error(f"NaN/Inf in prev_targets_attention output at tooth {tooth_idx}, stage {stage_idx}")
                            raise RuntimeError("NaN/Inf detected in prev_targets_attention")
                        logger.debug(f"prev_targets_attention output: min={embedded_param_activity.min().item():.4f}, max={embedded_param_activity.max().item():.4f}")

                # Combine embeddings
                pos_embed = self.pos_embed[:, stage_idx, :]
                cum_embed = cumulative_embed[:, stage_idx, tooth_idx, :]
                tgt = embedded_target + embedded_activity + embedded_param_activity + pos_embed + cum_embed
                # Log combined embeddings
                if torch.isnan(tgt).any() or torch.isinf(tgt).any():
                    logger.error(f"NaN/Inf in combined embeddings at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in combined embeddings")
                logger.debug(f"Combined embeddings: min={tgt.min().item():.4f}, max={tgt.max().item():.4f}")
                tgt = self.pre_norm(tgt.unsqueeze(1))

                # Normalize inputs to cumulative attention
                tgt = self.pre_cumulative_norm(tgt)
                cumulative_embed_stage = self.pre_cumulative_norm(cumulative_embed[:, stage_idx])

                # Cumulative attention with scaling
                cum_attn, attn_weights = self.cumulative_attention(tgt, cumulative_embed_stage, cumulative_embed_stage)
                # Log attention weights
                if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
                    logger.error(f"NaN/Inf in cumulative_attention weights at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in cumulative_attention weights")
                logger.debug(f"Cumulative attention weights: min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}")
                # Scale attention output
                cum_attn = cum_attn * 0.1  # Additional scaling to prevent explosion
                tgt = tgt + cum_attn
                # Log after cumulative attention
                if torch.isnan(tgt).any() or torch.isinf(tgt).any():
                    logger.error(f"NaN/Inf after cumulative attention at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected after cumulative attention")
                logger.debug(f"Post-cumulative attention: min={tgt.min().item():.4f}, max={tgt.max().item():.4f}")

                # Decode
                output = self.tooth_decoders[tooth_idx](tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
                output = self.final_norm(output.squeeze(1))
                output_all[:, stage_idx, tooth_idx, :] = output
                # Log decoder output
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.error(f"NaN/Inf in decoder output at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in decoder output")
                logger.debug(f"Decoder output: min={output.min().item():.4f}, max={output.max().item():.4f}")

                # Stage activity prediction (using output from first tooth for simplicity)
                if tooth_idx == 0:
                    stage_activity_logit = self.stage_activity_mlp(output).squeeze(-1)
                    stage_activity_logits[:, stage_idx] = stage_activity_logit

                activity_mlp_out = self.activity_mlp(output)
                param_activity_mlp_out = self.param_activity_mlp(output)
                transform_mlp_out = self.transform_mlp(output)
                # Log MLP outputs
                if torch.isnan(activity_mlp_out).any() or torch.isinf(activity_mlp_out).any():
                    logger.error(f"NaN/Inf in activity_mlp output at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in activity_mlp")
                if torch.isnan(param_activity_mlp_out).any() or torch.isinf(param_activity_mlp_out).any():
                    logger.error(f"NaN/Inf in param_activity_mlp output at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in param_activity_mlp")
                if torch.isnan(transform_mlp_out).any() or torch.isinf(transform_mlp_out).any():
                    logger.error(f"NaN/Inf in transform_mlp output at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in transform_mlp")

                activity_logit = self.activity_head(activity_mlp_out).squeeze(-1)
                param_activity_logit = self.param_activity_head(param_activity_mlp_out)
                transforms = self.out_layer(transform_mlp_out)
                # Log final outputs
                if torch.isnan(activity_logit).any() or torch.isinf(activity_logit).any():
                    logger.error(f"NaN/Inf in activity_logit at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in activity_logit")
                if torch.isnan(param_activity_logit).any() or torch.isinf(param_activity_logit).any():
                    logger.error(f"NaN/Inf in param_activity_logit at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in param_activity_logit")
                if torch.isnan(transforms).any() or torch.isinf(transforms).any():
                    logger.error(f"NaN/Inf in transforms at tooth {tooth_idx}, stage {stage_idx}")
                    raise RuntimeError("NaN/Inf detected in transforms")

                activity_mask = (activity_logit > 0.5).float()
                activity_logits[:, stage_idx, tooth_idx] = activity_logit

                param_activity_mask = (param_activity_logit > 0.5).float() * activity_mask.unsqueeze(-1)
                param_activity_logits[:, stage_idx, tooth_idx, :] = param_activity_logit
                param_activity_masks[:, stage_idx, tooth_idx, :] = param_activity_mask

                transforms_sequence[:, stage_idx, tooth_idx, :] = transforms

                prev_transform_seq[:, stage_idx, :] = transforms.detach()
                prev_activity_seq[:, stage_idx, :] = activity_logit.detach().unsqueeze(-1)
                prev_param_activity_seq[:, stage_idx, :] = param_activity_logit.detach()
                if use_tf and stage_idx < self.max_stages - 1:
                    if targets is not None:
                        prev_transform_seq[:, stage_idx, :] = mix_ratio * targets[:, stage_idx, tooth_idx, :] + (1 - mix_ratio) * transforms.detach()
                    if activity_targets is not None:
                        prev_activity_seq[:, stage_idx, :] = mix_ratio * activity_targets[:, stage_idx, tooth_idx].unsqueeze(-1) + (1 - mix_ratio) * activity_logit.detach().unsqueeze(-1)
                    if param_activity_targets is not None:
                        prev_param_activity_seq[:, stage_idx, :] = mix_ratio * param_activity_targets[:, stage_idx, tooth_idx, :] + (1 - mix_ratio) * param_activity_logit.detach()

        # Apply stage activity mask
        stage_activity_mask = (stage_activity_logits > 0.5).float().unsqueeze(-1).unsqueeze(-1)
        transforms_sequence = transforms_sequence * stage_activity_mask
        activity_logits = activity_logits * stage_activity_mask.squeeze(-1)
        param_activity_logits = param_activity_logits * stage_activity_mask
        # Log after masking
        if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
            logger.error(f"NaN/Inf in transforms_sequence after masking")
            raise RuntimeError("NaN/Inf detected in transforms_sequence")

        # Enforce zero transforms for zero cumulative transforms
        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)
        transforms_sequence = transforms_sequence * (1 - cumulative_zero_mask)
        # Log after zero transform enforcement
        if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
            logger.error(f"NaN/Inf in transforms_sequence after zero transform enforcement")
            raise RuntimeError("NaN/Inf detected in transforms_sequence")

        # Non-uniform residual distribution
        stage_weights = torch.softmax(self.stage_weights, dim=1)
        stage_mask = torch.ones(B, self.max_stages, 1, 1, device=device)
        if num_stages is not None and training:
            for i in range(B):
                stage_mask[i, num_stages[i]:] = 0.0
        else:
            stage_mask = stage_activity_mask

        pred_sum = (transforms_sequence * stage_mask).sum(dim=1)
        residual = cumulative_transforms - pred_sum
        residual_expanded = residual.unsqueeze(1).expand(-1, self.max_stages, -1, -1)
        adjustment = residual_expanded * stage_weights * stage_mask * param_activity_masks * 0.5
        # Log residual adjustment
        if torch.isnan(adjustment).any() or torch.isinf(adjustment).any():
            logger.error(f"NaN/Inf in residual adjustment")
            raise RuntimeError("NaN/Inf detected in residual adjustment")

        transforms_sequence = transforms_sequence + adjustment
        # Log final transforms_sequence
        if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
            logger.error(f"NaN/Inf in final transforms_sequence")
            raise RuntimeError("NaN/Inf detected in final transforms_sequence")

        # Clip gradients for stability
        for p in self.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                p.grad.clamp_(-0.3, 0.3)  # Even tighter clipping

        logger.debug(f"Stage activity mask mean: {stage_activity_mask.mean().item():.4f}")
        logger.debug(f"Activity mask mean: {activity_mask.mean().item():.4f}")
        logger.debug(f"Param activity mask mean: {param_activity_masks.mean().item():.4f}")
        logger.debug(f"Transforms sequence min: {transforms_sequence.min().item():.4f}, max: {transforms_sequence.max().item():.4f}")
        logger.debug(f"Consistency error: {((transforms_sequence * stage_mask).sum(dim=1) - cumulative_transforms).abs().mean().item():.4f}")

        return [transforms_sequence, activity_logits, param_activity_logits, stage_activity_logits]