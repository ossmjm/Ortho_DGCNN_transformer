import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

class PerToothTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, num_teeth=14, max_stages=25, num_layers=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_stages, embed_dim))
        self.target_embed = nn.Linear(6, embed_dim)
        self.cumulative_embed = nn.Linear(6, embed_dim)
        self.activity_target_embed = nn.Linear(1, embed_dim)
        self.param_activity_target_embed = nn.Linear(6, embed_dim)
        
        # Cross-tooth attention
        self.cross_tooth_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.2, batch_first=True)
        
        # Previous targets attention
        self.prev_targets_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.2, batch_first=True)
        
        # Cumulative attention
        self.cumulative_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.2, batch_first=True)
        self.pre_cumulative_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        
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

        # Simplified MLP heads
        self.stage_activity_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        self.activity_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU()
        )
        self.param_activity_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU()
        )
        self.transform_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU()
        )

        self.activity_head = nn.Linear(64, 1)
        self.param_activity_head = nn.Linear(64, 6)
        self.out_layer = nn.Linear(32, 6)
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # Stage weights
        self.stage_weights = nn.Parameter(torch.ones(1, max_stages, 1, 1))
        
        # Gradient scaling factor
        self.grad_scale = 0.1
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain for stability
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Specific initialization for attention output projections
        for attn in [self.cross_tooth_attention, self.prev_targets_attention, self.cumulative_attention]:
            nn.init.xavier_uniform_(attn.out_proj.weight, gain=0.1)
            if attn.out_proj.bias is not None:
                nn.init.zeros_(attn.out_proj.bias)

    def _get_teacher_forcing_params(self, epoch, total_epochs, stage_idx, num_stages, val_loss=None, base_tf_prob=0.9):
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

        # Input validation
        expected_shapes = {
            'memory': (B, self.num_teeth, self.embed_dim),
            'cumulative_transforms': (B, self.num_teeth, 6),
            'targets': (B, self.max_stages, self.num_teeth, 6) if targets is not None else None,
            'activity_targets': (B, self.max_stages, self.num_teeth) if activity_targets is not None else None,
            'param_activity_targets': (B, self.max_stages, self.num_teeth, 6) if param_activity_targets is not None else None
        }
        for name, expected in expected_shapes.items():
            if expected is None:
                continue
            tensor = locals()[name]
            if tensor.shape != expected:
                logger.error(f"Invalid {name} shape: got {tensor.shape}, expected {expected}")
                raise RuntimeError(f"{name} shape mismatch")

        # Handle NaNs
        tensors = [memory, cumulative_transforms, targets, activity_targets, param_activity_targets]
        for tensor in tensors:
            if tensor is not None:
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        # Cross-tooth attention
        memory_key_padding_mask = torch.zeros(B, self.num_teeth, dtype=torch.bool, device=device)
        memory_attn, _ = self.cross_tooth_attention(memory, memory, memory, key_padding_mask=memory_key_padding_mask)
        memory = memory + memory_attn.clamp(-10, 10)

        # Cumulative embedding
        with torch.amp.autocast(device_type='cuda', enabled=False):
            cumulative_embed = self.cumulative_embed(cumulative_transforms.float())
        cumulative_embed = torch.nan_to_num(cumulative_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_embed = cumulative_embed.unsqueeze(1).expand(-1, self.max_stages, -1, -1)

        # Initialize outputs
        transforms_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, device=device)
        param_activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        param_activity_masks = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        stage_activity_logits = torch.zeros(B, self.max_stages, device=device)

        # Initialize sequences
        prev_transform_seq = torch.zeros(B, self.max_stages, 6, device=device)
        prev_activity_seq = torch.zeros(B, self.max_stages, 1, device=device)
        prev_param_activity_seq = torch.zeros(B, self.max_stages, 6, device=device)

        # Process each tooth and stage
        for tooth_idx in range(self.num_teeth):
            for stage_idx in range(self.max_stages):
                tf_prob, num_stages_to_use = self._get_teacher_forcing_params(epoch, total_epochs, stage_idx, num_stages, val_loss)
                effective_tf_prob = min(tf_prob, use_teacher_forcing if isinstance(use_teacher_forcing, float) else 1.0)
                use_tf = training and stage_idx > 0 and (torch.rand(B, device=device) < effective_tf_prob).any()

                # Prepare target embeddings
                start_idx = max(0, stage_idx - num_stages_to_use)
                if use_tf and targets is not None:
                    targets_prev = targets[:, start_idx:stage_idx, tooth_idx, :]
                    predicted_prev = prev_transform_seq[:, start_idx:stage_idx, :]
                    mix_ratio = effective_tf_prob
                    targets_prev = mix_ratio * targets_prev + (1 - mix_ratio) * predicted_prev
                    embedded_target = self.target_embed(targets_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)
                    embedded_target = embedded_target.squeeze(1).clamp(-10, 10)
                else:
                    targets_prev = prev_transform_seq[:, start_idx:stage_idx, :]
                    with torch.amp.autocast(device_type = 'cuda', enabled=False):
                        embedded_target = self.target_embed(targets_prev.float() if targets_prev.shape[1] > 0 else torch.zeros(B, 6, device=device).float())
                    if targets_prev.shape[1] > 0:
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)
                        embedded_target = embedded_target.squeeze(1).clamp(-10, 10)

                # Activity target embeddings
                if use_tf and activity_targets is not None:
                    activity_prev = activity_targets[:, start_idx:stage_idx, tooth_idx].unsqueeze(-1)
                    predicted_activity_prev = prev_activity_seq[:, start_idx:stage_idx, :]
                    activity_prev = mix_ratio * activity_prev + (1 - mix_ratio) * predicted_activity_prev
                    embedded_activity = self.activity_target_embed(activity_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)
                    embedded_activity = embedded_activity.squeeze(1).clamp(-10, 10)
                else:
                    activity_prev = prev_activity_seq[:, start_idx:stage_idx, :]
                    with torch.amp.autocast(device_type='cuda',enabled=False):
                        embedded_activity = self.activity_target_embed(activity_prev.float() if activity_prev.shape[1] > 0 else torch.zeros(B, 1, device=device).float())
                    if activity_prev.shape[1] > 0:
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)
                        embedded_activity = embedded_activity.squeeze(1).clamp(-10, 10)

                # Param activity target embeddings
                if use_tf and param_activity_targets is not None:
                    param_activity_prev = param_activity_targets[:, start_idx:stage_idx, tooth_idx, :]
                    predicted_param_activity_prev = prev_param_activity_seq[:, start_idx:stage_idx, :]
                    param_activity_prev = mix_ratio * param_activity_prev + (1 - mix_ratio) * predicted_param_activity_prev
                    embedded_param_activity = self.param_activity_target_embed(param_activity_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)
                    embedded_param_activity = embedded_param_activity.squeeze(1).clamp(-10, 10)
                else:
                    param_activity_prev = prev_param_activity_seq[:, start_idx:stage_idx, :]
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        embedded_param_activity = self.param_activity_target_embed(param_activity_prev.float() if param_activity_prev.shape[1] > 0 else torch.zeros(B, 6, device=device).float())
                    if param_activity_prev.shape[1] > 0:
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)
                        embedded_param_activity = embedded_param_activity.squeeze(1).clamp(-10, 10)

                # Combine embeddings with residual connection
                pos_embed = self.pos_embed[:, stage_idx, :]
                cum_embed = cumulative_embed[:, stage_idx, tooth_idx, :]
                tgt = embedded_target + 0.1 * embedded_activity + 0.1 * embedded_param_activity + pos_embed + cum_embed
                tgt = self.pre_norm(tgt.unsqueeze(1))

                # Cumulative attention
                tgt = self.pre_cumulative_norm(tgt)
                cumulative_embed_stage = self.pre_cumulative_norm(cumulative_embed[:, stage_idx])
                cum_attn, _ = self.cumulative_attention(tgt, cumulative_embed_stage, cumulative_embed_stage)
                cum_attn = cum_attn.clamp(-10, 10) * 0.1
                tgt = tgt + cum_attn

                # Decode
                output = self.tooth_decoders[tooth_idx](tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
                output = self.final_norm(output.squeeze(1))

                # Predictions
                if tooth_idx == 0:
                    stage_activity_logit = self.stage_activity_mlp(output).squeeze(-1)
                    stage_activity_logits[:, stage_idx] = stage_activity_logit

                activity_mlp_out = self.activity_mlp(output)
                param_activity_mlp_out = self.param_activity_mlp(output)
                transform_mlp_out = self.transform_mlp(output)

                activity_logit = self.activity_head(activity_mlp_out).squeeze(-1)
                param_activity_logit = self.param_activity_head(param_activity_mlp_out)
                transforms = self.out_layer(transform_mlp_out)

                activity_mask = torch.sigmoid(activity_logit) > 0.5
                activity_logits[:, stage_idx, tooth_idx] = activity_logit

                param_activity_mask = (torch.sigmoid(param_activity_logit) > 0.5).float() * activity_mask.unsqueeze(-1)
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
        stage_activity_mask = (torch.sigmoid(stage_activity_logits) > 0.5).float().unsqueeze(-1).unsqueeze(-1)
        transforms_sequence = transforms_sequence * stage_activity_mask
        activity_logits = activity_logits * stage_activity_mask.squeeze(-1)
        param_activity_logits = param_activity_logits * stage_activity_mask

        # Enforce zero transforms for zero cumulative transforms
        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)
        transforms_sequence = transforms_sequence * (1 - cumulative_zero_mask)

        # Residual adjustment
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
        transforms_sequence = transforms_sequence + adjustment

        # Gradient clipping and scaling
        for p in self.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                p.grad.clamp_(-0.3, 0.3)

        # # Debug logging
        # logger.debug(f"Stage activity mask mean: {stage_activity_mask.mean().item():.4f}")
        # logger.debug(f"Activity mask mean: {activity_mask.mean().item():.4f}")
        # logger.debug(f"Param activity mask mean: {param_activity_masks.mean().item():.4f}")
        # logger.debug(f"Transforms sequence min: {transforms_sequence.min().item():.4f}, max: {transforms_sequence.max().item():.4f}")
        # logger.debug(f"Consistency error: {((transforms_sequence * stage_mask).sum(dim=1) - cumulative_transforms).abs().mean().item():.4f}")

        return [transforms_sequence, activity_logits, param_activity_logits, stage_activity_logits]