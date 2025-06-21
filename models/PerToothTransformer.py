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
        self.direction_embed = nn.Linear(6, embed_dim)
        self.activity_target_embed = nn.Linear(1, embed_dim)
        self.param_activity_target_embed = nn.Linear(6, embed_dim)
        
        # Normalization layers for attention inputs
        self.target_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.activity_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.param_activity_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.direction_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        
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

        # Enhanced MLP heads
        self.activity_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU()
        )
        self.param_activity_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU()
        )
        self.transform_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 6), nn.ReLU()
        )

        self.activity_head = nn.Linear(64, 1)
        self.param_activity_head = nn.Linear(64, 6)
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # Gradient scaling factor
        self.grad_scale = 0.1
        
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
        for attn in [self.cross_tooth_attention, self.prev_targets_attention, self.cumulative_attention]:
            nn.init.xavier_uniform_(attn.out_proj.weight, gain=1.0)
            if attn.out_proj.bias is not None:
                nn.init.zeros_(attn.out_proj.bias)

    def _get_teacher_forcing_params(self, epoch, total_epochs, stage_idx, num_stages, val_loss=None, base_tf_prob=0.9):
        logger = logging.getLogger('TrainLogger')
        if val_loss is not None:
            normalized_loss = 1 - torch.exp(torch.tensor(-val_loss / 2.0))
            tf_prob = base_tf_prob * normalized_loss.item()
            tf_prob = min(max(tf_prob, 0.3), 0.95)
        else:
            progress = epoch / total_epochs
            tf_prob = max(0.3, base_tf_prob - 0.6 * progress)
        logger.debug(f"Teacher forcing prob: epoch={epoch}, stage_idx={stage_idx}, tf_prob={tf_prob:.4f}")
        return tf_prob

    def forward(self, memory, cumulative_transforms, directions=None, num_stages=None, targets=None, activity_targets=None, param_activity_targets=None, use_teacher_forcing=True, training=False, epoch=0, total_epochs=100, val_loss=None):
        logger = logging.getLogger('TrainLogger')
        B = memory.size(0)
        device = memory.device

        expected_shapes = {
            'memory': (B, self.num_teeth, self.embed_dim),
            'cumulative_transforms': (B, self.num_teeth, 6),
            'directions': (B, self.num_teeth, 6) if directions is not None else None,
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

        # Pad param_activity_targets to max_stages if necessary
        if param_activity_targets is not None and param_activity_targets.shape[1] != self.max_stages:
            logger.warning(f"param_activity_targets has {param_activity_targets.shape[1]} stages, padding to {self.max_stages}")
            padding = torch.zeros(B, self.max_stages - param_activity_targets.shape[1], self.num_teeth, 6, device=device)
            param_activity_targets = torch.cat([param_activity_targets, padding], dim=1)

        # Log input shapes for debugging
        logger.debug(f"Input shapes: memory={memory.shape}, cumulative_transforms={cumulative_transforms.shape}, "
                     f"directions={directions.shape if directions is not None else None}, "
                     f"num_stages={num_stages.tolist() if num_stages is not None else None}, "
                     f"targets={targets.shape if targets is not None else None}, "
                     f"activity_targets={activity_targets.shape if activity_targets is not None else None}, "
                     f"param_activity_targets={param_activity_targets.shape if param_activity_targets is not None else None}")

        tensors = [memory, cumulative_transforms, directions, targets, activity_targets, param_activity_targets]
        for tensor in tensors:
            if tensor is not None:
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        memory_key_padding_mask = torch.zeros(B, self.num_teeth, dtype=torch.bool, device=device)
        memory_attn, _ = self.cross_tooth_attention(memory, memory, memory, key_padding_mask=memory_key_padding_mask)
        memory = memory + memory_attn.clamp(0, 30)

        cumulative_embed = self.cumulative_embed(cumulative_transforms.float())
        cumulative_embed = torch.nan_to_num(cumulative_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_embed = cumulative_embed.unsqueeze(1).expand(-1, self.max_stages, -1, -1)

        direction_embed = self.direction_embed(directions.float()) if directions is not None else torch.zeros(B, self.num_teeth, self.embed_dim, device=device)
        direction_embed = self.direction_norm(direction_embed)
        direction_embed = direction_embed.unsqueeze(1).expand(-1, self.max_stages, -1, -1)

        transforms_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, device=device)
        param_activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        param_activity_masks = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)

        # Initialize previous sequences as lists to avoid inplace modifications
        prev_transform_seq = [torch.zeros(B, 6, device=device) for _ in range(self.max_stages)]
        prev_activity_seq = [torch.zeros(B, 1, device=device) for _ in range(self.max_stages)]
        prev_param_activity_seq = [torch.zeros(B, 6, device=device) for _ in range(self.max_stages)]

        # Initialize teacher forcing counter
        tf_count = 0

        for tooth_idx in range(self.num_teeth):
            for stage_idx in range(self.max_stages):
                # Single teacher forcing probability for all target types
                tf_prob = self._get_teacher_forcing_params(
                    epoch, total_epochs, stage_idx, num_stages, val_loss)
                effective_tf_prob = min(tf_prob, use_teacher_forcing if isinstance(use_teacher_forcing, float) else True)
                batch_tf_mask = (torch.rand(B, device=device) < effective_tf_prob)
                use_tf = training and batch_tf_mask.any().item()
                if use_tf:
                    tf_count += batch_tf_mask.sum().item()
                    logger.debug(f"Applying teacher forcing for tooth {tooth_idx}, stage {stage_idx}, "
                                 f"batch_tf_mask={batch_tf_mask.sum().item()}/{B}")

                start_idx = 0
                if use_tf and targets is not None:
                    targets_prev = targets[:, start_idx:stage_idx, tooth_idx, :]  # Shape: (B, stage_idx, 6)
                    embedded_target = self.target_embed(targets_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_target = self.target_norm(embedded_target)
                    if embedded_target.isnan().any():
                        logger.warning(f"NaN in embedded_target for tooth {tooth_idx}, stage {stage_idx}, replacing with zeros")
                        embedded_target = torch.zeros_like(embedded_target)
                    embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)
                    embedded_target = embedded_target.squeeze(1).clamp(0, 30)
                else:
                    predicted_prev = torch.stack(prev_transform_seq[start_idx:stage_idx], dim=1) if stage_idx > start_idx else torch.zeros(B, 0, 6, device=device)
                    embedded_target = self.target_embed(predicted_prev.float() if predicted_prev.shape[1] > 0 else torch.zeros(B, 6, device=device).float())
                    if predicted_prev.shape[1] > 0:
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_target = self.target_norm(embedded_target)
                        if embedded_target.isnan().any():
                            logger.warning(f"NaN in embedded_target for tooth {tooth_idx}, stage {stage_idx}, replacing with zeros")
                            embedded_target = torch.zeros_like(embedded_target)
                        embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)
                        embedded_target = embedded_target.squeeze(1).clamp(0, 30)

                if use_tf and activity_targets is not None:
                    activity_prev = activity_targets[:, start_idx:stage_idx, tooth_idx].unsqueeze(-1)
                    embedded_activity = self.activity_target_embed(activity_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_activity = self.activity_norm(embedded_activity)
                    if embedded_activity.isnan().any():
                        logger.warning(f"NaN in embedded_activity for tooth {tooth_idx}, stage {stage_idx}, replacing with zeros")
                        embedded_activity = torch.zeros_like(embedded_activity)
                    embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)
                    embedded_activity = embedded_activity.squeeze(1).clamp(0, 30)
                else:
                    predicted_activity_prev = torch.stack(prev_activity_seq[start_idx:stage_idx], dim=1) if stage_idx > start_idx else torch.zeros(B, 0, 1, device=device)
                    embedded_activity = self.activity_target_embed(predicted_activity_prev.float() if predicted_activity_prev.shape[1] > 0 else torch.zeros(B, 1, device=device).float())
                    if predicted_activity_prev.shape[1] > 0:
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_activity = self.activity_norm(embedded_activity)
                        if embedded_activity.isnan().any():
                            logger.warning(f"NaN in embedded_activity for tooth {tooth_idx}, stage {stage_idx}, replacing with zeros")
                            embedded_activity = torch.zeros_like(embedded_activity)
                        embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)
                        embedded_activity = embedded_activity.squeeze(1).clamp(0, 30)

                if use_tf and param_activity_targets is not None:
                    param_activity_prev = param_activity_targets[:, start_idx:stage_idx, tooth_idx, :]
                    embedded_param_activity = self.param_activity_target_embed(param_activity_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_param_activity = self.param_activity_norm(embedded_param_activity)
                    if embedded_param_activity.isnan().any():
                        logger.warning(f"NaN in embedded_param_activity for tooth {tooth_idx}, stage {stage_idx}, replacing with zeros")
                        embedded_param_activity = torch.zeros_like(embedded_param_activity)
                    embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)
                    embedded_param_activity = embedded_param_activity.squeeze(1).clamp(0, 30)
                else:
                    predicted_param_activity_prev = torch.stack(prev_param_activity_seq[start_idx:stage_idx], dim=1) if stage_idx > start_idx else torch.zeros(B, 0, 6, device=device)
                    embedded_param_activity = self.param_activity_target_embed(predicted_param_activity_prev.float() if predicted_param_activity_prev.shape[1] > 0 else torch.zeros(B, 6, device=device).float())
                    if predicted_param_activity_prev.shape[1] > 0:
                        query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                        embedded_param_activity = self.param_activity_norm(embedded_param_activity)
                        if embedded_param_activity.isnan().any():
                            logger.warning(f"NaN in embedded_param_activity for tooth {tooth_idx}, stage {stage_idx}, replacing with zeros")
                            embedded_param_activity = torch.zeros_like(embedded_param_activity)
                        embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)
                        embedded_param_activity = embedded_param_activity.squeeze(1).clamp(0, 30)

                pos_embed = self.pos_embed[:, stage_idx, :]
                cum_embed = cumulative_embed[:, stage_idx, tooth_idx, :]
                dir_embed = direction_embed[:, stage_idx, tooth_idx, :]
                tgt = embedded_target + 0.1 * embedded_activity + 0.1 * embedded_param_activity + pos_embed + cum_embed + 0.1 * dir_embed
                tgt = self.pre_norm(tgt.unsqueeze(1))

                tgt = self.pre_cumulative_norm(tgt)
                cumulative_embed_stage = self.pre_cumulative_norm(cumulative_embed[:, stage_idx])
                cum_attn, _ = self.cumulative_attention(tgt, cumulative_embed_stage, cumulative_embed_stage)
                cum_attn = cum_attn * 0.1
                tgt = tgt + cum_attn

                output = self.tooth_decoders[tooth_idx](tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
                output = self.final_norm(output.squeeze(1))

                activity_mlp_out = self.activity_mlp(output)
                param_activity_mlp_out = self.param_activity_mlp(output)
                transforms = self.transform_mlp(output)
                transforms = torch.nan_to_num(transforms, nan=0.0, posinf=1.0, neginf=0.0)  # Ensure numerical stability

                activity_logit = self.activity_head(activity_mlp_out).squeeze(-1)
                param_activity_logit = self.param_activity_head(param_activity_mlp_out)

                activity_mask = torch.sigmoid(activity_logit) > 0.5
                activity_logits[:, stage_idx, tooth_idx] = activity_logit

                param_activity_mask = (torch.sigmoid(param_activity_logit) > 0.5).float() * activity_mask.unsqueeze(-1)
                param_activity_logits[:, stage_idx, tooth_idx, :] = param_activity_logit
                param_activity_masks[:, stage_idx, tooth_idx, :] = param_activity_mask

                transforms_sequence[:, stage_idx, tooth_idx, :] = transforms

                # Update previous sequences without inplace operations
                prev_transform_seq[stage_idx] = transforms.detach()
                prev_activity_seq[stage_idx] = activity_logit.detach().unsqueeze(-1)
                prev_param_activity_seq[stage_idx] = param_activity_logit.detach()

                # Teacher forcing update using non-inplace operation
                if use_tf and stage_idx < self.max_stages - 1 and targets is not None:
                    tf_updates = targets[batch_tf_mask, stage_idx, tooth_idx, :].detach()
                    new_transform_seq = prev_transform_seq[stage_idx].clone()
                    new_transform_seq[batch_tf_mask] = tf_updates
                    prev_transform_seq[stage_idx] = new_transform_seq
                if use_tf and stage_idx < self.max_stages - 1 and activity_targets is not None:
                    tf_updates = activity_targets[batch_tf_mask, stage_idx, tooth_idx].unsqueeze(-1).detach()
                    new_activity_seq = prev_activity_seq[stage_idx].clone()
                    new_activity_seq[batch_tf_mask] = tf_updates
                    prev_activity_seq[stage_idx] = new_activity_seq
                if use_tf and stage_idx < self.max_stages - 1 and param_activity_targets is not None:
                    tf_updates = param_activity_targets[batch_tf_mask, stage_idx, tooth_idx, :].detach()
                    new_param_activity_seq = prev_param_activity_seq[stage_idx].clone()
                    new_param_activity_seq[batch_tf_mask] = tf_updates
                    prev_param_activity_seq[stage_idx] = new_param_activity_seq

        # Validate transforms_sequence shape
        if transforms_sequence.shape != (B, self.max_stages, self.num_teeth, 6):
            logger.error(f"Invalid transforms_sequence shape: got {transforms_sequence.shape}, expected {(B, self.max_stages, self.num_teeth, 6)}")
            raise RuntimeError("transforms_sequence shape mismatch")

        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)
        transforms_sequence = transforms_sequence * (1 - cumulative_zero_mask)

        logger.debug(f"Output stats: transforms_mean={transforms_sequence.mean().item():.4f}, "
                     f"transforms_min={transforms_sequence.min().item():.4f}, "
                     f"transforms_max={transforms_sequence.max().item():.4f}, "
                     f"activity_logits_mean={activity_logits.mean().item():.4f}, "
                     f"param_activity_logits_mean={param_activity_logits.mean().item():.4f}")

        return [transforms_sequence, activity_logits, param_activity_logits, tf_count]