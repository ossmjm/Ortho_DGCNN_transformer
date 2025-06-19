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
        # self.direction_embed = nn.Linear(6, embed_dim)  # Embed binary directions
        self.activity_target_embed = nn.Linear(1, embed_dim)
        self.param_activity_target_embed = nn.Linear(6, embed_dim)
        
        # Normalization layers for attention inputs
        self.target_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.activity_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.param_activity_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.direction_norm = nn.LayerNorm(embed_dim, eps=1e-5)  # Norm for direction embeddings
        
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
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.activity_head = nn.Linear(64, 1)
        self.param_activity_head = nn.Linear(64, 6)
        self.out_layer = nn.Linear(32, 6)
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # # Stage weights MLP
        # self.stage_index_embed = nn.Parameter(torch.zeros(max_stages, embed_dim // 4))
        # self.stage_weights_mlp = nn.Sequential(
        #     nn.Linear(embed_dim // 4, embed_dim // 8),
        #     nn.GELU(),
        #     nn.Linear(embed_dim // 8, 1)
        # )
        # self.stage_weight_scale = nn.Parameter(torch.tensor(1.0))  # Learnable scale
        
        # Gradient scaling factor
        self.grad_scale = 0.1
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                if m is self.stage_weight_scale:
                    nn.init.constant_(m, 1.0)
                elif m is self.stage_index_embed:
                    nn.init.trunc_normal_(m, mean=0.0, std=0.02)
                else:
                    nn.init.trunc_normal_(m, std=0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for attn in [self.cross_tooth_attention, self.prev_targets_attention, self.cumulative_attention]:
            nn.init.xavier_uniform_(attn.out_proj.weight, gain=0.05)
            if attn.out_proj.bias is not None:
                nn.init.zeros_(attn.out_proj.bias)

    def _get_teacher_forcing_params(self, epoch, total_epochs, stage_idx, num_stages, val_loss=None, base_tf_prob=0.9):
        if val_loss is not None:
            normalized_loss = 1 - torch.exp(torch.tensor(-val_loss / 2.0))
            tf_prob = base_tf_prob * normalized_loss.item()
            tf_prob = min(max(tf_prob, 0.3), 0.95)
        else:
            progress = epoch / total_epochs
            tf_prob = max(0.3, base_tf_prob - 0.6 * progress)
        
        return tf_prob

    # def _get_stage_weights(self):
    #     # Compute stage weights using MLP
    #     stage_indices = self.stage_index_embed  # Shape: (max_stages, embed_dim // 4)
    #     stage_weights = self.stage_weights_mlp(stage_indices)  # Shape: (max_stages, 1)
    #     stage_weights = stage_weights.view(1, self.max_stages, 1, 1)  # Shape: (1, max_stages, 1, 1)
    #     stage_weights = torch.softmax(stage_weights, dim=1)  # Normalize across stages
    #     return stage_weights

    def forward(self, memory, cumulative_transforms, directions=None, num_stages=None, targets=None, activity_targets=None, param_activity_targets=None, use_teacher_forcing=False, training=False, epoch=0, total_epochs=100, val_loss=None):
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

        # Validate cumulative_transforms shape
        if cumulative_transforms.shape != (B, self.num_teeth, 6):
            logger.error(f"Invalid cumulative_transforms shape: got {cumulative_transforms.shape}, expected {(B, self.num_teeth, 6)}")
            raise RuntimeError("cumulative_transforms shape mismatch")

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

        # direction_embed = self.direction_embed(directions.float()) if directions is not None else torch.zeros(B, self.num_teeth, self.embed_dim, device=device)
        # direction_embed = self.direction_norm(direction_embed)
        # direction_embed = direction_embed.unsqueeze(1).expand(-1, self.max_stages, -1, -1)

        transforms_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, device=device)
        param_activity_logits = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        param_activity_masks = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        stage_activity_logits = torch.zeros(B, self.max_stages, device=device)

        # Initialize previous sequences as lists to avoid inplace modifications
        prev_transform_seq = [torch.zeros(B, 6, device=device) for _ in range(self.max_stages)]
        prev_activity_seq = [torch.zeros(B, 1, device=device) for _ in range(self.max_stages)]
        prev_param_activity_seq = [torch.zeros(B, 6, device=device) for _ in range(self.max_stages)]

        # Initialize teacher forcing counter
        tf_count = 0

        for tooth_idx in range(self.num_teeth):
            for stage_idx in range(self.max_stages):
                tf_prob = self._get_teacher_forcing_params(epoch, total_epochs, stage_idx, num_stages, val_loss)
                effective_tf_prob = min(tf_prob, use_teacher_forcing if isinstance(use_teacher_forcing, float) else 1.0)
                use_tf = training and stage_idx > 0 and stage_idx < num_stages.min().item() and (torch.rand(1, device=device) < effective_tf_prob).item() if num_stages is not None else False
                # Increment teacher forcing counter
                if use_tf:
                    tf_count += 1

                start_idx = 0  # Use all previous stages
                if use_tf and targets is not None:
                    targets_prev = targets[:, start_idx:stage_idx, tooth_idx, :]  # Shape: (B, stage_idx, 6)
                    embedded_target = self.target_embed(targets_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_target = self.target_norm(embedded_target)
                    if embedded_target.isnan().any():
                        logger.warning("NaN detected in embedded_target, replacing with zeros")
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
                            logger.warning("NaN detected in embedded_target, replacing with zeros")
                            embedded_target = torch.zeros_like(embedded_target)
                        embedded_target, _ = self.prev_targets_attention(query, embedded_target, embedded_target)
                        embedded_target = embedded_target.squeeze(1).clamp(0, 30)

                if use_tf and activity_targets is not None:
                    activity_prev = activity_targets[:, start_idx:stage_idx, tooth_idx].unsqueeze(-1)  # Shape: (B, stage_idx, 1)
                    embedded_activity = self.activity_target_embed(activity_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_activity = self.activity_norm(embedded_activity)
                    if embedded_activity.isnan().any():
                        logger.warning("NaN detected in embedded_activity, replacing with zeros")
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
                            logger.warning("NaN detected in embedded_activity, replacing with zeros")
                            embedded_activity = torch.zeros_like(embedded_activity)
                        embedded_activity, _ = self.prev_targets_attention(query, embedded_activity, embedded_activity)
                        embedded_activity = embedded_activity.squeeze(1).clamp(0, 30)

                if use_tf and param_activity_targets is not None:
                    param_activity_prev = param_activity_targets[:, start_idx:stage_idx, tooth_idx, :]  # Shape: (B, stage_idx, 6)
                    embedded_param_activity = self.param_activity_target_embed(param_activity_prev)
                    query = self.pos_embed[:, stage_idx, :].expand(B, 1, -1)
                    embedded_param_activity = self.param_activity_norm(embedded_param_activity)
                    if embedded_param_activity.isnan().any():
                        logger.warning("NaN detected in embedded_param_activity, replacing with zeros")
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
                            logger.warning("NaN detected in embedded_param_activity, replacing with zeros")
                            embedded_param_activity = torch.zeros_like(embedded_param_activity)
                        embedded_param_activity, _ = self.prev_targets_attention(query, embedded_param_activity, embedded_param_activity)
                        embedded_param_activity = embedded_param_activity.squeeze(1).clamp(0, 30)

                pos_embed = self.pos_embed[:, stage_idx, :]
                cum_embed = cumulative_embed[:, stage_idx, tooth_idx, :]
                # dir_embed = direction_embed[:, stage_idx, tooth_idx, :]
                tgt = embedded_target + 0.1 * embedded_activity + 0.1 * embedded_param_activity + pos_embed + cum_embed
                tgt = self.pre_norm(tgt.unsqueeze(1))

                tgt = self.pre_cumulative_norm(tgt)
                cumulative_embed_stage = self.pre_cumulative_norm(cumulative_embed[:, stage_idx])
                cum_attn, _ = self.cumulative_attention(tgt, cumulative_embed_stage, cumulative_embed_stage)
                cum_attn = cum_attn * 0.1
                tgt = tgt + cum_attn

                output = self.tooth_decoders[tooth_idx](tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
                output = self.final_norm(output.squeeze(1))

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

                # Update previous sequences without inplace operations
                prev_transform_seq[stage_idx] = transforms.detach()
                prev_activity_seq[stage_idx] = activity_logit.detach().unsqueeze(-1)
                prev_param_activity_seq[stage_idx] = param_activity_logit.detach()

                if use_tf and stage_idx < self.max_stages - 1:
                    if targets is not None:
                        prev_transform_seq[stage_idx] = targets[:, stage_idx, tooth_idx, :].detach()
                    if activity_targets is not None:
                        prev_activity_seq[stage_idx] = activity_targets[:, stage_idx, tooth_idx].unsqueeze(-1).detach()
                    if param_activity_targets is not None:
                        prev_param_activity_seq[stage_idx] = param_activity_targets[:, stage_idx, tooth_idx, :].detach()

        # Validate transforms_sequence shape
        if transforms_sequence.shape != (B, self.max_stages, self.num_teeth, 6):
            logger.error(f"Invalid transforms_sequence shape: got {transforms_sequence.shape}, expected {(B, self.max_stages, self.num_teeth, 6)}")
            raise RuntimeError("transforms_sequence shape mismatch")

        stage_activity_mask = (torch.sigmoid(stage_activity_logits) > 0.5).float().unsqueeze(-1).unsqueeze(-1)
        transforms_sequence = transforms_sequence * stage_activity_mask
        activity_logits = activity_logits * stage_activity_mask.squeeze(-1)
        param_activity_logits = param_activity_logits * stage_activity_mask

        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)
        transforms_sequence = transforms_sequence * (1 - cumulative_zero_mask)

        # # Residual adjustment per tooth and parameter
        # stage_weights = self._get_stage_weights()  # Shape: (1, max_stages, 1, 1)
        # logger.debug(f"Stage weights mean: {stage_weights.mean().item():.4f}, std: {stage_weights.std().item():.4f}")
        # # print(stage_weights)
        # # Dynamic stage_mask: use num_stages in training, stage_activity_logits in inference
        # stage_mask = torch.ones(B, self.max_stages, 1, 1, device=device)  # Shape: (B, 25, 1, 1)
        # if training and num_stages is not None:
        #     for i in range(B):
        #         stage_mask[i, num_stages[i]:] = 0.0
        # else:
        #     stage_activity_mask = (torch.sigmoid(stage_activity_logits) > 0.5).float().unsqueeze(-1).unsqueeze(-1)  # (B, 25, 1, 1)
        #     stage_mask = stage_mask * stage_activity_mask  # Shape: (B, 25, 1, 1)

        # # Validate stage_mask shape
        # if stage_mask.shape != (B, self.max_stages, 1, 1):
        #     logger.error(f"Invalid stage_mask shape: got {stage_mask.shape}, expected {(B, self.max_stages, 1, 1)}")
        #     raise RuntimeError("stage_mask shape mismatch")

        # # Log shapes for debugging
        # logger.debug(f"Residual adjustment shapes: stage_weights={stage_weights.shape}, stage_mask={stage_mask.shape}, "
        #              f"param_activity_masks={param_activity_masks.shape}")

        # # Initialize adjustment tensor
        # adjustment = torch.zeros_like(transforms_sequence)
        # # Parameter-specific clamping: translations (mm), rotations (degrees)
        # clamp_ranges_residual = torch.tensor([1.0, 1.0, 1.0, 10.0, 10.0, 10.0], device=device)  # Translations, rotations
        # clamp_ranges_adjustment = torch.tensor([0.5, 0.5, 0.5, 6.0, 6.0, 6.0], device=device)  # Relaxed clamping
        # for tooth_idx in range(self.num_teeth):
        #     for param_idx in range(6):
        #         masked_preds = transforms_sequence[:, :, tooth_idx, param_idx] * stage_mask.squeeze(-1).squeeze(-1)  # (B, 25)
        #         # Validate masked_preds shape
        #         if masked_preds.shape != (B, self.max_stages):
        #             logger.error(f"masked_preds shape {masked_preds.shape} does not match expected {(B, self.max_stages)}")
        #             raise RuntimeError("masked_preds shape mismatch")
                
        #         pred_sum = masked_preds.sum(dim=1)  # Shape: (B,)
        #         residual = cumulative_transforms[:, tooth_idx, param_idx] - pred_sum  # Shape: (B,)
                
        #         # Ensure residual is (B,)
        #         residual = residual.view(B)  # Explicitly reshape to (B,)
        #         if residual.shape != (B,):
        #             logger.error(f"Residual shape {residual.shape} does not match expected (B,)=({B},)")
        #             raise RuntimeError("Residual shape mismatch")

        #         # Log shapes for debugging
        #         logger.debug(f"Shapes for tooth {tooth_idx} param {param_idx}: "
        #                      f"masked_preds={masked_preds.shape}, "
        #                      f"cumulative_transforms[:, {tooth_idx}, {param_idx}]={cumulative_transforms[:, tooth_idx, param_idx].shape}, "
        #                      f"pred_sum={pred_sum.shape}, residual={residual.shape}")

        #         residual = torch.clamp(residual, -clamp_ranges_residual[param_idx], clamp_ranges_residual[param_idx])
        #         residual_expanded = residual.unsqueeze(1).unsqueeze(-1).expand(B, self.max_stages, 1)  # Shape: (B, 25, 1)
                
        #         # Log residual_expanded shape
        #         logger.debug(f"residual_expanded shape for tooth {tooth_idx} param {param_idx}: {residual_expanded.shape}")
                
        #         param_mask = param_activity_masks[:, :, tooth_idx, param_idx].unsqueeze(-1)  # Shape: (B, 25, 1)
        #         logger.debug(f"param_mask shape for tooth {tooth_idx} param {param_idx}: {param_mask.shape}")
        #         if param_mask.sum(dim=1).eq(0).any():
        #             logger.debug(f"No active stages for tooth {tooth_idx} param {param_idx}, skipping adjustment")
        #             continue
        #         residual_expanded_4d = residual_expanded.unsqueeze(-1)  # (B, 25, 1, 1)
        #         param_mask_4d = param_mask.unsqueeze(-1)  # (B, 25, 1, 1)
        #         stage_adjustment = residual_expanded_4d * stage_weights * stage_mask * param_mask_4d * self.stage_weight_scale

        #         logger.debug(f"stage_adjustment shape for tooth {tooth_idx} param {param_idx}: {stage_adjustment.shape}")
        #         stage_adjustment = torch.clamp(stage_adjustment, -clamp_ranges_adjustment[param_idx], clamp_ranges_adjustment[param_idx])
                
        #         adjustment[:, :, tooth_idx, param_idx] = stage_adjustment.squeeze(-1).squeeze(-1)  # (B, 25)
                
        #         if logger.isEnabledFor(logging.DEBUG):
        #             logger.debug(f"Residual tooth {tooth_idx} param {param_idx}: "
        #                          f"residual_mean={residual.mean().item():.4f}, "
        #                          f"adjustment_mean={stage_adjustment.mean().item():.4f}, "
        #                          f"pred_sum_mean={pred_sum.mean().item():.4f}")

        # # Apply adjustment to transforms_sequence
        # transforms_sequence = transforms_sequence + adjustment

        # # Log overall adjustment statistics
        # logger.debug(f"Residual adjustment: mean={adjustment.mean().item():.4f}, max={adjustment.max().item():.4f}, "
        #              f"min={adjustment.min().item():.4f}")

        # Post-adjustment validation check
        # if training:
        #     for tooth_idx in range(self.num_teeth):
        #         for param_idx in range(6):
        #             final_sum = (transforms_sequence[:, :, tooth_idx, param_idx] * stage_mask.squeeze(-1).squeeze(-1)).sum(dim=1)
        #             error = torch.abs(final_sum - cumulative_transforms[:, tooth_idx, param_idx]).mean()
        #             if error > 1:
        #                 logger.warning(f"Post-adjustment tooth {tooth_idx} param {param_idx}: error={error.item():.4f}"
        #                                 f"clamping may be too restrictive")

        # Gradient clipping
        for p in self.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                p.grad.clamp_(-0.3, 0.3)

        return [transforms_sequence, activity_logits, param_activity_logits, stage_activity_logits, tf_count]