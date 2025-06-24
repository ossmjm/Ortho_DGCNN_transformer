import torch
import torch.nn as nn
import logging

class GRUDecoder(nn.Module):
    def __init__(self, embed_dim, num_teeth, max_stages, num_layers=1, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        
        # Positional embeddings for stages
        self.pos_embed = nn.Parameter(torch.zeros(1, max_stages, embed_dim))
        self.cumulative_embed = nn.Linear(6, embed_dim)
        self.ratio_embed = nn.Linear(6, embed_dim)  # Separate embedding for teacher-forced ratios
        self.direction_embed = nn.Linear(6, embed_dim)  # Separate embedding for teacher-forced directions
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=0.3 if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Prediction heads
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
        
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        self.tooth_pos_embed = nn.Parameter(torch.zeros(1, num_teeth, embed_dim))
        
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
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _get_teacher_forcing_params(self, epoch, total_epochs, stage_idx, val_loss=None, base_tf_prob=0.95):
        logger = logging.getLogger('TrainLogger')
        min_tf_prob = 0.3
        min_loss = 0.05
        max_loss = 1.0
        
        if val_loss is not None:
            # Linear normalization of val_loss
            normalized_loss = (val_loss - min_loss) / (max_loss - min_loss)
            normalized_loss = max(0.0, min(1.0, normalized_loss))  # Clip to [0, 1]
            tf_prob = min_tf_prob + (base_tf_prob - min_tf_prob) * normalized_loss
            tf_prob = min(max(tf_prob, min_tf_prob), base_tf_prob)
            
            if val_loss < min_loss or val_loss > max_loss:
                logger.warning(f"val_loss {val_loss:.4f} outside expected range [{min_loss}, {max_loss}]; tf_prob clipped to {tf_prob:.4f}")
        else:
            # Fallback to epoch-based scheduling
            progress = epoch / total_epochs
            tf_prob = max(min_tf_prob, base_tf_prob - 0.6 * progress)
        
        logger.debug(f"Teacher forcing prob: epoch={epoch}, stage_idx={stage_idx}, val_loss={val_loss if val_loss is not None else 'None'}, tf_prob={tf_prob:.4f}")
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

        # Initialize outputs
        ratios_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
        directions_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)

        # Cumulative embedding
        cumulative_embed = self.cumulative_embed(cumulative_transforms)  # [B, num_teeth, embed_dim]
        cumulative_embed = torch.nan_to_num(cumulative_embed, nan=0.0, posinf=1.0, neginf=-1.0)

        # Prepare GRU input
        stage_outputs = []
        prev_ratios_seq = []
        prev_directions_seq = []
        hidden = torch.zeros(1, B * self.num_teeth, self.embed_dim, device=device)  # GRU hidden state

        for stage_idx in range(self.max_stages):
            tf_prob = self._get_teacher_forcing_params(epoch, total_epochs, stage_idx, val_loss)
            effective_tf_prob = tf_prob if training else 0.0
            use_tf = training and stage_idx > 0 and torch.rand(1).item() < effective_tf_prob

            # Prepare input embeddings
            start_idx = max(0, stage_idx - 1)
            if use_tf and targets is not None and directions is not None:
                ratios_prev = targets[:, start_idx:stage_idx, :, :]  # [B, num_stages_to_use, num_teeth, 6]
                directions_prev = directions[:, start_idx:stage_idx, :, :]  # [B, num_stages_to_use, num_teeth, 6]
                if ratios_prev.shape[1] > 0:
                    embedded_ratios = self.ratio_embed(ratios_prev)  # [B, num_stages_to_use, num_teeth, embed_dim]
                    embedded_directions = self.direction_embed(directions_prev)  # [B, num_stages_to_use, num_teeth, embed_dim]
                    embedded_prev = embedded_ratios + embedded_directions  # [B, num_stages_to_use, num_teeth, embed_dim]
                    embedded_prev = embedded_prev.mean(dim=1)  # Aggregate across stages: [B, num_teeth, embed_dim]
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
                    else:
                        embedded_prev = self.ratio_embed(torch.zeros(B, self.num_teeth, 6, device=device))  # [B, num_teeth, embed_dim]
                else:
                    embedded_prev = self.ratio_embed(torch.zeros(B, self.num_teeth, 6, device=device))  # [B, num_teeth, embed_dim]

            # Combine embeddings
            pos_embed = self.pos_embed[:, stage_idx, :].unsqueeze(1).expand(-1, self.num_teeth, -1)
            tooth_pos = self.tooth_pos_embed.expand(B, -1, -1)
            tgt = embedded_prev + pos_embed + tooth_pos + cumulative_embed  # [B, num_teeth, embed_dim]
            tgt = self.pre_norm(tgt)

            # GRU forward pass
            tgt = tgt.view(B * self.num_teeth, 1, self.embed_dim)  # [B * num_teeth, 1, embed_dim]
            output, hidden = self.gru(tgt, hidden)  # output: [B * num_teeth, 1, embed_dim], hidden: [1, B * num_teeth, embed_dim]
            output = output.view(B, self.num_teeth, self.embed_dim)  # [B, num_teeth, embed_dim]
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

        # Predict ratios for all stages at once
        ratios = self.ratio_head(stage_outputs.view(-1, self.embed_dim))  # [B * max_stages * num_teeth, max_stages * 6]
        ratios = ratios.view(B, self.max_stages, self.num_teeth, self.max_stages, 6)[:, :, :, 0, :]  # Select first stage, [B, max_stages, num_teeth, 6]
        ratios_sequence = torch.softmax(ratios, dim=1)  # Softmax across stages per tooth/parameter

        # Predict directions for all stages
        directions = self.direction_head(stage_outputs.view(-1, self.embed_dim))  # [B * max_stages * num_teeth, max_stages * 6]
        directions = directions.view(B, self.max_stages, self.num_teeth, self.max_stages, 6)[:, :, :, 0, :]  # Select first stage, [B, max_stages, num_teeth, 6]
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