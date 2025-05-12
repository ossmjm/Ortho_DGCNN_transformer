import torch
import torch.nn as nn
import logging

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_teeth, max_stages, num_layers=1, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_stages, embed_dim))
        self.target_embed = nn.Linear(6, embed_dim)
        self.cumulative_embed = nn.Linear(6, embed_dim)
        
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-4)
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
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Activity and parameter activity prediction heads
        self.activity_head = nn.Linear(embed_dim, 1)
        self.param_activity_head = nn.Linear(embed_dim, 6)
        self.out_layer = nn.Linear(embed_dim, 6)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.01)
        nn.init.xavier_uniform_(self.target_embed.weight, gain=0.1)
        nn.init.zeros_(self.target_embed.bias)
        nn.init.xavier_uniform_(self.cumulative_embed.weight, gain=0.1)
        nn.init.zeros_(self.cumulative_embed.bias)
    
    def forward(self, memory, cumulative_transforms, num_stages=None, targets=None, use_teacher_forcing=False, training=False):
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

        memory = torch.nan_to_num(memory, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_transforms = torch.nan_to_num(cumulative_transforms, nan=0.0, posinf=1.0, neginf=-1.0)

        logger.debug(f"Memory min: {memory.min().item():.4f}, max: {memory.max().item():.4f}, has_nan: {torch.isnan(memory).any().item()}")
        logger.debug(f"Cumulative transforms min: {cumulative_transforms.min().item():.4f}, max: {cumulative_transforms.max().item():.4f}, has_nan: {torch.isnan(cumulative_transforms).any().item()}")

        # Keep memory as [B, num_teeth, embed_dim]
        memory_key_padding_mask = torch.zeros(B, self.num_teeth, dtype=torch.bool, device=device)

        cumulative_embed = self.cumulative_embed(cumulative_transforms)
        cumulative_embed = torch.nan_to_num(cumulative_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        cumulative_embed = cumulative_embed.unsqueeze(1).expand(-1, self.max_stages, -1, -1)  # [B, max_stages, num_teeth, embed_dim]

        # Prepare target embeddings with partial teacher forcing
        if use_teacher_forcing and targets is not None and training:
            expected_targets_shape = (B, self.max_stages, self.num_teeth, 6)
            if targets.shape != expected_targets_shape:
                logger.error(f"Invalid targets shape: got {targets.shape}, expected {expected_targets_shape}")
                raise RuntimeError(f"Targets shape mismatch: got {targets.shape}, expected {expected_targets_shape}")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=-1.0)
            embedded_targets = self.target_embed(targets)
            pos_embed_expanded = self.pos_embed.unsqueeze(2).expand(-1, -1, self.num_teeth, -1)  # [1, max_stages, num_teeth, embed_dim]
            embedded_targets = embedded_targets + pos_embed_expanded + cumulative_embed
            tgt = embedded_targets.view(B, self.max_stages * self.num_teeth, self.embed_dim)
        else:
            pos_embed_expanded = self.pos_embed.unsqueeze(2).expand(-1, -1, self.num_teeth, -1)  # [1, max_stages, num_teeth, embed_dim]
            tgt = (pos_embed_expanded + cumulative_embed).view(B, self.max_stages * self.num_teeth, self.embed_dim)
        
        tgt = self.pre_norm(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_stages * self.num_teeth).to(device)
        
        # Log shapes before decoder
        logger.debug(f"tgt shape: {tgt.shape}, memory shape: {memory.shape}, tgt_mask shape: {tgt_mask.shape}, memory_key_padding_mask shape: {memory_key_padding_mask.shape}")

        # Decoder forward pass
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.final_norm(output)
        output = output.view(B, self.max_stages, self.num_teeth, self.embed_dim)

        # Step 1: Predict tooth activity
        activity_logits = self.activity_head(output).squeeze(-1)  # [B, max_stages, num_teeth]
        activity_probs = torch.sigmoid(activity_logits)
        activity_mask = (activity_probs > 0.5).float()  # [B, max_stages, num_teeth]

        # Step 2: Predict parameter activity for active teeth
        param_activity_logits = self.param_activity_head(output)  # [B, max_stages, num_teeth, 6]
        param_activity_probs = torch.sigmoid(param_activity_logits)
        param_activity_mask = (param_activity_probs > 0.5).float() * activity_mask.unsqueeze(-1)  # [B, max_stages, num_teeth, 6]

        # Step 3: Predict transformations only for active teeth and parameters
        transforms_sequence = self.out_layer(output)  # [B, max_stages, num_teeth, 6]
        transforms_sequence = transforms_sequence * param_activity_mask  # Zero out inactive parameters

        # Enforce zero stage-wise transforms if cumulative transform is zero
        cumulative_zero_mask = (cumulative_transforms == 0).float().unsqueeze(1)  # [B, 1, num_teeth, 6]
        transforms_sequence = transforms_sequence * (1 - cumulative_zero_mask) + transforms_sequence * (cumulative_zero_mask * 0)

        # Enhanced cumulative transform distribution
        stage_mask = torch.ones(B, self.max_stages, 1, 1, device=device)
        if num_stages is not None:
            for i in range(B):
                assert isinstance(num_stages[i], (int, torch.Tensor)) and 0 <= num_stages[i] <= self.max_stages
                stage_mask[i, num_stages[i]:] = 0.0
        
        pred_sum = (transforms_sequence * stage_mask).sum(dim=1)  # [B, num_teeth, 6]
        residual = cumulative_transforms - pred_sum  # [B, num_teeth, 6]
        # active_stages = stage_mask.sum(dim=1).clamp(min=1e-4)  # [B, 1, 1]
        active_stages = stage_mask.sum(dim=1, keepdim=True).clamp(min=1e-4)  # [B, 1, 1, 1]
        residual_expanded = residual.unsqueeze(1).expand(-1, self.max_stages, -1, -1)  # [B, max_stages, num_teeth, 6]
        adjustment = (residual_expanded / active_stages) * stage_mask * param_activity_mask  # [B, max_stages, num_teeth, 6]
        transforms_sequence = transforms_sequence + adjustment

        logger.debug(f"Activity mask mean: {activity_mask.mean().item():.4f}")
        logger.debug(f"Param activity mask mean: {param_activity_mask.mean().item():.4f}")
        logger.debug(f"Transforms sequence min: {transforms_sequence.min().item():.4f}, max: {transforms_sequence.max().item():.4f}")
        logger.debug(f"Consistency error: {((transforms_sequence * stage_mask).sum(dim=1) - cumulative_transforms).abs().mean().item():.4f}")

        return output, transforms_sequence, activity_logits, param_activity_logits