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
        self.out_layer = nn.Linear(embed_dim, 6)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.01)
        nn.init.xavier_uniform_(self.target_embed.weight, gain=0.1)
        nn.init.zeros_(self.target_embed.bias)
        nn.init.xavier_uniform_(self.cumulative_embed.weight, gain=0.1)
        nn.init.zeros_(self.cumulative_embed.bias)
    
    def _stabilize_gradient(self, x):
        if self.training:
            def clip_grad_hook(grad):
                return torch.clamp(grad, -0.5, 0.5)
            x.register_hook(clip_grad_hook)
        return x
    
    def forward(self, memory, cumulative_transforms, num_stages=None, targets=None, use_teacher_forcing=False, training=False):
        logger = logging.getLogger('TrainLogger')
        B = memory.size(0)
        device = memory.device

        assert memory.shape == (B, self.num_teeth, self.embed_dim)
        assert cumulative_transforms.shape == (B, self.num_teeth, 6)
        
        memory = torch.nan_to_num(memory, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-100, 100)
        cumulative_transforms = torch.nan_to_num(cumulative_transforms, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-100, 100)
        
        logger.debug(f"Memory min: {memory.min().item():.4f}, max: {memory.max().item():.4f}, has_nan: {torch.isnan(memory).any().item()}")
        logger.debug(f"Cumulative transforms min: {cumulative_transforms.min().item():.4f}, max: {cumulative_transforms.max().item():.4f}, has_nan: {torch.isnan(cumulative_transforms).any().item()}")

        cumulative_embed = self.cumulative_embed(cumulative_transforms)
        cumulative_embed = torch.nan_to_num(cumulative_embed, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-100, 100)
        cumulative_embed = self._stabilize_gradient(cumulative_embed)
        cumulative_embed = cumulative_embed.unsqueeze(1).repeat(1, self.max_stages, 1, 1)
        logger.debug(f"Cumulative embed min: {cumulative_embed.min().item():.4f}, max: {cumulative_embed.max().item():.4f}, has_nan: {torch.isnan(cumulative_embed).any().item()}")

        if use_teacher_forcing and targets is not None:
            assert targets.shape == (B, self.max_stages, self.num_teeth, 6)
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-5, 5)
            logger.debug(f"Targets min: {targets.min().item():.4f}, max: {targets.max().item():.4f}, has_nan: {torch.isnan(targets).any().item()}")
            
            embedded_targets = self.target_embed(targets)
            embedded_targets = torch.nan_to_num(embedded_targets, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-100, 100)
            embedded_targets = self._stabilize_gradient(embedded_targets)
            logger.debug(f"Embedded targets min: {embedded_targets.min().item():.4f}, max: {embedded_targets.max().item():.4f}, has_nan: {torch.isnan(embedded_targets).any().item()}")
            
            embedded_targets = embedded_targets + self.pos_embed.unsqueeze(2) + cumulative_embed
            tgt = embedded_targets.view(B, self.max_stages * self.num_teeth, self.embed_dim)
        else:
            tgt = (self.pos_embed.unsqueeze(2) + cumulative_embed).view(B, self.max_stages * self.num_teeth, self.embed_dim)
        
        tgt = self.pre_norm(tgt.clamp(-100, 100))
        logger.debug(f"Tgt min: {tgt.min().item():.4f}, max: {tgt.max().item():.4f}, has_nan: {torch.isnan(tgt).any().item()}")

        cumulative_embed_avg = cumulative_embed.mean(dim=1)
        memory = memory + cumulative_embed_avg.contiguous()
        memory = memory.clamp(-100, 100)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_stages * self.num_teeth).to(device)
        
        try:
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            output = self.final_norm(output.clamp(-100, 100))
            output = output.view(B, self.max_stages, self.num_teeth, self.embed_dim)
            transforms_sequence = self.out_layer(output).clamp(-50, 50)
            
            logger.debug(f"Decoder output min: {output.min().item():.4f}, max: {output.max().item():.4f}, has_nan: {torch.isnan(output).any().item()}")
            logger.debug(f"Transforms sequence min: {transforms_sequence.min().item():.4f}, max: {transforms_sequence.max().item():.4f}, has_nan: {torch.isnan(transforms_sequence).any().item()}")
        
        except RuntimeError as e:
            logger.error(f"Runtime error in decoder: {e}")
            output = torch.zeros(B, self.max_stages, self.num_teeth, self.embed_dim, device=device)
            transforms_sequence = torch.zeros(B, self.max_stages, self.num_teeth, 6, device=device)
            return output, transforms_sequence

        stage_mask = torch.ones(B, self.max_stages, 1, 1, device=device)
        if num_stages is not None:
            for i in range(B):
                assert isinstance(num_stages[i], (int, torch.Tensor)) and 0 <= num_stages[i] <= self.max_stages
                stage_mask[i, num_stages[i]:] = 0.0
        
        pred_sum = (transforms_sequence * stage_mask).sum(dim=1)
        residual = (cumulative_transforms - pred_sum).clamp(-2, 2)
        residual_expanded = residual.unsqueeze(1).repeat(1, self.max_stages, 1, 1).contiguous()
        active_stages = stage_mask.sum(dim=1).view(B, 1, 1, 1).contiguous() + 1e-4
        adjustment = (residual_expanded / active_stages) * stage_mask
        # Avoid inplace operation by creating a new tensor
        transforms_sequence_adjusted = transforms_sequence + adjustment.contiguous()
        transforms_sequence = transforms_sequence_adjusted.clamp(-50, 50)
        
        logger.debug(f"Sum consistency: {((transforms_sequence * stage_mask).sum(dim=1) - cumulative_transforms).abs().mean().item():.4f}")
        logger.debug(f"Cumulative embed weight min: {self.cumulative_embed.weight.min().item():.4f}, max: {self.cumulative_embed.weight.max().item():.4f}")
        return output, transforms_sequence