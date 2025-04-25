import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import logging

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_teeth, max_stages, num_layers=1, num_heads=4, mlp_ratio=4.0, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        
        # Positional encoding for stages
        self.pos_embed = nn.Parameter(torch.zeros(1, max_stages, embed_dim))
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer for transformations (outputs unnormalized values)
        self.out_layer = nn.Linear(embed_dim, 6)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, memory, cumulative_transforms, num_stages=None, targets=None, use_teacher_forcing=False, training=False):
        logger = logging.getLogger('TrainLogger')
        B = memory.size(0)
        device = memory.device
        
        # Prepare target sequence
        tgt = torch.zeros(B, self.max_stages, self.num_teeth, self.embed_dim, device=device)
        if use_teacher_forcing and targets is not None:
            targets_scaled = targets / (targets.abs().sum(dim=1, keepdim=True) + 1e-6)  # Normalize
            targets_scaled = targets_scaled * cumulative_transforms.unsqueeze(1)  # Scale by cumulative
            tgt[:, :-1, :, :] = self.out_layer.weight.new_zeros(targets_scaled[:, :-1, :, :].shape)  # Dummy for simplicity
        
        # Add positional encoding
        tgt = tgt + self.pos_embed.unsqueeze(2)  # [B, max_stages, num_teeth, embed_dim]
        
        # Reshape for transformer
        memory = memory.view(B, self.num_teeth, -1)  # [B, num_teeth, embed_dim]
        tgt = tgt.view(B, self.max_stages * self.num_teeth, self.embed_dim)  # [B, max_stages * num_teeth, embed_dim]
        
        # Create causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_stages * self.num_teeth).to(device)
        
        # Decode
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # [B, max_stages * num_teeth, embed_dim]
        output = output.view(B, self.max_stages, self.num_teeth, self.embed_dim)
        
        # Predict unnormalized transformations
        transforms_sequence = self.out_layer(output)  # [B, max_stages, num_teeth, 6]
        
        # Scale by cumulative_transforms
        transforms_sequence = transforms_sequence * cumulative_transforms.unsqueeze(1)  # [B, max_stages, num_teeth, 6]
        
        # Normalize to ensure sum equals cumulative_transforms
        if training and num_stages is not None:
            stage_mask = torch.ones(B, self.max_stages, 1, 1, device=device)
            for i in range(B):
                stage_mask[i, num_stages[i]:] = 0.0
            stage_sum = (transforms_sequence * stage_mask).sum(dim=1)  # [B, num_teeth, 6]
            scale_factor = cumulative_transforms / (stage_sum + 1e-6)  # [B, num_teeth, 6]
            scale_factor = torch.where(stage_sum.abs() < 1e-6, torch.ones_like(scale_factor), scale_factor)
            transforms_sequence = transforms_sequence * scale_factor.unsqueeze(1) * stage_mask
        else:
            stage_sum = transforms_sequence.sum(dim=1)  # [B, num_teeth, 6]
            scale_factor = cumulative_transforms / (stage_sum + 1e-6)  # [B, num_teeth, 6]
            scale_factor = torch.where(stage_sum.abs() < 1e-6, torch.ones_like(scale_factor), scale_factor)
            transforms_sequence = transforms_sequence * scale_factor.unsqueeze(1)
        
        logger.debug(f"Decoder output range: min={transforms_sequence.min().item():.4f}, max={transforms_sequence.max().item():.4f}")
        logger.debug(f"Sum consistency: {((transforms_sequence.sum(dim=1) - cumulative_transforms).abs().mean().item()):.4f}")
        return output, transforms_sequence

class MViTv2(nn.Module):
    def __init__(
        self,
        embed_dim: int = 96,
        num_teeth: int = 14,
        max_stages: int = 25,
        depths: list = [1, 2, 11, 2],
        num_heads: list = [3, 3, 3, 3],
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        decoder_layers: int = 1,
        teacher_forcing: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        self.teacher_forcing = teacher_forcing
        
        # Load pretrained MViTv2-small from torchvision
        weights = MViT_V2_S_Weights.DEFAULT
        self.mvit = mvit_v2_s(weights=weights)
        self.mvit.head = nn.Identity()  # Remove classification head
        
        # Input adapter to aggregate features
        self.input_conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=(2, 1, 4), stride=(2, 1, 4))  # Reduce a and b
        self.feature_proj = nn.Linear(embed_dim, embed_dim * 4)  # Project to MViTv2 output dim
        self.pos_embed = nn.Parameter(torch.zeros(1, num_teeth, embed_dim * 4))  # Adjusted for MViTv2 output (384)
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            embed_dim=embed_dim * 4,
            num_teeth=num_teeth,
            max_stages=max_stages,
            num_layers=decoder_layers,
            num_heads=num_heads[-1],
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
        logger = logging.getLogger('TrainLogger')
        logger.info(f"Loaded pretrained MViTv2-small from torchvision with embed_dim={embed_dim}")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, cumulative_transforms, num_stages=None, targets=None, epoch=None, total_epochs=None, cumulative_teacher_forcing=False):
        logger = logging.getLogger('TrainLogger')
        B = x.size(0)
        
        # Process DGCNN features: [B, 2, 14, 4, embed_dim] -> [B, embed_dim, 2, 14, 4]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, embed_dim, 2, 14, 4]
        x = self.input_conv(x)  # [B, embed_dim, 1, 14, 1]
        x = x.squeeze(2).squeeze(4)  # [B, embed_dim, 14]
        x = x.permute(0, 2, 1).contiguous()  # [B, 14, embed_dim]
        
        # Project to MViTv2 feature space
        x = self.feature_proj(x)  # [B, 14, embed_dim * 4]
        x = x + self.pos_embed  # [B, 14, embed_dim * 4]
        
        # Use cumulative transforms directly or apply teacher forcing
        cumulative_input = cumulative_transforms if not cumulative_teacher_forcing else targets
        
        # Decode stage-wise transformations
        use_stage_teacher_forcing = self.training and self.teacher_forcing and targets is not None
        alpha = min(1.0, epoch / (total_epochs * 0.5)) if epoch is not None and total_epochs is not None else 1.0
        use_stage_teacher_forcing = use_stage_teacher_forcing and torch.rand(1).item() < alpha
        
        decoder_features, transforms_sequence = self.decoder(
            memory=x,
            cumulative_transforms=cumulative_input,
            num_stages=num_stages,
            targets=targets,
            use_teacher_forcing=use_stage_teacher_forcing,
            training=self.training
        )  # [B, max_stages, num_teeth, embed_dim * 4], [B, max_stages, num_teeth, 6]
        
        logger.debug(f"MViTv2 output range: min={transforms_sequence.min().item():.4f}, max={transforms_sequence.max().item():.4f}")
        return decoder_features, transforms_sequence