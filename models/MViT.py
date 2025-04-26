import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Output layer for transformations
        self.out_layer = nn.Linear(embed_dim, 6)
        self.target_embed = nn.Linear(6, embed_dim)  # <<< New line
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, memory, cumulative_transforms, num_stages=None, targets=None, use_teacher_forcing=False, training=False):
        logger = logging.getLogger('TrainLogger')
        B = memory.size(0)
        device = memory.device

        # Prepare target sequence
        tgt = torch.zeros(B, self.max_stages, self.num_teeth, self.embed_dim, device=device)
        if use_teacher_forcing and targets is not None:            
            targets_scaled = targets / (targets.abs().sum(dim=1, keepdim=True) + 1e-6)
            targets_scaled = targets_scaled * cumulative_transforms.unsqueeze(1)
            
            embedded_targets = self.target_embed(targets_scaled[:, :-1, :, :]).contiguous()
            print(f"tgt shape: {tgt.shape}")
            print(f"targets_scaled shape: {targets_scaled.shape}")
            print(f"embedded_targets shape: {embedded_targets.shape}")

            tgt[:, :-1, :, :] = embedded_targets

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
            scale_factor = cumulative_transforms / (stage_sum + 1e-6)
            scale_factor = torch.where(stage_sum.abs() < 1e-6, torch.ones_like(scale_factor), scale_factor)
            transforms_sequence = transforms_sequence * scale_factor.unsqueeze(1) * stage_mask
        else:
            stage_sum = transforms_sequence.sum(dim=1)  # [B, num_teeth, 6]
            scale_factor = cumulative_transforms / (stage_sum + 1e-6)
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
        num_points: int = 256,
        channels: int = 13,
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
        self.num_points = num_points
        self.channels = channels
        self.teacher_forcing = teacher_forcing
        
        # Load pretrained MViTv2-small
        weights = MViT_V2_S_Weights.DEFAULT
        self.mvit = mvit_v2_s(weights=weights)
        self.mvit.head = nn.Identity()  # Remove classification head
        
        # Input adapter for point clouds
        self.input_adapter = nn.Sequential(
            nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        # Channel projection to match MViTv2 input (96 -> 3 channels)
        self.channel_proj = nn.Conv3d(embed_dim, 3, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.channel_proj.weight, mode='fan_out', nonlinearity='relu')
        
        # Feature projection
        self.feature_proj = nn.Linear(768, embed_dim * 4)  # MViTv2 outputs 768-dim features
        self.pos_embed = nn.Parameter(torch.zeros(1, num_teeth, embed_dim * 4))
        
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
        logger.info(f"Initialized MViTv2 with embed_dim={embed_dim}, num_points={num_points}")
    
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
        
        # Input: [B, num_teeth, num_points, channels]
        x = x.view(B, self.num_teeth * self.num_points, self.channels).permute(0, 2, 1)  # [B, channels, T*N]
        x = x.view(B, self.channels, self.num_teeth, self.num_points)  # [B, channels, T, N]
        
        # Input adapter
        x = self.input_adapter(x)  # [B, embed_dim, T, N]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, N, embed_dim]
        
        # Reshape for MViTv2: Treat num_teeth as temporal dimension, num_points as spatial
        spatial_dim = int(self.num_points ** 0.5)  # e.g., sqrt(256) = 16
        x = x.view(B, self.num_teeth, spatial_dim, spatial_dim, self.embed_dim)  # [B, T, H, W, embed_dim]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, embed_dim, T, H, W]
        
        # Pad spatial dimensions to 224x224 if necessary
        target_h = target_w = 224
        h, w = x.size(3), x.size(4)
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad width, then height

        # Pad temporal dimension (T) up to 16 frames
        if x.size(2) < 16:
            pad_t = 16 - x.size(2)
            x = torch.cat([x, torch.zeros(x.size(0), x.size(1), pad_t, x.size(3), x.size(4), device=x.device)], dim=2)

        # Project channels to 3 for MViT
        x = self.channel_proj(x)  # [B, 3, T, 224, 224]
        
        # logger.debug(f"Input to mvit: {x.shape}")  # [B, 3, T, 224, 224]
        # print(f"MViT input {x.shape}")
        # Pass through MViTv2
        x = self.mvit(x)  # x: [B, N, 768]
        # print(f"After mvit: {x.shape}")
        # x = x.mean(dim=1)  # Global average pooling over tokens -> [B, 768]
        x = x.unsqueeze(1).expand(-1, self.num_teeth, -1)  # [B, num_teeth, 768]

        # Project features
        x = self.feature_proj(x)  # [B, num_teeth, embed_dim * 4]
        x = x + self.pos_embed  # [B, num_teeth, embed_dim * 4]
        
        # Rest is the same: decoding
        cumulative_input = cumulative_transforms if not cumulative_teacher_forcing else targets
        
        use_stage_teacher_forcing = self.training and self.teacher_forcing and targets is not None
        alpha = min(1.0, epoch / (total_epochs * 0.5)) if epoch is not None and total_epochs is not None else 1.0
        use_stage_teacher_forcing = use_stage_teacher_forcing and torch.rand(1).item() < alpha
        # Fix targets if extra dimension
        if targets is not None and targets.dim() == 5:
            print(f'Targets dim before:{targets.shape}')
            targets = targets[:, 0]
            print(f'Targets dim after:{targets.shape}')

        # Call decoder
        decoder_features, transforms_sequence = self.decoder(
            memory=x,
            cumulative_transforms=cumulative_input,
            num_stages=num_stages,
            targets=targets,
            use_teacher_forcing=use_stage_teacher_forcing,
            training=self.training
        )
        logger.debug(f"MViTv2 output range: min={transforms_sequence.min().item():.4f}, max={transforms_sequence.max().item():.4f}")
        return decoder_features, transforms_sequence
