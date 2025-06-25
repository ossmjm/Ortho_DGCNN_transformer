import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-3)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x

class CumulativeTransformationModel(nn.Module):
    def __init__(self, num_teeth=14, embed_dim=384, num_heads=4):
        super().__init__()
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.logger = logging.getLogger('TrainLogger')
        
        # Attention layer to capture inter-tooth relationships
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                
        self.attn_ffn = FeedForward(embed_dim, embed_dim * 2)
        
        # Shared MLP for initial feature extraction
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.Linear(512, 450),
            nn.BatchNorm1d(450, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Shared MLP for translation path
        self.shared_mlp_trans = nn.Sequential(
            nn.Linear(450, 256),
            nn.Linear(256, 200),
            nn.BatchNorm1d(200, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Shared MLP for rotation path
        self.shared_mlp_rot = nn.Sequential(
            nn.Linear(450, 256),
            nn.Linear(256, 200),
            nn.BatchNorm1d(200, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Translation-specific heads
        self.cumulative_transform_trans = nn.Sequential(
            nn.Linear(200, 128),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(96, 3)  # First 3 parameters (translations)
        )

        self.param_activity_head_trans = nn.Sequential(
            nn.Linear(200, 128),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)  # First 3 parameters (translations)
        )

        # Rotation-specific heads
        self.cumulative_transform_rot = nn.Sequential(
            nn.Linear(200, 128),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(96, 3)  # Last 3 parameters (rotations)
        )

        self.activity_head = nn.Sequential(
            nn.Linear(450, 200),
            nn.BatchNorm1d(200, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(200, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # One output per tooth input
        )

        self.param_activity_head_rot = nn.Sequential(
            nn.Linear(200, 128),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)  # Last 3 parameters (rotations)
        )

        # Shared directions head
        self.directions_trans = nn.Sequential(
            nn.Linear(200, 128),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)
        )

        # Shared directions head
        self.directions_rot = nn.Sequential(
            nn.Linear(200, 128),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-3)
        self.final_norm_trans = nn.LayerNorm(3, eps=1e-3)  # Separate norms for translations and rotations
        self.final_norm_rot = nn.LayerNorm(3, eps=1e-3)
        
        self._init_weights()
        
        # Freeze BatchNorm statistics during early training
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size, num_teeth, embed_dim = x.size()
        assert num_teeth == self.num_teeth, f"Expected num_teeth={self.num_teeth}, got {num_teeth}"
        assert embed_dim == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {embed_dim}"
        
        # Apply attention to capture inter-tooth relationships
        x, _ = self.attention(x, x, x)
        x = self.attn_ffn(x)
        x = self.norm(x)
        
        self.logger.debug(f"Attention output min: {x.min().item():.4f}, max: {x.max().item():.4f}, has_nan: {torch.isnan(x).any().item()}")
        
        # Flatten for MLP processing
        x_flat = x.view(batch_size * num_teeth, embed_dim)
        
        # Shared MLP for initial feature extraction
        x = self.shared_mlp(x_flat)

        activity_logits = self.activity_head(x).view(batch_size, num_teeth, -1)[:, :, 0]  # Reshape and take the single logit per tooth
        # Split into translation and rotation paths
        x_trans = self.shared_mlp_trans(x)
        x_rot = self.shared_mlp_rot(x)
        
        # Predict translations
        transforms_trans = self.cumulative_transform_trans(x_trans)
        param_activity_logits_trans = self.param_activity_head_trans(x_trans).view(batch_size, num_teeth, 3)
        
        transforms_trans = transforms_trans.view(batch_size, num_teeth, 3)
        transforms_trans = self.final_norm_trans(transforms_trans)
        
        self.logger.debug(f"MLP output (transforms_trans) min: {transforms_trans.min().item():.4f}, max: {transforms_trans.max().item():.4f}, has_nan: {torch.isnan(transforms_trans).any().item()}")
        
        # Predict rotations
        transforms_rot = self.cumulative_transform_rot(x_rot)
        param_activity_logits_rot = self.param_activity_head_rot(x_rot).view(batch_size, num_teeth, 3)
        
        transforms_rot = transforms_rot.view(batch_size, num_teeth, 3)
        transforms_rot = self.final_norm_rot(transforms_rot)
        
        self.logger.debug(f"MLP output (transforms_rot) min: {transforms_rot.min().item():.4f}, max: {transforms_rot.max().item():.4f}, has_nan: {torch.isnan(transforms_rot).any().item()}")
        
        # Predict directions (shared for both paths)
        directions_logits_rot = self.directions_rot(x_rot).view(batch_size, num_teeth, 3)
        
        directions_logits_trans = self.directions_trans(x_trans).view(batch_size, num_teeth, 3)

        
        self.logger.debug(f"Directions rot min: {directions_logits_rot.min().item():.4f}, max: {directions_logits_rot.max().item():.4f}, has_nan: {torch.isnan(directions_logits_rot).any().item()}")
        self.logger.debug(f"Directions trans min: {directions_logits_trans.min().item():.4f}, max: {directions_logits_trans.max().item():.4f}, has_nan: {torch.isnan(directions_logits_trans).any().item()}")
        
        self.logger.debug(f"CumulativeTransformationModel output shapes: transforms_trans={transforms_trans.shape} "
                         f"param_activity_logits_trans={param_activity_logits_trans.shape}, transforms_rot={transforms_rot.shape}, "
                         f"activity_logits={activity_logits.shape}, param_activity_logits_rot={param_activity_logits_rot.shape}, "
                         f"directions_rot={directions_logits_rot.shape},"
                        f"directions_trans={directions_logits_trans.shape}")
        
        return (transforms_trans, activity_logits, param_activity_logits_trans, 
                transforms_rot, param_activity_logits_rot, 
                directions_logits_rot, directions_logits_trans)