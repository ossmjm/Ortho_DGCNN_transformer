import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def sample_and_group(x, npoint, nsample, radius=None, k=16):
    """Sample npoint points using farthest point sampling and group nsample neighbors."""
    batch_size, num_teeth, num_points, channels = x.size()
    device = x.device
    
    # Flatten for FPS
    x_flat = x.view(batch_size * num_teeth, num_points, channels)
    idx = torch.zeros(batch_size * num_teeth, npoint, dtype=torch.long, device=device)
    
    # Simple FPS: Select random points (approximation for speed)
    for i in range(batch_size * num_teeth):
        perm = torch.randperm(num_points, device=device)[:npoint]
        idx[i] = perm
    
    # Gather sampled points
    sampled_points = x_flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, channels))
    sampled_points = sampled_points.view(batch_size, num_teeth, npoint, channels)
    
    # Group neighbors (KNN)
    x_trans = x_flat.transpose(1, 2).contiguous()  # [B*T, C, N]
    dists = torch.cdist(x_flat[:, :, :3], sampled_points[:, :, :3])  # Distance on XYZ
    _, neighbor_idx = dists.topk(k=nsample, dim=2, largest=False)  # [B*T, N', K]
    
    # Gather neighbor features
    neighbor_points = x_flat.gather(1, neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, channels))
    neighbor_points = neighbor_points.view(batch_size, num_teeth, npoint, nsample, channels)
    
    return sampled_points, neighbor_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channels, mlp, k=16):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.k = k
        self.mlp = nn.ModuleList()
        last_channels = in_channels
        for out_channels in mlp:
            self.mlp.append(nn.Sequential(
                nn.Conv2d(last_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            last_channels = out_channels
    
    def forward(self, x):
        # x: [batch_size, num_teeth, num_points, channels]
        sampled_points, grouped_points = sample_and_group(x, self.npoint, self.nsample, k=self.k)
        # grouped_points: [batch_size, num_teeth, npoint, nsample, channels]
        
        # Center points
        grouped_points = grouped_points - sampled_points.unsqueeze(3)  # [B, T, N', K, C]
        grouped_points = grouped_points.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, C, N', K]
        grouped_points = grouped_points.view(-1, grouped_points.size(2), grouped_points.size(3), grouped_points.size(4))
        
        # Apply MLPs
        for mlp_layer in self.mlp:
            grouped_points = mlp_layer(grouped_points)
        
        # Max-pool over neighbors
        features = grouped_points.max(dim=-1)[0]  # [B*T, C', N']
        features = features.view(-1, x.size(1), features.size(1), features.size(2))  # [B, T, C', N']
        features = features.permute(0, 1, 3, 2).contiguous()  # [B, T, N', C']
        
        return features

class CumulativeTransformationModel(nn.Module):
    def __init__(self, num_teeth=14, num_points=256, channels=13, embed_dim=256, num_heads=4):
        super().__init__()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.channels = channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # PointNet++ Set Abstraction Layers
        self.sa1 = PointNetSetAbstraction(
            npoint=64, nsample=32, in_channels=channels, mlp=[64, 128], k=16
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=16, nsample=16, in_channels=128, mlp=[256, 512], k=8
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        
        # Global pooling and MLP
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"CumulativeTransformationModel input shape={x.shape}")
        
        # Input: [batch_size, num_teeth, num_points, channels]
        batch_size, num_teeth, num_points, channels = x.size()
        
        # Validate input
        if num_teeth != self.num_teeth or num_points != self.num_points or channels != self.channels:
            raise ValueError(f"Expected input [B, {self.num_teeth}, {self.num_points}, {self.channels}], got {x.shape}")
        
        # Set Abstraction
        x = self.sa1(x)  # [batch_size, num_teeth, 64, 128]
        x = self.sa2(x)  # [batch_size, num_teeth, 16, 512]
        
        # Multi-head attention
        x = x.view(batch_size * num_teeth, 16, 512)  # [B*T, 16, 512]
        attn_output, _ = self.attention(x, x, x)  # [B*T, 16, 512]
        x = x + attn_output  # Residual connection
        x = x.view(batch_size, num_teeth, 16, 512)  # [B, T, 16, 512]
        
        # Global max-pooling
        x = x.max(dim=2)[0]  # [batch_size, num_teeth, 512]
        
        # MLP for transformation prediction
        x = x.view(-1, 512)  # [B*T, 512]
        transforms = self.mlp(x)  # [B*T, 6]
        transforms = transforms.view(batch_size, num_teeth, 6)  # [batch_size, num_teeth, 6]
        
        logger.debug(f"CumulativeTransformationModel output shape={transforms.shape}")
        return transforms