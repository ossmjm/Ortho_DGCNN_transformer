# models/DGCNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DGCNN(nn.Module):
    def __init__(self, in_channels=13, embed_dim=256, num_teeth=14, k=10):
        super(DGCNN, self).__init__()
        self.k = k
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # EdgeConv layers for graph feature extraction
        self.conv1 = nn.Conv2d(2 * in_channels, 64, kernel_size=1, bias=False)  # Fixed: 2 * in_channels
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, embed_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        
        # Global feature aggregation
        self.conv4 = nn.Conv1d(embed_dim * num_teeth, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Final layers
        self.linear1 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, num_teeth * embed_dim)

    def knn(self, x, k):
        # x: (batch_size, num_dims, num_points)
        batch_size, num_dims, num_points = x.size()
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

    def get_graph_feature(self, x, k=10, idx=None):
        # x: (batch_size, num_dims, num_points)
        batch_size, num_dims, num_points = x.size()
        
        if idx is None:
            idx = self.knn(x, k=k)  # (batch_size, num_points, k)
        
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature  # (batch_size, 2*num_dims, num_points, k)

    def forward(self, x):
        # x: (batch_size, num_teeth, num_patches, channels, patch_size)
        batch_size, num_teeth, num_patches, channels, patch_size = x.size()
        num_points = num_patches * patch_size
        
        # Reshape for processing: treat each tooth's point cloud independently
        x = x.view(batch_size * num_teeth, num_patches, channels, patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * num_teeth, channels, num_points)
        
        # Use only spatial coordinates (x, y, z) for graph construction
        x_spatial = x[:, :3, :]  # (batch_size * num_teeth, 3, num_points)
        
        # Graph feature extraction using EdgeConv
        x = self.get_graph_feature(x, k=self.k)  # (batch_size * num_teeth, 2*channels, num_points, k)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size * num_teeth, 64, num_points)
        
        x = self.get_graph_feature(x, k=self.k)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size * num_teeth, 64, num_points)
        
        x = self.get_graph_feature(x, k=self.k)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size * num_teeth, embed_dim, num_points)
        
        # Global pooling over points
        x = torch.max(x, 2)[0]  # (batch_size * num_teeth, embed_dim)
        x = x.view(batch_size, num_teeth * self.embed_dim)
        
        # Global feature aggregation
        x = x.view(batch_size, num_teeth * self.embed_dim, 1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = x.view(batch_size, 256)
        
        # Final processing
        x = F.leaky_relu(self.bn5(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = self.linear2(x)  # (batch_size, num_teeth * embed_dim)
        
        return x