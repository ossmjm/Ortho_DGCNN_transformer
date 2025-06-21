import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def knn(x, k):
    # Note: For larger num_points (e.g., 1000), consider optimizing KNN with torch_cluster.knn or faiss for efficiency.
    batch_size_teeth, num_dims, num_points = x.size()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k, idx=None):
    batch_size_teeth, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x[:, :3, :], k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size_teeth, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size_teeth * num_points, -1)[idx, :]
    feature = feature.view(batch_size_teeth, num_points, k, num_dims)
    x = x.view(batch_size_teeth, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNN(nn.Module):
    def __init__(self, in_channels=3, embed_dim=384, num_teeth=14, num_points=256, k=20, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.num_points = num_points
        
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.01)
        self.bn4 = nn.BatchNorm2d(256, momentum=0.01)
        self.bn5 = nn.BatchNorm1d(embed_dim, momentum=0.01)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * in_channels, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(1024, embed_dim, kernel_size=1, bias=False),  # Changed from 512 to 1024 for max+mean pooling
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        logger = logging.getLogger('TrainLogger')
        batch_size, num_teeth, num_points, channels = x.size()
        assert num_teeth == self.num_teeth, f"Expected num_teeth={self.num_teeth}, got {num_teeth}"
        assert num_points == self.num_points, f"Expected num_points={self.num_points}, got {num_points}"
        assert channels == self.in_channels, f"Expected channels={self.in_channels}, got {channels}"
        
        logger.debug(f"DGCNN input shape: {x.shape}")
        
        x = x.view(batch_size * num_teeth, num_points, channels).permute(0, 2, 1).contiguous()
        
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.conv5(x)
        x = torch.cat([torch.max(x, dim=2)[0], torch.mean(x, dim=2)], dim=1)  # Max+mean pooling
        x = x.view(batch_size, num_teeth, self.embed_dim)
        
        logger.debug(f"DGCNN output shape: {x.shape}")
        return x