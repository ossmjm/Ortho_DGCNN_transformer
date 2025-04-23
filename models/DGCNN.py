import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    logger = logging.getLogger('TrainLogger')
    batch_size, num_dims, num_teeth, num_points = x.size()
    logger.debug(f"get_graph_feature: input shape=[{batch_size}, {num_dims}, {num_teeth}, {num_points}], k={k}")
    
    # Validate input
    if len(x.shape) != 4:
        raise ValueError(f"Expected 4D input [batch_size, num_dims, num_teeth, num_points], got shape {x.shape}")
    
    # Reshape to [batch_size, num_dims, num_teeth * num_points] for KNN
    x = x.view(batch_size, num_dims, num_teeth * num_points)
    total_points = num_teeth * num_points
    
    if idx is None:
        if dim9:
            idx = knn(x[:, 6:], k=k)
        else:
            idx = knn(x, k=k)
    
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * total_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x = x.transpose(2, 1).contiguous()  # [batch_size, total_points, num_dims]
    feature = x.view(batch_size * total_points, -1)[idx, :]
    feature = feature.view(batch_size, total_points, k, num_dims)
    x = x.view(batch_size, total_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3)  # [batch_size, total_points, k, 2*num_dims]
    feature = feature.view(batch_size, num_teeth, num_points, k, 2*num_dims)
    feature = feature.permute(0, 4, 1, 2, 3).contiguous()  # [batch_size, 2*num_dims, num_teeth, num_points*k]
    feature = feature.view(batch_size, 2*num_dims, num_teeth, num_points)  # [batch_size, 2*num_dims, num_teeth, num_points]
    
    logger.debug(f"get_graph_feature: output shape={feature.shape}, expected channels={2*num_dims}")
    if feature.size(1) != 2 * num_dims:
        raise ValueError(f"Expected {2*num_dims} channels, got {feature.size(1)}")
    
    return feature

class DGCNN(nn.Module):
    def __init__(self, k=20, embed_dim=256, num_dims=13):
        super(DGCNN, self).__init__()
        self.k = k
        self.embed_dim = embed_dim
        self.num_dims = num_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(embed_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * num_dims, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, embed_dim, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"DGCNN input shape={x.shape}")
        
        # Validate input shape
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input [batch_size, num_teeth, num_points, num_dims], got shape {x.shape}")
        batch_size, num_teeth, num_points, num_dims = x.size()
        if num_teeth != 14:
            logger.warning(f"Unexpected num_teeth={num_teeth}, expected 14")
        if num_dims != self.num_dims:
            logger.warning(f"Unexpected num_dims={num_dims}, expected {self.num_dims}")
        
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch_size, num_dims, num_teeth, num_points]
        logger.debug(f"After permute: shape={x.shape}")
        x = get_graph_feature(x, k=self.k)  # [batch_size, 2*num_dims, num_teeth, num_points]
        logger.debug(f"After get_graph_feature: shape={x.shape}")
        
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
        x = x.view(batch_size, num_teeth, -1)
        x = x.view(batch_size, 2, 7, -1)
        logger.debug(f"DGCNN output shape={x.shape}")
        return x