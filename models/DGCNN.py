import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x, k=10):
        batch_size, num_points, num_dims = x.size()
        x = x.transpose(1, 2).contiguous()  # [B, num_dims, num_points]
        idx = self.get_knn_idx(x, k)  # [B, num_points, k]
        
        x_knn = self.get_knn_features(x, idx, k)  # [B, num_dims, num_points, k]
        x = x.unsqueeze(-1).repeat(1, 1, 1, k)  # [B, num_dims, num_points, k]
        x = torch.cat((x_knn - x, x), dim=1)  # [B, num_dims*2, num_points, k]
        
        x = self.conv(x)  # [B, out_channels, num_points, k]
        x = x.max(dim=-1, keepdim=False)[0]  # [B, out_channels, num_points]
        return x.transpose(1, 2).contiguous()  # [B, num_points, out_channels]
    
    def get_knn_idx(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        return pairwise_distance.topk(k=k, dim=-1)[1]
    
    def get_knn_features(self, x, idx, k):
        batch_size, num_dims, num_points = x.size()
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.transpose(2, 1).contiguous()  # [B, num_points, num_dims]
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        return feature.permute(0, 3, 1, 2)  # [B, num_dims, num_points, k]

class DGCNN(nn.Module):
    def __init__(self, in_channels=13, embed_dim=256, num_teeth=14, k=10):
        super(DGCNN, self).__init__()
        self.k = k
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        
        self.conv1 = EdgeConv(in_channels, 64)
        self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 128)
        self.conv4 = EdgeConv(128, 256)
        
        self.mlp = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, embed_dim, 1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.grid_mapping = torch.zeros(2, 7, dtype=torch.long)
        for idx in range(num_teeth):
            row = idx // 7
            col = idx % 7
            self.grid_mapping[row, col] = idx
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3))  # [B*num_teeth, 2048, in_channels]
        
        x = self.conv1(x, self.k)  # [B*num_teeth, 2048, 64]
        x1 = x
        x = self.conv2(x, self.k)  # [B*num_teeth, 2048, 64]
        x2 = x
        x = self.conv3(x, self.k)  # [B*num_teeth, 2048, 128]
        x3 = x
        x = self.conv4(x, self.k)  # [B*num_teeth, 2048, 256]
        x4 = x
        
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # [B*num_teeth, 2048, 512]
        x = x.transpose(1, 2).contiguous()  # [B*num_teeth, 512, 2048]
        x = self.mlp(x)  # [B*num_teeth, embed_dim, 2048]
        x = x.max(dim=-1, keepdim=False)[0]  # [B*num_teeth, embed_dim]
        
        x = x.view(batch_size, self.num_teeth, self.embed_dim)  # [B, num_teeth, embed_dim]
        
        grid_output = torch.zeros(batch_size, 2, 7, self.embed_dim, device=x.device)
        for row in range(2):
            for col in range(7):
                tooth_idx = self.grid_mapping[row, col].item()
                if tooth_idx < self.num_teeth:
                    grid_output[:, row, col] = x[:, tooth_idx]
        
        return grid_output  # [B, 2, 7, embed_dim]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)