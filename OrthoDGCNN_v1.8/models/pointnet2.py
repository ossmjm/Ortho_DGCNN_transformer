import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation

class PointNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, embed_dim=384, num_teeth=14, num_points=256, dropout=0.3):
        super(PointNetPlusPlus, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.num_points = num_points

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=in_channels,
            mlp_list=[[32, 64], [64, 128]],
            points_none=True
        )
        sa1_out_channels = 64 + 128  # 192
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=64,
            radius_list=[0.1, 0.2],
            nsample_list=[32, 64],
            in_channel=sa1_out_channels,
            mlp_list=[[128, 128], [128, 256]],
            points_none=False
        )
        sa2_out_channels = 128 + 256  # 384
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=sa2_out_channels,
            mlp=[256, 512],
            group_all=True,
            include_xyz=False  # Skip XYZ concatenation
        )
        sa3_out_channels = 512

        # Feature Propagation layers
        self.fp1 = PointNetFeaturePropagation(
            in_channel=sa3_out_channels + sa2_out_channels,  # 512 + 384 = 896
            mlp=[256, 256]
        )
        fp1_out_channels = 256
        self.fp2 = PointNetFeaturePropagation(
            in_channel=fp1_out_channels + sa1_out_channels,  # 256 + 192 = 448
            mlp=[128, 128]
        )
        fp2_out_channels = 128

        # Final convolution
        self.conv_final = nn.Sequential(
            nn.Conv1d(fp2_out_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Input:
            x: point cloud data, [B, num_teeth, num_points, in_channels]
        Output:
            x: encoded features, [B, num_teeth, embed_dim]
        """
        logger = logging.getLogger('TrainLogger')
        batch_size, num_teeth, num_points, channels = x.size()
        assert num_teeth == self.num_teeth, f"Expected num_teeth={self.num_teeth}, got {num_teeth}"
        assert num_points == self.num_points, f"Expected num_points={self.num_points}, got {num_points}"
        assert channels == self.in_channels, f"Expected channels={self.in_channels}, got {channels}"
        logger.debug(f"PointNet++ input: [B={batch_size}, T={num_teeth}, N={num_points}, C={channels}]")

        # Reshape for processing
        x = x.view(batch_size * num_teeth, num_points, channels).permute(0, 2, 1).contiguous()  # [B*T, C, N]

        # Set Abstraction
        xyz1, points1 = self.sa1(x, None)  # [B*T, 3, 128], [B*T, 192, 128]
        logger.debug(f"sa1 output: xyz1={xyz1.shape}, points1={points1.shape}")
        xyz2, points2 = self.sa2(xyz1, points1)  # [B*T, 3, 64], [B*T, 384, 64]
        logger.debug(f"sa2 output: xyz2={xyz2.shape}, points2={points2.shape}")
        xyz3, points3 = self.sa3(xyz2, points2)  # [B*T, 3, 1], [B*T, 512, 1]
        logger.debug(f"sa3 output: xyz3={xyz3.shape}, points3={points3.shape}")

        # Feature Propagation
        x = self.fp1(xyz2, xyz3, points2, points3)  # [B*T, 256, 64]
        logger.debug(f"fp1 output: x={x.shape}")
        x = self.fp2(xyz1, xyz2, points1, x)  # [B*T, 128, 128]
        logger.debug(f"fp2 output: x={x.shape}")

        # Final transformation
        x = self.conv_final(x)  # [B*T, embed_dim, 128]
        logger.debug(f"conv_final output: x={x.shape}")
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # [B*T, embed_dim]
        x = x.view(batch_size, num_teeth, self.embed_dim)  # [B, T, embed_dim]
        logger.debug(f"PointNet++ output: [B={batch_size}, T={num_teeth}, D={self.embed_dim}]")
        return x