import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.pointnet2_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg,PointNetFeaturePropagation

class PointNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, num_teeth=14, num_points=1000, dropout=0.5):
        super(PointNetPlusPlus, self).__init__()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Set Abstraction layers for local feature extraction
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.1, 0.2],
            nsample_list=[16, 32],
            in_channel=in_channels,
            mlp_list=[[64, 64], [64, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=64,
            radius_list=[0.2, 0.4],
            nsample_list=[32, 64],
            in_channel=64 + 128 + 3,  # Previous features + xyz
            mlp_list=[[128, 128], [128, 256]]
        )
        # Global feature extraction
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=128 + 256 + 3,
            mlp=[256, 512],
            group_all=True
        )

        # Feature propagation for combining local and global features
        self.fp1 = PointNetFeaturePropagation(
            in_channel=512 + 128 + 256,
            mlp=[256, 256]
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=256 + 64 + 128,
            mlp=[128, 128]
        )

        # Final convolution to match DGCNN output
        self.conv_final = nn.Sequential(
            nn.Conv1d(128, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim, momentum=0.01),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
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

        logger.debug(f"PointNet++ input shape: {x.shape}")

        # Reshape to process each tooth independently
        x = x.view(batch_size * num_teeth, num_points, channels).permute(0, 2, 1).contiguous()

        # Set Abstraction layers
        xyz1, points1 = self.sa1(x, None)
        xyz2, points2 = self.sa2(xyz1, points1)
        xyz3, points3 = self.sa3(xyz2, points2)

        # Feature propagation
        x = self.fp1(xyz2, xyz3, points2, points3)
        x = self.fp2(x, xyz2, None, x)

        # Final transformation to match output dimension
        x = self.conv_final(x)

        # Reshape to match DGCNN output
        x = x.view(batch_size, num_teeth, self.embed_dim)

        logger.debug(f"PointNet++ output shape: {x.shape}")
        return x