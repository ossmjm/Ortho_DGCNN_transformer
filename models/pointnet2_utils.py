import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

def pc_normalize(pc):
    """
    Normalize point cloud by centering and scaling.
    Input: pc [N, C]
    Output: normalized pc [N, C]
    """
    centroid = torch.mean(pc, dim=0, keepdim=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
    pc = pc / (m + 1e-8)
    return pc

def square_distance(src, dst):
    """
    Calculate Euclidean distance between each pair of points.
    Input:
        src: [B, N, C]
        dst: [B, M, C]
    Output:
        dist: [B, N, M]
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Index points using given indices.
    Input:
        points: [B, N, C]
        idx: [B, S]
    Output:
        new_points: [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Sample points using farthest point sampling.
    Input:
        xyz: [B, N, C]
        npoint: number of samples
    Output:
        centroids: [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Find points within a ball of radius around query points.
    Input:
        radius: float
        nsample: int
        xyz: [B, N, C]
        new_xyz: [B, S, C]
    Output:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Sample and group points.
    Input:
        npoint: int
        radius: float
        nsample: int
        xyz: [B, N, C]
        points: [B, N, D] or None
    Output:
        new_xyz: [B, npoint, C]
        new_points: [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Group all points.
    Input:
        xyz: [B, N, C]
        points: [B, N, D] or None
    Output:
        new_xyz: [B, 1, C]
        new_points: [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3 if not group_all else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, groups=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: [B, C, N]
            points: [B, D, N] or None
        Output:
            new_xyz: [B, C, S]
            new_points: [B, D', S]
        """
        logger = logging.getLogger('TrainLogger')
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        B, N, C = xyz.shape
        if points is not None:
            _, _, D = points.shape
            logger.debug(f"SA input: xyz=[{B}, {N}, {C}], points=[{B}, {N}, {D}]")
        else:
            logger.debug(f"SA input: xyz=[{B}, {N}, {C}], points=None")

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.leaky_relu(bn(conv(new_points)), negative_slope=0.2)
        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, npoint]
        logger.debug(f"SA output: new_xyz=[{B}, {C}, {new_points.shape[2]}], new_points=[{B}, {new_points.shape[1]}, {new_points.shape[2]}]")
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, points_none=True):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.points_none = points_none
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for mlp in mlp_list:
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp:
                convs.append(nn.Conv2d(last_channel, out_channel, 1, groups=1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: [B, C, N]
            points: [B, D, N] or None
        Output:
            new_xyz: [B, C, S]
            new_points: [B, D', S]
        """
        logger = logging.getLogger('TrainLogger')
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        B, N, C = xyz.shape
        if points is not None:
            _, _, D = points.shape
            logger.debug(f"SAMsg input: xyz=[{B}, {N}, {C}], points=[{B}, {N}, {D}]")
        else:
            logger.debug(f"SAMsg input: xyz=[{B}, {N}, {C}], points=None")

        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, (radius, nsample) in enumerate(zip(self.radius_list, self.nsample_list)):
            group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
            else:
                grouped_points = grouped_xyz  # Use xyz as features if points is None

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C or D, K, S]
            for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                grouped_points = F.leaky_relu(bn(conv(grouped_points)), negative_slope=0.2)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        logger.debug(f"SAMsg output: new_xyz=[{B}, {C}, {S}], new_points=[{B}, {new_points_concat.shape[1]}, {S}]")
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, groups=1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: [B, C, N]
            xyz2: [B, C, S]
            points1: [B, D1, N] or None
            points2: [B, D2, S]
        Output:
            new_points: [B, D', N]
        """
        logger = logging.getLogger('TrainLogger')
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]
        B, N, C1 = xyz1.shape
        _, S, C2 = xyz2.shape
        logger.debug(f"FP input: xyz1=[{B}, {N}, {C1}], xyz2=[{B}, {S}, {C2}]")

        points2 = points2.permute(0, 2, 1)  # [B, S, D2]
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, N, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # [B, D+D', N]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.leaky_relu(bn(conv(new_points)), negative_slope=0.2)
        logger.debug(f"FP output: new_points=[{B}, {new_points.shape[1]}, {N}]")
        return new_points