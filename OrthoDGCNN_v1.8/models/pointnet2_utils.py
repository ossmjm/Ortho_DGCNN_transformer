import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def square_distance(src, dst):
    """
    Calculate Euclidean distance between each pair of points.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Output:
        new_points: indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: point cloud data, [B, N, C]
        npoint: number of samples
    Output:
        centroids: sampled point cloud indices, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Output:
        group_idx: grouped points indices, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint: number of sampled points
        radius: local region radius
        nsample: max sample number in local region
        xyz: input points position, [B, N, C]
        points: input points data, [B, N, D]
    Output:
        new_xyz: sampled points position, [B, npoint, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, npoint, nsample, C]
    if points is not None:
        grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

def sample_and_group_all(xyz, points, include_xyz=True):
    """
    Input:
        xyz: input points position, [B, N, C]
        points: input points data, [B, N, D]
        include_xyz: whether to concatenate xyz coordinates
    Output:
        new_xyz: sampled points position, [B, 1, C]
        new_points: sampled points data, [B, 1, N, D] or [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        if include_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, 1, N, C+D]
        else:
            new_points = grouped_points  # [B, 1, N, D]
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, include_xyz=True):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.include_xyz = include_xyz
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position, [B, C, N]
            points: input points data, [B, D, N]
        Output:
            new_xyz: sampled points position, [B, C, S]
            new_points: sampled points features, [B, D', S]
        """
        logger = logging.getLogger('TrainLogger')
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        B, N, C = xyz.shape
        if points is not None:
            _, _, D = points.shape
            logger.debug(f"PointNetSetAbstraction input: xyz=[{B}, {N}, {C}], points=[{B}, {N}, {D}]")
        else:
            logger.debug(f"PointNetSetAbstraction input: xyz=[{B}, {N}, {C}], points=None")

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points, include_xyz=self.include_xyz)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        logger.debug(f"PointNetSetAbstraction new_points before conv: {new_points.shape}")
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, dim=2)[0]  # [B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, npoint]
        logger.debug(f"PointNetSetAbstraction output: new_xyz=[{B}, {C}, {new_xyz.shape[2]}], new_points=[{B}, {new_points.shape[1]}, {new_xyz.shape[2]}]")
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, points_none=False):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.points_none = points_none
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel if points_none else in_channel + 3  # Include XYZ if not points_none
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position, [B, C, N]
            points: input points data, [B, D, N]
        Output:
            new_xyz: sampled points position, [B, C, S]
            new_points: sampled points features, [B, D', S]
        """
        logger = logging.getLogger('TrainLogger')
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        B, N, C = xyz.shape
        if points is not None:
            _, _, D = points.shape
            logger.debug(f"PointNetSetAbstractionMsg input: xyz=[{B}, {N}, {C}], points=[{B}, {N}, {D}]")
        else:
            logger.debug(f"PointNetSetAbstractionMsg input: xyz=[{B}, {N}, {C}], points=None")

        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  # [B, S, C]
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  # [B, S, K]
            grouped_xyz = index_points(xyz, group_idx)  # [B, S, K, C]
            grouped_xyz -= new_xyz.view(B, S, 1, C)  # [B, S, K, C]
            if points is not None and not self.points_none:
                grouped_points = index_points(points, group_idx)  # [B, S, K, D]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, S, K, D+C]
            else:
                grouped_points = grouped_xyz  # [B, S, K, C]

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C or D+C, K, S]
            logger.debug(f"PointNetSetAbstractionMsg grouped_points shape: {grouped_points.shape}")
            for j, conv in enumerate(self.conv_blocks[i]):
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, dim=2)[0]  # [B, out_channel, S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, S]
        new_points = torch.cat(new_points_list, dim=1)  # [B, D', S]
        logger.debug(f"PointNetSetAbstractionMsg output: new_xyz=[{B}, {C}, {S}], new_points=[{B}, {new_points.shape[1]}, {S}]")
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position, [B, C, N]
            xyz2: sampled points position, [B, C, S]
            points1: input points data, [B, D1, N]
            points2: input points data, [B, D2, S]
        Output:
            new_points: upsampled points features, [B, D', N]
        """
        logger = logging.getLogger('TrainLogger')
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]
        B, N, C1 = xyz1.shape
        _, S, C2 = xyz2.shape
        logger.debug(f"PointNetFeaturePropagation input: xyz1=[{B}, {N}, {C1}], xyz2=[{B}, {S}, {C2}]")
        assert C1 == C2 == 3, f"Expected xyz1 and xyz2 to have 3 channels, got {C1} and {C2}"

        points2 = points2.permute(0, 2, 1)  # [B, S, D2]
        _, _, D2 = points2.shape
        logger.debug(f"PointNetFeaturePropagation points: points1={points1.shape if points1 is not None else None}, points2=[{B}, {S}, {D2}]")

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)  # [B, N, D2]
        else:
            dists = square_distance(xyz1, xyz2)  # [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # [B, N, D2]

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, N, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D1+D2]
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # [B, D1+D2, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        logger.debug(f"PointNetFeaturePropagation output: new_points=[{B}, {new_points.shape[1]}, {N}]")
        return new_points