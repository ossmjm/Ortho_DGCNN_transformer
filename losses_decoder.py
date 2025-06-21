import torch
import torch.nn as nn

class TransformLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def forward(self, ratios_sequence, ratios, num_stages, device):
        # ratios_sequence: [B, S, T, P], pre-softmaxed predicted ratios
        # ratios: [B, S, T, P], pre-softmaxed ground truth ratios
        # num_stages: [B], number of active stages per batch
        # device: torch.device
        B, S, T, P = ratios_sequence.shape  # B: batch, S: stages, T: teeth, P: params
        # Stage mask: [B, S, 1, 1], True for s < num_stages[b]
        stage_mask = torch.arange(S, device=device)[None, :, None, None] < num_stages[:, None, None, None]
        # Activity mask: [B, S, T, P], True where ratios > 1e-6
        activity_mask = ratios > 1e-6
        # Combined mask: [B, S, T, P], active stages and teeth/params
        mask = stage_mask & activity_mask
        num_active = mask.sum().clamp(min=1)  # Scalar, number of active elements

        # KLDivLoss inputs: log(pred), target (both pre-softmaxed)
        pred = torch.log(ratios_sequence.clamp(min=1e-6))  # [B, S, T, P]
        target = ratios.clamp(min=1e-6)  # [B, S, T, P]
        loss = (self.kl_loss(pred, target) * mask.float()).sum() / num_active  # Scalar
        return self.weight * loss  # Weighted scalar loss

class PaddedLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, ratios_sequence, num_stages, device):
        # ratios_sequence: [B, S, T, P], predicted ratios
        # num_stages: [B], number of active stages
        # device: torch.device
        B, S, T, P = ratios_sequence.shape  # B: batch, S: stages, T: teeth, P: params
        # Padded mask: [B, S, 1, 1], True for s >= num_stages[b]
        padded_mask = torch.arange(S, device=device)[None, :, None, None] >= num_stages[:, None, None, None]
        num_padded = padded_mask.sum().clamp(min=1)  # Scalar, number of padded elements

        # Target: [B, S, T, P], zeros for padded stages
        target = torch.zeros_like(ratios_sequence, device=device)
        pred = ratios_sequence.clamp(min=0.0, max=1.0)  # [B, S, T, P]
        loss = (self.mse_loss(pred, target) * padded_mask.float()).sum() / num_padded  # Scalar
        return self.weight * loss  # Weighted scalar loss

class ConsistencyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, ratios_sequence, ratios, num_stages, device):
        # ratios_sequence: [B, S, T, P], predicted ratios
        # ratios: [B, S, T, P], ground truth ratios
        # num_stages: [B], number of active stages
        # device: torch.device
        B, S, T, P = ratios_sequence.shape  # B: batch, S: stages, T: teeth, P: params
        # Stage mask: [B, S, 1, 1], True for s < num_stages[b]
        stage_mask = torch.arange(S, device=device)[None, :, None, None] < num_stages[:, None, None, None]
        # Activity mask: [B, S, T, P], True where ratios > 1e-6 in active stages
        activity_mask = (ratios > 1e-6) & stage_mask
        # Active teeth/params: [B, T, P], True if active in any stage
        active_tp_mask = activity_mask.any(dim=1)
        num_active_tp = active_tp_mask.sum().clamp(min=1)  # Scalar, number of active teeth/params

        # Sum ratios over active stages per tooth/param: [B, T, P]
        ratios_sum = (ratios_sequence * stage_mask.float()).sum(dim=1)
        pred = ratios_sum.clamp(min=0.0, max=2.0)  # [B, T, P]
        # Target: [B, T, P], 1.0 for active teeth/params, 0.0 otherwise
        target = torch.ones_like(ratios_sum, device=device) * active_tp_mask.float()
        loss = (self.mse_loss(pred, target) * active_tp_mask.float()).sum() / num_active_tp  # Scalar
        return self.weight * loss  # Weighted scalar loss

class DirectionLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, directions_sequence, directions, ratios, num_stages, device):
        # directions_sequence: [B, S, T, P], predicted probabilities (sigmoid-applied)
        # directions: [B, S, T, P], ground truth probabilities (0 or 1)
        # ratios: [B, S, T, P], for activity mask
        # num_stages: [B], number of active stages
        # device: torch.device
        B, S, T, P = directions_sequence.shape  # B: batch, S: stages, T: teeth, P: params
        # Stage mask: [B, S, 1, 1], True for s < num_stages[b]
        stage_mask = torch.arange(S, device=device)[None, :, None, None] < num_stages[:, None, None, None]
        # Activity mask: [B, S, T, P], True where ratios > 1e-6
        activity_mask = ratios > 1e-6
        # Combined mask: [B, S, T, P], active stages and teeth/params
        mask = stage_mask & activity_mask
        num_active = mask.sum().clamp(min=1)  # Scalar, number of active elements

        # Compute class weights based on masked directions
        pos_count = (directions[mask] > 0.5).float().sum().clamp(min=1)  # Scalar, count of positives
        neg_count = (directions[mask] <= 0.5).float().sum().clamp(min=1)  # Scalar, count of negatives
        pos_weight = neg_count / (pos_count + neg_count)  # Scalar, weight for positives
        neg_weight = pos_count / (pos_count + neg_count)  # Scalar, weight for negatives
        # Weight tensor: [B, S, T, P], pos_weight for 1, neg_weight for 0
        weights = torch.where(directions > 0.5, pos_weight, neg_weight).to(device)

        # Compute weighted BCE
        pred = directions_sequence.clamp(min=0.0, max=1.0)  # [B, S, T, P]
        target = directions.clamp(min=0.0, max=1.0)  # [B, S, T, P]
        loss = (self.bce_loss(pred, target) * weights * mask.float()).sum() / num_active  # Scalar
        return self.weight * loss  # Weighted scalar loss

def compute_loss(ratios_sequence, directions_sequence, ratios, directions, num_stages, device, args):
    # ratios_sequence, directions_sequence, ratios, directions: [B, S, T, P]
    # num_stages: [B]
    # args: contains w_trans, w_padded, w_consistency, w_directions
    transform_loss_fn = TransformLoss(weight=args.w_trans).to(device)
    padded_loss_fn = PaddedLoss(weight=args.w_padded).to(device)
    consistency_loss_fn = ConsistencyLoss(weight=args.w_consistency).to(device)
    direction_loss_fn = DirectionLoss(weight=args.w_directions).to(device)

    loss_trans = transform_loss_fn(ratios_sequence, ratios, num_stages, device)
    loss_padded = padded_loss_fn(ratios_sequence, num_stages, device)
    loss_consistency = consistency_loss_fn(ratios_sequence, ratios, num_stages, device)
    loss_directions = direction_loss_fn(directions_sequence, directions, ratios, num_stages, device)

    losses = {
        'loss_trans': loss_trans,
        'loss_padded': loss_padded,
        'loss_consistency': loss_consistency,
        'loss_directions': loss_directions
    }

    total_loss = sum(losses.values())
    # Return: total_loss, losses dict, three placeholder zeros
    return total_loss, losses