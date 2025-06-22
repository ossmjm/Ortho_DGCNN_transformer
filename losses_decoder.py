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
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, directions_sequence, directions, ratios, num_stages, device):
        # directions_sequence: [B, S, T, P], predicted logits (pre-sigmoid)
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

        # Compute pos_weight for class imbalance
        pos_count = (directions[mask] > 0.5).float().sum().clamp(min=1)  # Scalar, count of positives
        neg_count = (directions[mask] <= 0.5).float().sum().clamp(min=1)  # Scalar, count of negatives
        pos_weight = neg_count / pos_count  # Scalar, weight for positives
        weights = torch.ones_like(directions, device=device) * pos_weight  # [B, S, T, P]

        # Compute weighted BCE with logits
        pred = directions_sequence  # [B, S, T, P], logits
        target = directions  # [B, S, T, P]
        loss = (self.bce_loss(pred, target) * weights * mask.float()).sum() / num_active  # Scalar

        # Compute F1 score for active elements
        pred_probs = torch.sigmoid(pred)  # [B, S, T, P], probabilities
        pred_binary = (pred_probs > 0.5).float()  # [B, S, T, P], binary predictions
        true_binary = (target > 0.5).float()  # [B, S, T, P], binary ground truth
        tp = (pred_binary * true_binary * mask.float()).sum()  # True positives
        fp = (pred_binary * (1 - true_binary) * mask.float()).sum()  # False positives
        fn = ((1 - pred_binary) * true_binary * mask.float()).sum()  # False negatives
        precision = tp / (tp + fp + 1e-6)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-6)  # Avoid division by zero
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # F1 score

        return self.weight * loss, f1  # Return weighted loss and F1 score

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
    loss_directions, f1_directions = direction_loss_fn(directions_sequence, directions, ratios, num_stages, device)

    losses = {
        'loss_trans': loss_trans,
        'loss_padded': loss_padded,
        'loss_consistency': loss_consistency,
        'loss_directions': loss_directions,
        'loss_directions_f1': f1_directions
    }

    total_loss = sum(l for k, l in losses.items() if k != 'loss_directions_f1')  # Exclude F1 from total loss
    return total_loss, losses