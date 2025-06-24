import torch
import torch.nn as nn
import logging

class TransformLoss(nn.Module):
    def __init__(self):
        super().__init__()
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
        return loss  # Weighted scalar loss

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
        return loss  # Weighted scalar loss

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
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
        return loss  # Weighted scalar loss

class DirectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
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

        return loss, f1  # Return weighted loss and F1 score

class NumStagesLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # Mapping of num_stages values to contiguous indices (0 to 20)
        self.num_stages_map = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
            11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 19: 17,
            20: 18, 21: 19, 23: 20
        }
        self.max_valid_stage = 23  # Maximum valid num_stages value
        self.num_classes = 21  # Number of classes (0 to 20)

    def forward(self, num_stages_logits, num_stages, device):
        # num_stages_logits: [B, num_classes], predicted logits for 21 classes
        # num_stages: [B], ground truth number of stages
        # device: torch.device
        logger = logging.getLogger('TrainLogger')
        B, num_classes = num_stages_logits.shape  # B: batch, num_classes: 21

        # Clamp num_stages to valid range and convert to mapped indices
        num_stages = num_stages.clamp(min=1, max=self.max_valid_stage).long()
        mapped_stages = torch.zeros_like(num_stages, device=device, dtype=torch.long)
        unmapped_values = []

        for i in range(B):
            stage = num_stages[i].item()
            if stage in self.num_stages_map:
                mapped_stages[i] = self.num_stages_map[stage]
            else:
                unmapped_values.append(stage)
                mapped_stages[i] = self.num_stages_map[self.max_valid_stage]  # Default to max stage

        if unmapped_values:
            logger.warning(f"Unmapped num_stages values encountered: {unmapped_values}. Defaulting to max stage index.")

        # Convert mapped_stages to one-hot encoding: [B, num_classes]
        target = torch.zeros(B, num_classes, device=device)
        target[torch.arange(B, device=device), mapped_stages] = 1.0

        # Compute class weights to handle imbalance
        class_counts = target.sum(dim=0).clamp(min=1.0)  # [num_classes]
        total_samples = B
        class_weights = total_samples / (num_classes * class_counts)  # [num_classes]
        class_weights = class_weights.clamp(min=0.1, max=10.0)  # Clip to avoid extreme weights
        weights = class_weights.unsqueeze(0).expand(B, -1)  # [B, num_classes]

        # Compute weighted BCE with logits
        pred = num_stages_logits  # [B, num_classes], logits
        loss = (self.bce_loss(pred, target) * weights).sum() / B  # Scalar

        # Compute F1 score
        pred_probs = torch.sigmoid(pred)  # [B, num_classes], probabilities
        pred_binary = (pred_probs > 0.5).float()  # [B, num_classes], binary predictions
        true_binary = target  # [B, num_classes], binary ground truth
        tp = (pred_binary * true_binary).sum()  # True positives
        fp = (pred_binary * (1 - true_binary)).sum()  # False positives
        fn = ((1 - pred_binary) * true_binary).sum()  # False negatives
        precision = tp / (tp + fp + 1e-6)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-6)  # Avoid division by zero
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # F1 score

        return loss, f1  # Return weighted loss and F1 score

def compute_loss(ratios_sequence, directions_sequence, num_stages_logits, ratios, directions, num_stages, device, args):
    # ratios_sequence, directions_sequence, ratios, directions: [B, S, T, P]
    # num_stages_logits: [B, num_classes]
    # num_stages: [B]
    # args: contains w_trans, w_padded, w_consistency, w_directions, w_num_stages
    transform_loss_fn = TransformLoss().to(device)
    padded_loss_fn = PaddedLoss().to(device)
    consistency_loss_fn = ConsistencyLoss().to(device)
    direction_loss_fn = DirectionLoss().to(device)
    num_stages_loss_fn = NumStagesLoss().to(device)

    loss_trans = transform_loss_fn(ratios_sequence, ratios, num_stages, device)
    loss_padded = padded_loss_fn(ratios_sequence, num_stages, device)
    loss_consistency = consistency_loss_fn(ratios_sequence, ratios, num_stages, device)
    loss_directions, f1_directions = direction_loss_fn(directions_sequence, directions, ratios, num_stages, device)
    loss_num_stages, f1_num_stages = num_stages_loss_fn(num_stages_logits, num_stages, device)

    losses = {
        'loss_trans': loss_trans,
        'loss_padded': loss_padded,
        'loss_consistency': loss_consistency,
        'loss_directions': loss_directions,
        'loss_directions_f1': f1_directions,
        'loss_num_stages': loss_num_stages,
        'loss_num_stages_f1': f1_num_stages
    }
    total_loss = (
        loss_trans * args.w_trans +
        loss_padded * args.w_padded +
        loss_consistency * args.w_consistency +
        loss_directions * args.w_directions +
        loss_num_stages * args.w_num_stages
    )

    return total_loss, losses