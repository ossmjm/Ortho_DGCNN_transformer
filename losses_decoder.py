import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_f1_score
import logging

def binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """Compute binary focal loss for sigmoid outputs with numerical stability."""
    logits = torch.clamp(logits, -50, 50)  # Reduced clamp range for stability
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.clamp(torch.exp(-bce), min=1e-6, max=1.0)
    focal_term = alpha * (1 - pt) ** gamma
    loss = focal_term * bce
    return loss.sum() if reduction == 'sum' else loss.mean()

class HybridTransformLoss(nn.Module):
    """Loss for per-stage tooth transformations, combining log-cosh and scaled MSE."""
    def __init__(self, weight=1.0, small_error_threshold=1.0, small_error_scale=5.0, max_error=40.0):
        super().__init__()
        self.weight = weight
        self.small_error_threshold = small_error_threshold
        self.small_error_scale = small_error_scale
        self.max_error = max_error

    def forward(self, pred, target, activity_mask=None, stage_weights=None):
        """Compute loss between predicted and target transformations."""
        if not (torch.all(pred.isfinite()) and torch.all(target.isfinite())):
            raise ValueError("Pred or target contains NaN/Inf")

        error = torch.clamp(pred - target, -self.max_error, self.max_error)
        abs_error = torch.abs(error)

        log_cosh = torch.log(torch.cosh(error + 1e-6))
        mse = error ** 2
        small_error_mask = (abs_error < self.small_error_threshold).float()
        loss = log_cosh + mse * (self.small_error_scale * small_error_mask + 1.0)

        if activity_mask is not None:
            if not torch.all(activity_mask.isfinite()):
                raise ValueError("Activity mask contains NaN/Inf")
            loss = loss * activity_mask

        if stage_weights is not None:
            if not torch.all(stage_weights.isfinite()):
                raise ValueError("Stage weights contain NaN/Inf")
            loss = loss * stage_weights

        num_active = activity_mask.sum() if activity_mask is not None else loss.numel()
        num_active = max(num_active, 1.0)
        loss = self.weight * loss.sum() / num_active

        return loss if loss.isfinite() else torch.tensor(0.0, device=pred.device, requires_grad=True)

class ToothActivityLoss(nn.Module):
    """Loss for predicting binary tooth activity per stage."""
    def __init__(self, weight=1.0, use_focal=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, true_num_stages):
        """Compute loss and F1 score for tooth activity predictions."""
        logger = logging.getLogger('TrainLogger')
        if not (torch.all(logits.isfinite()) and torch.all(labels.isfinite())):
            raise ValueError("Logits or labels contain NaN/Inf")

        batch_size, max_stages, _ = logits.shape
        device = logits.device

        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        stage_mask = (stage_mask < true_num_stages.unsqueeze(1)).float()

        if not torch.all((labels == 0.0) | (labels == 1.0)):
            logger.warning(f"Non-binary tooth activity labels: {torch.unique(labels).tolist()}")

        masked_logits = logits * stage_mask.unsqueeze(-1)
        masked_labels = labels * stage_mask.unsqueeze(-1)

        if self.use_focal:
            loss = binary_focal_loss(masked_logits, masked_labels, self.alpha, self.gamma, reduction='sum')
        else:
            loss = self.bce_loss(masked_logits, masked_labels).sum()

        num_valid = stage_mask.sum() * logits.size(-1)
        loss = self.weight * loss / max(num_valid, 1.0)

        valid_preds = torch.sigmoid(masked_logits[stage_mask.bool().unsqueeze(-1).expand_as(logits)])
        valid_labels = masked_labels[stage_mask.bool().unsqueeze(-1).expand_as(labels)]
        f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5) if valid_preds.numel() > 0 else torch.tensor(0.0, device=device)

        logger.debug(f"ToothActivityLoss: loss={loss.item():.4f}, f1={f1.item():.4f}, valid_elements={num_valid.item()}")
        return loss, f1

class ParamActivityLoss(nn.Module):
    """Loss for predicting binary parameter activity for active teeth."""
    def __init__(self, weight=1.0, use_focal=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, tooth_activity_labels, true_num_stages):
        """Compute loss and F1 score for parameter activity predictions."""
        logger = logging.getLogger('TrainLogger')
        if not (torch.all(logits.isfinite()) and torch.all(labels.isfinite()) and torch.all(tooth_activity_labels.isfinite())):
            raise ValueError("Logits, labels, or tooth_activity_labels contain NaN/Inf")

        batch_size, max_stages, num_teeth, num_params = logits.shape
        device = logits.device

        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        stage_mask = (stage_mask < true_num_stages.unsqueeze(1)).float().unsqueeze(-1).unsqueeze(-1)

        if not torch.all((labels == 0.0) | (labels == 1.0)):
            logger.warning(f"Non-binary param activity labels: {torch.unique(labels).tolist()}")

        tooth_active_mask = (tooth_activity_labels == 1.0).float().unsqueeze(-1).expand(-1, -1, -1, num_params)
        active_mask = tooth_active_mask * stage_mask
        masked_logits = logits * active_mask
        masked_labels = labels * active_mask

        if self.use_focal:
            loss = binary_focal_loss(masked_logits, masked_labels, self.alpha, self.gamma, reduction='sum')
        else:
            loss = self.bce_loss(masked_logits, masked_labels).sum()

        num_active = active_mask.sum()
        loss = self.weight * loss / max(num_active, 1.0)

        valid_preds = torch.sigmoid(masked_logits[active_mask.bool()])
        valid_labels = masked_labels[active_mask.bool()]
        f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5) if valid_preds.numel() > 0 else torch.tensor(0.0, device=device)

        logger.debug(f"ParamActivityLoss: loss={loss.item():.4f}, f1={f1.item():.4f}, active_elements={num_active.item()}")
        return loss, f1
class StageActivityLoss(nn.Module):
    """Loss for predicting binary stage activity."""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, true_num_stages):
        """Compute loss and F1 score for stage activity predictions."""
        logger = logging.getLogger('TrainLogger')
        if not torch.all(logits.isfinite()):
            raise ValueError("Stage activity logits contain NaN/Inf")

        batch_size, max_stages = logits.shape
        device = logits.device

        # Label active stages (1) and inactive stages (0)
        stage_labels = torch.zeros_like(logits)
        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        stage_labels[stage_mask < true_num_stages.unsqueeze(1)] = 1.0

        # Compute loss over all stages
        loss = self.bce_loss(logits, stage_labels)
        loss = loss.sum() / logits.numel()

        # Compute F1 score over all stages
        valid_preds = torch.sigmoid(logits)
        valid_labels = stage_labels
        f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5) if valid_preds.numel() > 0 else torch.tensor(0.0, device=device)

        logger.debug(f"StageActivityLoss: loss={loss.item():.4f}, f1={f1.item():.4f}, total_stages={logits.numel()}")
        return self.weight * loss, f1

class PaddedLoss(nn.Module):
    """Loss to penalize non-zero transformations in padded stages."""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_transforms, num_stages, max_stages):
        """Compute MSE for transformations in padded stages."""
        if not torch.all(pred_transforms.isfinite()):
            raise ValueError("Pred transforms contain NaN/Inf")

        batch_size = pred_transforms.size(0)
        device = pred_transforms.device

        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        padded_mask = (stage_mask >= num_stages.unsqueeze(1)).float().unsqueeze(-1).unsqueeze(-1)

        loss = (pred_transforms ** 2 * padded_mask).sum() / max(padded_mask.sum(), 1.0)
        return self.weight * loss

class ConsistencyLoss(nn.Module):
    """Loss to ensure sum of per-stage transformations matches cumulative transforms per tooth and parameter."""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_transforms, cumulative_transforms, num_stages, max_stages, device):
        """Compute L1 loss between summed predictions and cumulative transforms per tooth and parameter."""
        logger = logging.getLogger('TrainLogger')
        if not (torch.all(pred_transforms.isfinite()) and torch.all(cumulative_transforms.isfinite())):
            raise ValueError("Pred or cumulative transforms contain NaN/Inf")

        batch_size, _, num_teeth, num_params = pred_transforms.shape

        # Mask valid stages
        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        stage_mask = (stage_mask < num_stages.unsqueeze(1)).float().unsqueeze(-1).unsqueeze(-1)

        # Initialize loss
        total_loss = 0.0
        count = 0

        # Loop over each tooth and parameter
        for tooth_idx in range(num_teeth):
            for param_idx in range(num_params):
                # Sum predictions over valid stages for this tooth and parameter
                masked_preds = pred_transforms[:, :, tooth_idx, param_idx] * stage_mask.squeeze(-1).squeeze(-1)
                pred_sum = masked_preds.sum(dim=1)  # Shape: (batch_size,)
                target = cumulative_transforms[:, tooth_idx, param_idx]  # Shape: (batch_size,)
                
                # Compute L1 loss for this tooth-parameter pair
                loss = F.l1_loss(pred_sum, target, reduction='mean')
                total_loss += loss
                
                # Log loss for debugging
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"ConsistencyLoss tooth {tooth_idx} param {param_idx}: "
                                f"loss={loss.item():.4f}, pred_sum_mean={pred_sum.mean().item():.4f}, "
                                f"target_mean={target.mean().item():.4f}")
                
                count += 1

        # Average the loss over all tooth-parameter pairs
        total_loss = total_loss / max(count, 1)

        logger.debug(f"ConsistencyLoss: total_loss={total_loss.item():.4f}, "
                    f"num_combinations={count}")

        return self.weight * total_loss