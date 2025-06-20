import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_f1_score
import logging
import numpy as np

def binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """Compute binary focal loss for sigmoid outputs with numerical stability."""
    logits = torch.clamp(logits, -10, 10)  # Tighter clamp for stability
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.clamp(torch.exp(-bce), min=1e-6, max=1.0)
    focal_term = alpha * (1 - pt) ** gamma
    loss = focal_term * bce
    return loss.sum() if reduction == 'sum' else loss.mean()

class HybridTransformLoss(nn.Module):
    """Loss for per-stage tooth transformations, using adaptive Huber loss with sparsity term."""
    def __init__(self, weight=1.0, small_error_threshold=0.5, large_delta=5.0, sparse_weight=0.5, max_error=40.0, small_error_scale=5.0):
        super().__init__()
        self.weight = weight
        self.small_error_threshold = small_error_threshold
        self.large_delta = large_delta
        self.sparse_weight = sparse_weight
        self.max_error = max_error
        self.small_delta = 0.1  # Small delta for high precision
        self.small_error_scale = small_error_scale  # Scale for small errors

    def forward(self, pred, target, activity_mask=None, stage_weights=None):
        """Compute loss between predicted and target transformations."""
        logger = logging.getLogger('TrainLogger')
        if not (torch.all(pred.isfinite()) and torch.all(target.isfinite())):
            logger.error("Pred or target contains NaN/Inf")
            raise ValueError("Pred or target contains NaN/Inf")

        error = torch.clamp(pred - target, -self.max_error, self.max_error)
        abs_error = torch.abs(error)

        # Small error Huber loss (high sensitivity)
        small_mask = (abs_error < self.small_error_threshold).float()
        small_loss = torch.where(
            abs_error < self.small_delta,
            0.5 * (error ** 2) / self.small_delta,
            abs_error - 0.5 * self.small_delta
        ) * small_mask * self.small_error_scale  # Use small_error_scale

        # Large error Huber loss (tolerate high values)
        large_loss = torch.where(
            abs_error < self.large_delta,
            0.5 * (error ** 2) / self.large_delta,
            abs_error - 0.5 * self.large_delta
        ) * (1 - small_mask)

        # Sparsity term: penalize non-zero predictions for zero targets
        sparse_mask = (target == 0).float()
        sparse_loss = torch.abs(pred) * sparse_mask * self.sparse_weight

        loss = small_loss + large_loss + sparse_loss

        if activity_mask is not None:
            if not torch.all(activity_mask.isfinite()):
                logger.error("Activity mask contains NaN/Inf")
                raise ValueError("Activity mask contains NaN/Inf")
            loss = loss * activity_mask

        if stage_weights is not None:
            if not torch.all(stage_weights.isfinite()):
                logger.error("Stage weights contain NaN/Inf")
                raise ValueError("Stage weights contain NaN/Inf")
            loss = loss * stage_weights

        num_active = activity_mask.sum() if activity_mask is not None else loss.numel()
        num_active = max(num_active, 1.0)
        loss = self.weight * loss.sum() / num_active

        logger.debug(f"HybridTransformLoss: small_loss={small_loss.mean().item():.4f}, large_loss={large_loss.mean().item():.4f}, sparse_loss={sparse_loss.mean().item():.4f}, total={loss.item():.4f}")
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
            logger.error("Logits or labels contain NaN/Inf")
            raise ValueError("Logits or labels contain NaN/Inf")

        batch_size, max_stages, _ = logits.shape
        device = logits.device

        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        stage_mask = (stage_mask < true_num_stages.unsqueeze(1)).float()

        if not torch.all((labels == 0.0) | (labels == 1.0)):
            logger.warning(f"Non-binary tooth activity labels: {torch.unique(labels).tolist()}")

        masked_logits = logits * stage_mask.unsqueeze(-1)
        masked_labels = labels * stage_mask.unsqueeze(-1)

        # Dynamic alpha based on class imbalance
        pos_ratio = masked_labels.sum() / max(masked_labels.numel(), 1)
        alpha = self.alpha if pos_ratio > 0 else 0.5

        if self.use_focal:
            loss = binary_focal_loss(masked_logits, masked_labels, alpha, self.gamma, reduction='sum')
        else:
            loss = self.bce_loss(masked_logits, masked_labels).sum()

        num_valid = stage_mask.sum() * logits.size(-1)
        loss = self.weight * loss / max(num_valid, 1.0)

        valid_preds = torch.sigmoid(masked_logits[stage_mask.bool().unsqueeze(-1).expand_as(logits)])
        valid_labels = masked_labels[stage_mask.bool().unsqueeze(-1).expand_as(labels)]
        f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5) if valid_preds.numel() > 0 else torch.tensor(0.0, device=device)

        logger.debug(f"ToothActivityLoss: loss={loss.item():.4f}, f1={f1.item():.4f}, valid_elements={num_valid.item()}, alpha={alpha:.4f}")
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
            logger.error("Logits, labels, or tooth_activity_labels contain NaN/Inf")
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

        # Dynamic alpha based on class imbalance
        pos_ratio = masked_labels.sum() / max(masked_labels.numel(), 1)
        alpha = self.alpha if pos_ratio > 0 else 0.5

        if self.use_focal:
            loss = binary_focal_loss(masked_logits, masked_labels, alpha, self.gamma, reduction='sum')
        else:
            loss = self.bce_loss(masked_logits, masked_labels).sum()

        num_active = active_mask.sum()
        loss = self.weight * loss / max(num_active, 1.0)

        valid_preds = torch.sigmoid(masked_logits[active_mask.bool()])
        valid_labels = masked_labels[active_mask.bool()]
        f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5) if valid_preds.numel() > 0 else torch.tensor(0.0, device=device)

        logger.debug(f"ParamActivityLoss: loss={loss.item():.4f}, f1={f1.item():.4f}, active_elements={num_active.item()}, alpha={alpha:.4f}")
        return loss, f1
    
class StageActivityLoss(nn.Module):
    """Loss for predicting binary stage activity with sequential constraint."""
    def __init__(self, weight=1.0, seq_penalty=0.1, use_focal=True, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.seq_penalty = seq_penalty
        self.use_focal = use_focal
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, true_num_stages):
        """Compute loss and F1 score for stage activity predictions."""
        logger = logging.getLogger('TrainLogger')
        if not torch.all(logits.isfinite()):
            logger.error("Stage activity logits contain NaN/Inf")
            raise ValueError("Stage activity logits contain NaN/Inf")

        batch_size, max_stages = logits.shape
        device = logits.device

        # Label active stages (1) and inactive stages (0)
        stage_labels = torch.zeros_like(logits, device=device)
        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        active_mask = (stage_mask < true_num_stages.unsqueeze(1))
        stage_labels[active_mask] = 1.0

        # Debug: Log unique labels to verify
        logger.debug(f"Stage labels unique values: {torch.unique(stage_labels).tolist()}")

        # Dynamic alpha based on class imbalance
        pos_ratio = stage_labels.sum() / max(stage_labels.numel(), 1)
        alpha = 1.0 - pos_ratio if pos_ratio > 0 else 0.5  # Inverse of positive ratio for class weighting

        # Compute loss over all stages
        if self.use_focal:
            loss = binary_focal_loss(logits, stage_labels, alpha, self.gamma, reduction='sum')
        else:
            loss = self.bce_loss(logits, stage_labels).sum()

        loss = loss / max(logits.numel(), 1.0)

        # Sequential penalty: penalize active stages after inactive ones
        preds = torch.sigmoid(logits)
        seq_penalty = torch.zeros(1, device=device)
        for i in range(max_stages - 1):
            invalid_seq = (preds[:, i] < 0.5) & (preds[:, i + 1] > 0.5)
            seq_penalty += invalid_seq.float().sum()
        seq_penalty = self.seq_penalty * seq_penalty / max(batch_size, 1)

        total_loss = self.weight * (loss + seq_penalty)

        # F1 score over all stages
        valid_labels = stage_labels
        logger.debug(f"Valid labels unique values: {torch.unique(valid_labels).tolist()}")
        f1 = binary_f1_score(preds.flatten(), valid_labels.flatten(), threshold=0.5)
        logger.debug(f"StageActivityLoss: loss={loss.item():.4f}, seq_penalty={seq_penalty.item():.4f}, total={total_loss.item():.4f}, f1={f1.item():.4f}, total_stages={logits.numel()}, alpha={alpha:.4f}")
        return total_loss, f1
    
class PaddedLoss(nn.Module):
    """Loss to penalize non-zero transformations in padded stages."""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_transforms, num_stages, max_stages):
        """Compute L1 loss for transformations in padded stages."""
        logger = logging.getLogger('TrainLogger')
        if not torch.all(pred_transforms.isfinite()):
            logger.error("Pred transforms contain NaN/Inf")
            raise ValueError("Pred transforms contain NaN/Inf")

        batch_size = pred_transforms.size(0)
        device = pred_transforms.device

        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        padded_mask = (stage_mask >= num_stages.unsqueeze(1)).float().unsqueeze(-1).unsqueeze(-1)

        loss = (torch.abs(pred_transforms) * padded_mask).sum() / max(padded_mask.sum(), 1.0)
        logger.debug(f"PaddedLoss: loss={loss.item():.4f}, padded_elements={padded_mask.sum().item()}")
        return self.weight * loss

class ConsistencyLoss(nn.Module):
    """Loss to ensure sum of per-stage transformations matches cumulative transforms per tooth and parameter."""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_transforms, cumulative_transforms, num_stages, max_stages, device, use_scaler=False, scalers=None):
        """Compute MSE loss between summed predictions and cumulative transforms per tooth and parameter."""
        logger = logging.getLogger('TrainLogger')
        if not (torch.all(pred_transforms.isfinite()) and torch.all(cumulative_transforms.isfinite())):
            logger.error("Pred or cumulative transforms contain NaN/Inf")
            raise ValueError("Pred or cumulative transforms contain NaN/Inf")

        batch_size, _, num_teeth, num_params = pred_transforms.shape

        # Mask valid stages
        stage_mask = torch.arange(max_stages, device=device).unsqueeze(0).expand(batch_size, max_stages)
        stage_mask = (stage_mask < num_stages.unsqueeze(1)).float().unsqueeze(-1).unsqueeze(-1)

        # Activity mask for cumulative transforms
        active_mask = (cumulative_transforms != 0).float().unsqueeze(1)

        # Initialize loss
        total_loss = 0.0
        count = 0

        # Loop over each tooth and parameter
        for tooth_idx in range(num_teeth):
            for param_idx in range(num_params):
                # Get predictions for this tooth and parameter
                preds = pred_transforms[:, :, tooth_idx, param_idx]  # Shape: (batch_size, max_stages)

                # Apply inverse scaling to predictions if scaler is used
                if use_scaler and scalers is not None and scalers[param_idx] is not None:
                    try:
                        preds_np = preds.cpu().numpy()  # Shape: (batch_size, max_stages)
                        preds_unscaled = scalers[param_idx].inverse_transform(preds_np.reshape(-1, 1)).reshape(preds_np.shape)
                        preds = torch.tensor(preds_unscaled, device=device, dtype=torch.float32)
                    except Exception as e:
                        logger.error(f"Failed to inverse scale parameter {param_idx} for tooth {tooth_idx}: {e}")
                        raise ValueError(f"Failed to inverse scale parameter {param_idx} for tooth {tooth_idx}: {e}")

                # Sum unscaled (or original) predictions over valid stages
                masked_preds = preds * stage_mask.squeeze(-1).squeeze(-1)
                pred_sum = masked_preds.sum(dim=1)  # Shape: (batch_size,)

                target = cumulative_transforms[:, tooth_idx, param_idx]  # Shape: (batch_size,)

                # Apply active mask
                active = active_mask[:, 0, tooth_idx, param_idx]
                loss = F.mse_loss(pred_sum * active, target * active, reduction='sum') / max(active.sum(), 1)

                total_loss += loss

                # Log for debugging
                logger.debug(f"ConsistencyLoss tooth {tooth_idx} param {param_idx}: "
                             f"loss={loss.item():.4f}, pred_sum_mean={pred_sum.mean().item():.4f}, "
                             f"target_mean={target.mean().item():.4f}, active_count={active.sum().item()}")

                count += 1

        # Average the loss
        total_loss = total_loss / max(count, 1)
        logger.debug(f"ConsistencyLoss: total_loss={total_loss.item():.4f}, num_combinations={count}")
        return self.weight * total_loss