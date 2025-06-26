import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def compute_f1_score(preds, labels):
    """Compute F1-score for binary predictions per parameter."""
    preds = (preds > 0.5).float()
    tp = (preds * labels).sum(dim=0)  # Sum over batch
    pred_sum = preds.sum(dim=0)
    label_sum = labels.sum(dim=0)
    precision = tp / (pred_sum + 1e-6)
    recall = tp / (label_sum + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1

class CumulativeTranslationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred_translations, cumulative_translations, activity_mask):
        """Compute MAE loss for translation magnitudes."""
        # Compute loss on absolute values
        loss = self.mae_loss(pred_translations, cumulative_translations.abs())
        num_active = activity_mask.sum().clamp(min=1e-6)
        return (loss * activity_mask).sum() / num_active

class CumulativeRotationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred_rotations, cumulative_rotations, activity_mask):
        """Compute MAE loss for rotation magnitudes."""
        # Compute loss on absolute values
        loss = self.mae_loss(pred_rotations, cumulative_rotations.abs())
        num_active = activity_mask.sum().clamp(min=1e-6)
        return (loss * activity_mask).sum() / num_active

class CumulativeDirectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, directions_logits, direction_labels, activity_mask):
        """Compute weighted BCE loss for directions where activity_labels are non-zero."""
        # Compute pos_weight for class imbalance
        pos_samples = (direction_labels * activity_mask).sum(dim=0)  # [num_teeth, num_params]
        neg_samples = (activity_mask - (direction_labels * activity_mask)).sum(dim=0)  # [num_teeth, num_params]
        pos_weight = neg_samples / (pos_samples + 1e-6)  # [num_teeth, num_params]
        pos_weight = pos_weight.clamp(max=100.0)  # Prevent extreme weights
        
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"Direction pos_weight: {[f'{x:.2f}' for x in pos_weight.mean(dim=0)]}")
        
        # Compute weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            directions_logits,
            direction_labels,
            reduction='none',
            pos_weight=pos_weight
        )
        loss = loss * activity_mask  # Apply activity mask
        num_active = activity_mask.sum().clamp(min=1e-6)
        loss = loss.sum() / num_active
        
        # Compute F1 scores
        batch_size, num_teeth, num_params = directions_logits.shape
        f1_scores = compute_f1_score(
            torch.sigmoid(directions_logits).view(batch_size, -1),
            direction_labels.view(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)
        
        return loss, f1_scores

class CumulativeZeroLoss(nn.Module):
    def __init__(self, lambda_zero=1e-3):
        super().__init__()
        self.lambda_zero = lambda_zero
        self.mae_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred_translations, pred_rotations, cumulative_transforms):
        """Compute MAE loss for zero elements in cumulative_transforms."""
        # Identify zero elements (inactive) in cumulative_transforms
        zero_mask = (cumulative_transforms.abs() < 1e-6).float()  # [batch_size, 14, 6]
        # Split into translation and rotation
        zero_mask_trans = zero_mask[:, :, :3]  # [batch_size, 14, 3]
        zero_mask_rot = zero_mask[:, :, 3:]    # [batch_size, 14, 3]
        # Compute MAE loss only for zero elements
        trans_loss = self.mae_loss(pred_translations, torch.zeros_like(pred_translations))
        rot_loss = self.mae_loss(pred_rotations, torch.zeros_like(pred_rotations))
        trans_loss = (trans_loss * zero_mask_trans).sum()
        rot_loss = (rot_loss * zero_mask_rot).sum()
        num_zero = zero_mask.sum().clamp(min=1e-6)
        return (trans_loss + rot_loss) / num_zero

class CumulativeL1Regularization(nn.Module):
    def __init__(self, lambda_l1=1e-3):
        super().__init__()
        self.lambda_l1 = lambda_l1
    
    def forward(self, trans_magnitude, rot_magnitude):
        """Compute L1 regularization on magnitude predictions."""
        return self.lambda_l1 * (trans_magnitude.abs().sum() + rot_magnitude.abs().sum())

def compute_loss(trans_magnitude, rot_magnitude, directions_logits,
                 cumulative_transforms, direction_labels, activity_labels, device, logger, args):
    """Compute all loss components and total loss for training."""
    # Move inputs to device
    cumulative_transforms = cumulative_transforms.to(device)
    direction_labels = direction_labels.to(device)
    activity_labels = activity_labels.to(device)
    trans_magnitude, rot_magnitude, directions_logits = [
        x.to(device) for x in [trans_magnitude, rot_magnitude, directions_logits]
    ]
    
    # Split activity_labels and cumulative_transforms into trans and rot
    activity_mask_trans = activity_labels[:, :, :3]  # [batch_size, 14, 3]
    activity_mask_rot = activity_labels[:, :, 3:]    # [batch_size, 14, 3]
    cumulative_transforms_trans = cumulative_transforms[:, :, :3]  # [batch_size, 14, 3]
    cumulative_transforms_rot = cumulative_transforms[:, :, 3:]    # [batch_size, 14, 3]
    
    # Initialize loss functions
    translation_loss_fn = CumulativeTranslationLoss().to(device)
    rotation_loss_fn = CumulativeRotationLoss().to(device)
    direction_loss_fn = CumulativeDirectionLoss().to(device)
    zero_loss_fn = CumulativeZeroLoss().to(device)
    l1_loss_fn = CumulativeL1Regularization().to(device)
    
    # Log prediction statistics
    logger.debug(f"Trans mag min: {trans_magnitude.min():.4f}, max: {trans_magnitude.max():.4f}, "
                 f"has_nan: {torch.isnan(trans_magnitude).any()}")
    logger.debug(f"Rot mag min: {rot_magnitude.min():.4f}, max: {rot_magnitude.max():.4f}, "
                 f"has_nan: {torch.isnan(rot_magnitude).any()}")
    
    # Compute individual losses
    loss_trans = translation_loss_fn(trans_magnitude, cumulative_transforms_trans, activity_mask_trans)
    loss_rot = rotation_loss_fn(rot_magnitude, cumulative_transforms_rot, activity_mask_rot)
    loss_direction, direction_f1_scores = direction_loss_fn(directions_logits, direction_labels, activity_labels)
    loss_zero = zero_loss_fn(trans_magnitude, rot_magnitude, cumulative_transforms)
    loss_l1 = l1_loss_fn(trans_magnitude, rot_magnitude)
    
    # Collect losses and metrics
    losses = {
        'loss_trans': loss_trans,
        'loss_rot': loss_rot,
        'loss_direction': loss_direction,
        'loss_zero': loss_zero,
        'loss_l1': loss_l1,
        'direction_f1_scores': direction_f1_scores
    }
    
    # Check for NaN/Inf
    for name, loss in losses.items():
        if isinstance(loss, torch.Tensor) and (torch.isnan(loss).any() or torch.isinf(loss).any()):
            logger.error(f"{name} is NaN or Inf: {loss.item()}")
    
    # Compute total loss
    total_loss = (
        args.w_trans * loss_trans +
        args.w_rot * loss_rot +
        args.w_direction * loss_direction +
        args.w_zero * loss_zero +
        args.w_l1 * loss_l1
    )
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Total loss is NaN or Inf: {total_loss.item()}")
    
    return total_loss, losses