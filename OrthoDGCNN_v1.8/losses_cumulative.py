import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def compute_f1_score(preds, labels, threshold=0.5):
    """Compute F1-score for binary predictions per parameter with dynamic threshold."""
    preds = (preds > threshold).float()
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
        self.huber_loss = nn.HuberLoss(reduction='none', delta=0.1)
    
    def forward(self, pred_translations, cumulative_translations, activity_mask):
        """Compute Huber loss for translation magnitudes, normalized by active elements."""
        loss = self.huber_loss(pred_translations, cumulative_translations.abs())
        num_active = activity_mask.sum().clamp(min=1e-6)
        return (loss * activity_mask).sum() / num_active

class CumulativeRotationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber_loss = nn.HuberLoss(reduction='none', delta=0.1)
    
    def forward(self, pred_rotations, cumulative_rotations, activity_mask):
        """Compute Huber loss for rotation magnitudes, normalized by active elements."""
        loss = self.huber_loss(pred_rotations, cumulative_rotations.abs())
        num_active = activity_mask.sum().clamp(min=1e-6)
        return (loss * activity_mask).sum() / num_active

class CumulativeActiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, active_pred, active_labels):
        """Compute weighted BCE loss for active/inactive classification across all transformations."""

        # Compute positive and negative samples per parameter (P)
        pos_samples = (active_labels).sum(dim=(0, 1))  # Shape [P]
        neg_samples = ((1 - active_labels)).sum(dim=(0, 1))  # Shape [P]

        # Compute pos_weight for BCE loss to handle imbalance
        pos_weight = neg_samples / (pos_samples + 1e-6)
        # print(active_labels)
        # print(f"Positive samples per param: {pos_samples}")
        # print(f"Negative samples per param: {neg_samples}")
        # print(f"Computed pos_weight per param: {pos_weight}")
        pos_weight = pos_weight.clamp(max=5.0)  # Avoid extreme values
        
        # Log stats
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"Positive samples per param: {pos_samples}")
        logger.debug(f"Negative samples per param: {neg_samples}")
        logger.debug(f"Computed pos_weight per param: {pos_weight}")

        # Compute weighted BCE with matching shapes
        loss = F.binary_cross_entropy_with_logits(
            active_pred,  # Shape [batch_size, num_teeth, 6]
            active_labels,  # Shape [batch_size, num_teeth, 6]
            reduction='none',
            pos_weight=pos_weight
        )
        num_active = active_labels.sum().clamp(min=1e-6)
        batch_size, num_teeth, num_params = active_pred.shape

        f1_scores = compute_f1_score(
            torch.sigmoid(active_pred).view(batch_size, -1),
            active_labels.view(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)

        return loss.sum() / num_active,f1_scores

class CumulativeDirectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, direction_pred, direction_labels, activity_mask):
        """Compute weighted BCE loss for directions where activity_labels are non-zero."""
        # Compute pos_weight per direction parameter
        pos_samples = direction_labels.sum(dim=(0, 1))  # [P]
        neg_samples = (1 - direction_labels).sum(dim=(0, 1))  # [P]
        
        pos_weight = neg_samples / (pos_samples + 1e-6)
        pos_weight = pos_weight.clamp(max=5.0)

        logger = logging.getLogger('TrainLogger')
        logger.debug(f"Direction pos_weight: {[f'{x:.2f}' for x in pos_weight]}")
        # print(f"Positive samples per param: {pos_samples}")
        # print(f"Negative samples per param: {neg_samples}")
        # print(f"Computed pos_weight per param: {pos_weight}")
        # Compute weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            direction_pred,
            direction_labels,
            reduction='none',
            pos_weight=pos_weight
        )
        loss = loss * activity_mask  # Apply activity mask
        num_active = activity_mask.sum().clamp(min=1e-6)
        loss = loss.sum() / num_active
        
        # Compute F1 scores
        batch_size, num_teeth, num_params = direction_pred.shape
        f1_scores = compute_f1_score(
            torch.sigmoid(direction_pred).view(batch_size, -1),
            direction_labels.view(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)
        
        return loss, f1_scores

def compute_loss(trans_magnitude, rot_magnitude, active_pred, direction_pred,
                 cumulative_transforms, activity_labels, direction_labels, device, logger, args, is_freeze='none'):
    """Compute all loss components and total loss for training."""
    # Move inputs to device
    cumulative_transforms = cumulative_transforms.to(device)
    activity_labels = activity_labels.to(device)
    direction_labels = direction_labels.to(device)
    trans_magnitude, rot_magnitude, active_pred, direction_pred = [
        x.to(device) for x in [trans_magnitude, rot_magnitude, active_pred, direction_pred]
    ]
    
    # Split activity_labels and cumulative_transforms into trans and rot
    activity_mask_trans = activity_labels[:, :, :3]  # [batch_size, 14, 3]
    activity_mask_rot = activity_labels[:, :, 3:]    # [batch_size, 14, 3]
    cumulative_transforms_trans = cumulative_transforms[:, :, :3]  # [batch_size, 14, 3]
    cumulative_transforms_rot = cumulative_transforms[:, :, 3:]    # [batch_size, 14, 3]
    
    # Initialize loss functions
    translation_loss_fn = CumulativeTranslationLoss().to(device)
    rotation_loss_fn = CumulativeRotationLoss().to(device)
    active_loss_fn = CumulativeActiveLoss().to(device)
    direction_loss_fn = CumulativeDirectionLoss().to(device)
    
    # Log prediction statistics
    logger.debug(f"Trans mag min: {trans_magnitude.min():.4f}, max: {trans_magnitude.max():.4f}, "
                 f"has_nan: {torch.isnan(trans_magnitude).any()}")
    logger.debug(f"Rot mag min: {rot_magnitude.min():.4f}, max: {rot_magnitude.max():.4f}, "
                 f"has_nan: {torch.isnan(rot_magnitude).any()}")
    logger.debug(f"Active pred min: {active_pred.min():.4f}, max: {active_pred.max():.4f}, "
                 f"has_nan: {torch.isnan(active_pred).any()}")
    logger.debug(f"Direction pred min: {direction_pred.min():.4f}, max: {direction_pred.max():.4f}, "
                 f"has_nan: {torch.isnan(direction_pred).any()}")
    logger.debug(f"Activity mask sparsity: {activity_labels.sum() / activity_labels.numel():.4f}")
    
    # Compute individual losses
    loss_trans = translation_loss_fn(trans_magnitude, cumulative_transforms_trans, activity_mask_trans)
    loss_rot = rotation_loss_fn(rot_magnitude, cumulative_transforms_rot, activity_mask_rot)
    loss_active,active_f1_scores = active_loss_fn(active_pred, activity_labels)
    loss_direction, direction_f1_scores = direction_loss_fn(direction_pred, direction_labels, activity_labels)
    
    # Adjust weights based on freeze argument
    w_trans = 0.0 if is_freeze == 'classification' else args.w_trans
    w_rot = 0.0 if is_freeze == 'classification' else args.w_rot
    w_active = 0.0 if is_freeze == 'regression' else args.w_active
    w_dir = 0.0 if is_freeze == 'regression' else args.w_dir
    
    # Compute total loss
    total_loss = (
        w_trans * loss_trans +
        w_rot * loss_rot +
        w_active * loss_active +
        w_dir * loss_direction
    )
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Total loss is NaN or Inf: {total_loss.item()}")
    
    return total_loss, {
        'loss_trans': loss_trans,
        'loss_rot': loss_rot,
        'loss_active': loss_active,
        'loss_direction': loss_direction,
        'direction_f1_scores': direction_f1_scores,
        'active_f1_scores': active_f1_scores
    }