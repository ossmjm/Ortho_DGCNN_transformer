import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def compute_f1_score(preds, labels):
    """Compute F1-score for binary predictions per unit (tooth or parameter)."""
    preds = (preds > 0.5).float()
    tp = (preds * labels).sum(dim=0)  # Sum over batch
    pred_sum = preds.sum(dim=0)
    label_sum = labels.sum(dim=0)
    precision = tp / (pred_sum + 1e-6)
    recall = tp / (label_sum + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1

def compute_activity_mask(cumulative_transforms):
    """Compute activity mask for non-zero values in cumulative_transforms."""
    return (cumulative_transforms != 0).float()  # [batch_size, num_teeth, num_params]

class CumulativeLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, pred_cumulative, cumulative_transforms, activity_mask):
        """Compute MSE loss for transformations where activity_mask is non-zero."""
        loss = self.mse_loss(pred_cumulative, cumulative_transforms)
        loss = loss * activity_mask  # Mask non-zero transformations
        num_active = activity_mask.sum().clamp(min=1e-6)
        return self.weight * loss.sum() / num_active

class CumulativeActivityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, activity_logits, activity_labels):
        """Compute weighted BCE loss per tooth over all data, averaged across teeth."""
        batch_size, num_teeth = activity_labels.shape
        # Compute pos_weight for class imbalance
        pos_samples = activity_labels.sum(dim=0)  # [num_teeth]
        neg_samples = batch_size - pos_samples  # [num_teeth]
        pos_weight = neg_samples / (pos_samples + 1e-6)  # [num_teeth]
        pos_weight = pos_weight.clamp(max=100.0)  # Prevent extreme weights
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"Activity pos_weight: {[f'{x:.2f}' for x in pos_weight]}")
        
        # Compute weighted BCE per tooth
        loss_per_tooth = F.binary_cross_entropy_with_logits(
            activity_logits,
            activity_labels,
            reduction='none',
            pos_weight=pos_weight
        ).mean(dim=0)  # Mean over batch
        loss = loss_per_tooth.mean()  # Mean over teeth
        f1_scores = compute_f1_score(torch.sigmoid(activity_logits), activity_labels)
        return self.weight * loss, f1_scores

class CumulativeParamActivityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, param_activity_logits, param_activity_labels):
        """Compute weighted BCE loss per parameter per tooth over all data."""
        batch_size, num_teeth, num_params = param_activity_logits.shape
        # Compute pos_weight for class imbalance
        pos_samples = param_activity_labels.sum(dim=0)  # [num_teeth, num_params]
        neg_samples = batch_size - pos_samples  # [num_teeth, num_params]
        pos_weight = neg_samples / (pos_samples + 1e-6)  # [num_teeth, num_params]
        pos_weight = pos_weight.clamp(max=100.0)  # Prevent extreme weights
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"Param activity pos_weight: {[f'{x:.2f}' for x in pos_weight.mean(dim=0)]}")
        
        # Compute weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            param_activity_logits,
            param_activity_labels,
            reduction='none',
            pos_weight=pos_weight
        )
        loss = loss.mean(dim=0).mean()  # Mean over batch and teeth/params
        f1_scores = compute_f1_score(
            torch.sigmoid(param_activity_logits).view(batch_size, -1),
            param_activity_labels.view(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)
        return self.weight * loss, f1_scores

class CumulativeDirectionLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, directions_logits, directions_labels, activity_mask):
        """Compute weighted BCE loss for directions where directions_labels are non-zero."""
        batch_size, num_teeth, num_params = directions_logits.shape
        # Mask for non-zero direction labels
        direction_mask = (directions_labels != 0).float()  # [batch_size, num_teeth, num_params]
        combined_mask = direction_mask * activity_mask  # [batch_size, num_teeth, num_params]
        # Compute pos_weight for class imbalance
        pos_samples = (directions_labels * combined_mask).sum(dim=0)  # [num_teeth, num_params]
        neg_samples = (combined_mask - (directions_labels * combined_mask)).sum(dim=0)  # [num_teeth, num_params]
        pos_weight = neg_samples / (pos_samples + 1e-6)  # [num_teeth, num_params]
        pos_weight = pos_weight.clamp(max=100.0)  # Prevent extreme weights
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"Direction pos_weight: {[f'{x:.2f}' for x in pos_weight.mean(dim=0)]}")
        
        # Compute weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            directions_logits,
            directions_labels,
            reduction='none',
            pos_weight=pos_weight
        )
        loss = loss * combined_mask  # Apply combined mask
        num_active = combined_mask.sum().clamp(min=1e-6)
        loss = loss.sum() / num_active  # Normalize by number of active elements
        f1_scores = compute_f1_score(
            torch.sigmoid(directions_logits).view(batch_size, -1),
            directions_labels.view(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)
        return self.weight * loss, f1_scores

def compute_loss(pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits, directions_logits,
                 cumulative_transforms, cumulative_activity, cumulative_param_activity, directions_labels,
                 device, logger, args):
    """Compute all loss components and total loss for training."""
    activity_mask = compute_activity_mask(cumulative_transforms).to(device)
    
    cumulative_loss_fn = CumulativeLoss(weight=args.w_cumulative).to(device)
    cumulative_activity_loss_fn = CumulativeActivityLoss(weight=args.w_cumulative_activity).to(device)
    cumulative_param_activity_loss_fn = CumulativeParamActivityLoss(weight=args.w_cumulative_param_activity).to(device)
    cumulative_direction_loss_fn = CumulativeDirectionLoss(weight=args.w_cumulative_direction).to(device)
    
    logger.debug(f"Pred cumulative min: {pred_cumulative.min():.4f}, max: {pred_cumulative.max():.4f}, "
                 f"has_nan: {torch.isnan(pred_cumulative).any()}")
    
    loss_cumulative = cumulative_loss_fn(pred_cumulative, cumulative_transforms, activity_mask)
    loss_cumulative_activity, activity_f1_scores = cumulative_activity_loss_fn(cumulative_activity_logits, cumulative_activity)
    loss_cumulative_param_activity, param_activity_f1_scores = cumulative_param_activity_loss_fn(
        cumulative_param_activity_logits, cumulative_param_activity)
    loss_cumulative_direction, direction_f1_scores = cumulative_direction_loss_fn(
        directions_logits, directions_labels, activity_mask)
    
    logger.debug(f"Cumulative param activity logits min: {cumulative_param_activity_logits.min():.4f}, "
                 f"max: {cumulative_param_activity_logits.max():.4f}, "
                 f"has_nan: {torch.isnan(cumulative_param_activity_logits).any()}")
    logger.debug(f"Cumulative param activity labels min: {cumulative_param_activity.min():.4f}, "
                 f"max: {cumulative_param_activity.max():.4f}, "
                 f"has_nan: {torch.isnan(cumulative_param_activity).any()}")
    logger.debug(f"Directions logits min: {directions_logits.min():.4f}, "
                 f"max: {directions_logits.max():.4f}, "
                 f"has_nan: {torch.isnan(directions_logits).any()}")
    
    losses = {
        'loss_cumulative': loss_cumulative,
        'loss_cumulative_activity': loss_cumulative_activity,
        'loss_cumulative_param_activity': loss_cumulative_param_activity,
        'loss_cumulative_direction': loss_cumulative_direction,
        'activity_f1_scores': activity_f1_scores,
        'param_activity_f1_scores': param_activity_f1_scores,
        'direction_f1_scores': direction_f1_scores
    }
    for name, loss in losses.items():
        if isinstance(loss, torch.Tensor) and (torch.isnan(loss).any() or torch.isinf(loss).any()):
            logger.error(f"{name} is NaN or Inf: {loss.item()}")
    
    total_loss = (
        args.w_cumulative * loss_cumulative +
        args.w_cumulative_activity * loss_cumulative_activity +
        args.w_cumulative_param_activity * loss_cumulative_param_activity +
        args.w_cumulative_direction * loss_cumulative_direction
    )
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Total loss is NaN or Inf: {total_loss.item()}")
    
    return total_loss, losses