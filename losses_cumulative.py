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

class CumulativeTranslationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, pred_translations, cumulative_translations, activity_mask):
        """Compute MSE loss with logarithmic weighting for translations to emphasize low errors."""
        loss = self.mse_loss(pred_translations, cumulative_translations)
        # Apply logarithmic weighting to emphasize small errors
        weighted_loss = torch.log1p(loss * activity_mask) * activity_mask
        num_active = activity_mask.sum().clamp(min=1e-6)
        return weighted_loss.sum() / num_active

class CumulativeRotationLoss(nn.Module):
    def __init__(self, max_rotation_scale=90.0):
        super().__init__()
        self.max_rotation_scale = max_rotation_scale  # Assumed max rotation in degrees
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, pred_rotations, cumulative_rotations, activity_mask):
        """Compute MSE loss with scaling for rotations to handle higher values."""
        loss = self.mse_loss(pred_rotations, cumulative_rotations)
        # Normalize by max_rotation_scale to handle larger ranges
        scaled_loss = loss / (self.max_rotation_scale + 1e-6) * activity_mask
        num_active = activity_mask.sum().clamp(min=1e-6)
        return scaled_loss.sum() / num_active

class CumulativeActivityLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
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
        return loss, f1_scores

class CumulativeParamActivityLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
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
            param_activity_labels.reshape(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)
        return loss, f1_scores

class CumulativeDirectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, directions_logits_trans, directions_logits_rot, direction_trans, direction_rot, activity_mask_trans, activity_mask_rot):
        """Compute weighted BCE loss for directions where cumulative_transforms are non-zero."""
        batch_size, num_teeth, num_params = directions_logits_trans.shape
        
        # Use activity mask to compute loss only where cumulative_transforms are non-zero
        combined_mask_trans = activity_mask_trans  # Use activity mask for translations
        combined_mask_rot = activity_mask_rot      # Use activity mask for rotations
        
        # Compute pos_weight for class imbalance for translation directions
        pos_samples_trans = (direction_trans * combined_mask_trans).sum(dim=0)  # [num_teeth, num_params]
        neg_samples_trans = (combined_mask_trans - (direction_trans * combined_mask_trans)).sum(dim=0)  # [num_teeth, num_params]
        pos_weight_trans = neg_samples_trans / (pos_samples_trans + 1e-6)  # [num_teeth, num_params]
        pos_weight_trans = pos_weight_trans.clamp(max=100.0)  # Prevent extreme weights
        
        # Compute pos_weight for class imbalance for rotation directions
        pos_samples_rot = (direction_rot * combined_mask_rot).sum(dim=0)  # [num_teeth, num_params]
        neg_samples_rot = (combined_mask_rot - (direction_rot * combined_mask_rot)).sum(dim=0)  # [num_teeth, num_params]
        pos_weight_rot = neg_samples_rot / (pos_samples_rot + 1e-6)  # [num_teeth, num_params]
        pos_weight_rot = pos_weight_rot.clamp(max=100.0)  # Prevent extreme weights
        
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"Direction trans pos_weight: {[f'{x:.2f}' for x in pos_weight_trans.mean(dim=0)]}")
        logger.debug(f"Direction rot pos_weight: {[f'{x:.2f}' for x in pos_weight_rot.mean(dim=0)]}")
        
        # Compute weighted BCE for translation directions
        loss_trans = F.binary_cross_entropy_with_logits(
            directions_logits_trans,
            direction_trans,
            reduction='none',
            pos_weight=pos_weight_trans
        )
        loss_trans = loss_trans * combined_mask_trans  # Apply activity mask
        num_active_trans = combined_mask_trans.sum().clamp(min=1e-6)
        loss_trans = loss_trans.sum() / num_active_trans  # Normalize by number of active elements
        
        # Compute weighted BCE for rotation directions
        loss_rot = F.binary_cross_entropy_with_logits(
            directions_logits_rot,
            direction_rot,
            reduction='none',
            pos_weight=pos_weight_rot
        )
        loss_rot = loss_rot * combined_mask_rot  # Apply activity mask
        num_active_rot = combined_mask_rot.sum().clamp(min=1e-6)
        loss_rot = loss_rot.sum() / num_active_rot  # Normalize by number of active elements
        
        # Compute F1 scores
        f1_scores_trans = compute_f1_score(
            torch.sigmoid(directions_logits_trans).view(batch_size, -1),
            direction_trans.reshape(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)
        f1_scores_rot = compute_f1_score(
            torch.sigmoid(directions_logits_rot).view(batch_size, -1),
            direction_rot.reshape(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)
        
        return loss_trans, loss_rot, f1_scores_trans, f1_scores_rot

def compute_loss(pred_cumulative_trans, pred_cumulative_rot, activity_logits,
                 cumulative_param_activity_logits_trans, cumulative_param_activity_logits_rot,
                 directions_logits_trans, directions_logits_rot,
                 cumulative_transforms, cumulative_activity, cumulative_param_activity, directions_labels, 
                 device, logger, args):
    """Compute all loss components and total loss for training."""
    activity_mask = compute_activity_mask(cumulative_transforms).to(device)
    
    # Split activity_mask and cumulative_transforms into trans and rot
    activity_mask_trans = activity_mask[:, :, :3]  # First 3 parameters (translations)
    activity_mask_rot = activity_mask[:, :, 3:]    # Last 3 parameters (rotations)
    cumulative_transforms_trans = cumulative_transforms[:, :, :3]
    cumulative_transforms_rot = cumulative_transforms[:, :, 3:]
    direction_trans = directions_labels[:, :, :3]
    direction_rot = directions_labels[:, :, 3:]
    
    cumulative_translation_loss_fn = CumulativeTranslationLoss().to(device)
    cumulative_rotation_loss_fn = CumulativeRotationLoss().to(device)
    cumulative_activity_loss_fn = CumulativeActivityLoss().to(device)
    cumulative_param_activity_loss_trans_fn = CumulativeParamActivityLoss().to(device)
    cumulative_param_activity_loss_rot_fn = CumulativeParamActivityLoss().to(device)
    cumulative_direction_loss_fn = CumulativeDirectionLoss().to(device)
    logger.debug(f"Pred cumulative trans min: {pred_cumulative_trans.min():.4f}, max: {pred_cumulative_trans.max():.4f}, "
                 f"has_nan: {torch.isnan(pred_cumulative_trans).any()}")
    logger.debug(f"Pred cumulative rot min: {pred_cumulative_rot.min():.4f}, max: {pred_cumulative_rot.max():.4f}, "
                 f"has_nan: {torch.isnan(pred_cumulative_rot).any()}")
    
    loss_cumulative_trans = cumulative_translation_loss_fn(pred_cumulative_trans, cumulative_transforms_trans, activity_mask_trans)
    loss_cumulative_rot = cumulative_rotation_loss_fn(pred_cumulative_rot, cumulative_transforms_rot, activity_mask_rot)

    loss_cumulative_activity, activity_f1_scores = cumulative_activity_loss_fn(
        activity_logits, cumulative_activity)

    loss_cumulative_param_activity_trans, param_activity_f1_scores_trans = cumulative_param_activity_loss_trans_fn(
        cumulative_param_activity_logits_trans, cumulative_param_activity[:, :, :3])
    loss_cumulative_param_activity_rot, param_activity_f1_scores_rot = cumulative_param_activity_loss_rot_fn(
        cumulative_param_activity_logits_rot, cumulative_param_activity[:, :, 3:])
    
    (loss_cumulative_direction_trans, loss_cumulative_direction_rot,
     direction_f1_scores_trans, direction_f1_scores_rot) = cumulative_direction_loss_fn(
        directions_logits_trans, directions_logits_rot, direction_trans, direction_rot,
        activity_mask_trans, activity_mask_rot)
    
    losses = {
        'loss_cumulative_trans': loss_cumulative_trans,
        'loss_cumulative_rot': loss_cumulative_rot,
        'loss_cumulative_activity': loss_cumulative_activity,
        'loss_cumulative_param_activity_trans': loss_cumulative_param_activity_trans,
        'loss_cumulative_param_activity_rot': loss_cumulative_param_activity_rot,
        'loss_cumulative_direction_trans': loss_cumulative_direction_trans,
        'loss_cumulative_direction_rot': loss_cumulative_direction_rot,
        'activity_f1_scores': activity_f1_scores,
        'param_activity_f1_scores_trans': param_activity_f1_scores_trans,
        'param_activity_f1_scores_rot': param_activity_f1_scores_rot,
        'direction_f1_scores_trans': direction_f1_scores_trans,
        'direction_f1_scores_rot': direction_f1_scores_rot
    }
    for name, loss in losses.items():
        if isinstance(loss, torch.Tensor) and (torch.isnan(loss).any() or torch.isinf(loss).any()):
            logger.error(f"{name} is NaN or Inf: {loss.item()}")
    
    total_loss = (
        args.w_cumulative_trans * loss_cumulative_trans +
        args.w_cumulative_rot * loss_cumulative_rot +
        args.w_cumulative_activity * loss_cumulative_activity +
        args.w_cumulative_param_activity * (loss_cumulative_param_activity_trans + loss_cumulative_param_activity_rot) +
        args.w_cumulative_direction * (loss_cumulative_direction_trans + loss_cumulative_direction_rot)
    )
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Total loss is NaN or Inf: {total_loss.item()}")
    
    return total_loss, losses