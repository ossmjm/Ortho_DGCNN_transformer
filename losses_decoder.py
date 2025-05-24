import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_f1_score
import logging

def binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """Custom binary focal loss for sigmoid outputs."""
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.clamp(torch.exp(-ce_loss), min=1e-7, max=1.0)
    focal_term = alpha * (1 - pt) ** gamma
    loss = focal_term * ce_loss
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss

class HybridTransformLoss(nn.Module):
    def __init__(self, weight=1.0, small_error_threshold=1.0, small_error_scale=5.0, max_error=40.0):
        super().__init__()
        self.weight = weight
        self.small_error_threshold = small_error_threshold
        self.small_error_scale = small_error_scale
        self.max_error = max_error
    
    def forward(self, pred, target, activity_mask=None, stage_weights=None):
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise ValueError("Pred tensor contains NaN or Inf")
        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError("Target tensor contains NaN or Inf")
        
        error = pred - target
        error = torch.clamp(error, -self.max_error, self.max_error)
        
        abs_error = torch.abs(error)
        large_error_mask = (abs_error > self.max_error).float()
        log_cosh = large_error_mask * (abs_error - torch.log(torch.tensor(2.0, device=error.device))) + \
                   (1 - large_error_mask) * torch.log(torch.cosh(error + 1e-12))
        
        mse = error ** 2
        small_error_mask = (abs_error < self.small_error_threshold).float()
        scaled_mse = mse * (self.small_error_scale * small_error_mask + (1 - small_error_mask))
        
        loss = log_cosh + scaled_mse
        if activity_mask is not None:
            if torch.isnan(activity_mask).any() or torch.isinf(activity_mask).any():
                raise ValueError("Activity mask contains NaN or Inf")
            loss = loss * activity_mask
        
        if stage_weights is not None:
            if torch.isnan(stage_weights).any() or torch.isinf(stage_weights).any():
                raise ValueError("Stage weights contain NaN or Inf")
            stage_weights_expanded = stage_weights.unsqueeze(-1).unsqueeze(-1)
            loss = loss * stage_weights_expanded
        
        num_active = (activity_mask * stage_weights_expanded).sum() if activity_mask is not None and stage_weights is not None else loss.numel()
        if num_active == 0:
            num_active = 1.0
        return self.weight * loss.sum() / num_active

class ToothActivityLoss(nn.Module):
    def __init__(self, weight=1.0, use_focal=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.use_focal = use_focal
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels, true_num_stages):
        logger = logging.getLogger('TrainLogger')
        batch_size, max_stages, num_teeth = logits.shape
        loss = 0.0
        num_valid_teeth = 0
        logits = torch.clamp(logits, -100, 100)
        
        # Create indices for valid stages
        valid_indices = []
        for b in range(batch_size):
            for s in range(true_num_stages[b].item()):
                valid_indices.append((b, s))
        
        if not valid_indices:
            logger.warning("No valid stages found for ToothActivityLoss")
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
        
        # Gather valid logits and labels
        valid_logits = []
        valid_labels = []
        for tooth_idx in range(num_teeth):
            tooth_logits = logits[:, :, tooth_idx]  # [batch_size, max_stages]
            tooth_labels = labels[:, :, tooth_idx]  # [batch_size, max_stages]
            
            # Extract valid elements
            tooth_valid_logits = torch.stack([tooth_logits[b, s] for b, s in valid_indices])
            tooth_valid_labels = torch.stack([tooth_labels[b, s] for b, s in valid_indices])
            
            if self.use_focal:
                tooth_loss = binary_focal_loss(
                    tooth_valid_logits,
                    tooth_valid_labels,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    reduction='sum'
                )
                tooth_loss = tooth_loss / (len(valid_indices) + 1e-6)
            else:
                tooth_loss = self.bce_loss(tooth_valid_logits, tooth_valid_labels)
                tooth_loss = tooth_loss.sum() / (len(valid_indices) + 1e-6)
            
            loss += tooth_loss
            num_valid_teeth += 1
            
            valid_logits.append(tooth_valid_logits)
            valid_labels.append(tooth_valid_labels)
        
        loss = self.weight * loss / (num_valid_teeth + 1e-6)
        
        # Compute F1 score
        valid_preds = torch.sigmoid(torch.cat(valid_logits))
        valid_labels = torch.cat(valid_labels)
        if valid_preds.numel() > 0:
            f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5)
        else:
            logger.warning("No valid predictions for F1 score in ToothActivityLoss")
            f1 = torch.tensor(0.0, device=logits.device)
        
        logger.debug(f"ToothActivityLoss: num_valid_teeth: {num_valid_teeth}, valid_stages: {len(valid_indices)}")
        
        return loss, f1

class ParamActivityLoss(nn.Module):
    def __init__(self, weight=1.0, use_focal=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.use_focal = use_focal
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels, tooth_activity_labels, true_num_stages):
        logger = logging.getLogger('TrainLogger')
        batch_size, max_stages, num_teeth, num_params = logits.shape
        loss = 0.0
        num_active_teeth = 0
        logits = torch.clamp(logits, -100, 100)
        
        # Create indices for valid stages
        valid_indices = []
        for b in range(batch_size):
            for s in range(true_num_stages[b].item()):
                valid_indices.append((b, s))
        
        if not valid_indices:
            logger.warning("No valid stages found for ParamActivityLoss")
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
        
        # Compute active mask (binary check)
        active_mask = (tooth_activity_labels == 1.0).float().unsqueeze(-1)
        logger.debug(f"ParamActivityLoss active_mask shape: {active_mask.shape}, unique_values: {torch.unique(tooth_activity_labels).tolist()}")
        
        valid_logits = []
        valid_labels = []
        for tooth_idx in range(num_teeth):
            tooth_active = active_mask[:, :, tooth_idx, 0]
            tooth_active_sum = tooth_active.sum().item()
            logger.debug(f"Tooth {tooth_idx} active stages: {tooth_active_sum}")
            
            if tooth_active_sum > 0:
                tooth_logits = logits[:, :, tooth_idx, :]  # [batch_size, max_stages, num_params]
                tooth_labels = labels[:, :, tooth_idx, :]  # [batch_size, max_stages, num_params]
                tooth_loss = 0.0
                for param_idx in range(num_params):
                    param_logits = tooth_logits[:, :, param_idx]
                    param_labels = tooth_labels[:, :, param_idx]
                    
                    # Extract valid elements
                    param_valid_logits = torch.stack([param_logits[b, s] for b, s in valid_indices if tooth_active[b, s] > 0])
                    param_valid_labels = torch.stack([param_labels[b, s] for b, s in valid_indices if tooth_active[b, s] > 0])
                    
                    if param_valid_logits.numel() == 0:
                        continue
                    
                    if self.use_focal:
                        param_loss = binary_focal_loss(
                            param_valid_logits,
                            param_valid_labels,
                            alpha=self.alpha,
                            gamma=self.gamma,
                            reduction='sum'
                        )
                        param_loss = param_loss / (param_valid_logits.numel() + 1e-6)
                    else:
                        param_loss = self.bce_loss(param_valid_logits, param_valid_labels)
                        param_loss = param_loss.sum() / (param_valid_logits.numel() + 1e-6)
                    
                    tooth_loss += param_loss
                    valid_logits.append(param_valid_logits)
                    valid_labels.append(param_valid_labels)
                
                tooth_loss = tooth_loss / num_params
                loss += tooth_loss
                num_active_teeth += 1
        
        loss = self.weight * loss / (num_active_teeth + 1e-6) if num_active_teeth > 0 else torch.tensor(0.0, device=logits.device)
        
        # Compute F1 score
        if valid_logits:
            valid_preds = torch.sigmoid(torch.cat(valid_logits))
            valid_labels = torch.cat(valid_labels)
            if valid_preds.numel() > 0:
                f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5)
            else:
                logger.warning("No valid predictions for F1 score in ParamActivityLoss")
                f1 = torch.tensor(0.0, device=logits.device)
        else:
            logger.warning("No valid predictions for F1 score in ParamActivityLoss")
            f1 = torch.tensor(0.0, device=logits.device)
        
        # Log active mask details
        active_mask_sum = active_mask.sum().item()
        per_batch_sums = [active_mask[b].sum().item() for b in range(batch_size)]
        logger.debug(f"ParamActivityLoss: num_active_teeth: {num_active_teeth}, active_mask_sum: {active_mask_sum}, per_batch_sums: {per_batch_sums}")
        
        return loss, f1

class PaddedLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred_transforms, num_stages, max_stages):
        batch_size = pred_transforms.size(0)
        loss = 0.0
        for b in range(batch_size):
            true_stages = num_stages[b].item()
            if true_stages < max_stages:
                loss += torch.mean(pred_transforms[b, true_stages:, :, :]**2)
        return self.weight * loss / batch_size if batch_size > 0 else 0.0

class ConsistencyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_transforms, cumulative_transforms, num_stages, max_stages, device):
        batch_size = pred_transforms.size(0)
        loss = 0.0
        for i in range(batch_size):
            n_stages = min(num_stages[i].item(), max_stages)
            pred_sum = pred_transforms[i, :n_stages].sum(dim=0)
            loss += F.l1_loss(pred_sum, cumulative_transforms[i])
        return self.weight * loss / batch_size