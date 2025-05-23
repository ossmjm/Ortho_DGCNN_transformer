import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_f1_score
from torchvision.ops import sigmoid_focal_loss
import logging

class HybridTransformLoss(nn.Module):
    def __init__(self, weight=1.0, small_error_threshold=1.0, small_error_scale=5.0, max_error=40.0):
        super().__init__()
        self.weight = weight
        self.small_error_threshold = small_error_threshold
        self.small_error_scale = small_error_scale
        self.max_error = max_error  # Maximum error value to prevent overflow in cosh
    
    def forward(self, pred, target, activity_mask=None, stage_weights=None):
        # Input validation
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise ValueError("Pred tensor contains NaN or Inf")
        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError("Target tensor contains NaN or Inf")
        
        # Compute error and clamp to prevent overflow
        error = pred - target
        error = torch.clamp(error, -self.max_error, self.max_error)
        
        # Log-Cosh loss with stabilized computation
        abs_error = torch.abs(error)
        # For large errors, approximate log(cosh(x)) ≈ |x| - log(2)
        large_error_mask = (abs_error > self.max_error).float()
        log_cosh = large_error_mask * (abs_error - torch.log(torch.tensor(2.0, device=error.device))) + \
                   (1 - large_error_mask) * torch.log(torch.cosh(error + 1e-12))
        
        # Scaled MSE for small errors
        mse = error ** 2
        small_error_mask = (abs_error < self.small_error_threshold).float()
        scaled_mse = mse * (self.small_error_scale * small_error_mask + (1 - small_error_mask))
        
        # Combine losses
        loss = log_cosh + scaled_mse
        # Apply activity mask
        if activity_mask is not None:
            if torch.isnan(activity_mask).any() or torch.isinf(activity_mask).any():
                raise ValueError("Activity mask contains NaN or Inf")
            loss = loss * activity_mask
        
        # Apply stage weights
        if stage_weights is not None:
            if torch.isnan(stage_weights).any() or torch.isinf(stage_weights).any():
                raise ValueError("Stage weights contain NaN or Inf")
            stage_weights_expanded = stage_weights.unsqueeze(-1).unsqueeze(-1)
            loss = loss * stage_weights_expanded
        
        num_active = (activity_mask * stage_weights_expanded).sum() if activity_mask is not None and stage_weights is not None else loss.numel()
        if num_active == 0:
            num_active = 1.0  # Prevent division by zero
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
        batch_size, max_stages, num_teeth = logits.shape
        loss = 0.0
        num_valid_teeth = 0
        # Clamp logits for numerical stability
        logits = torch.clamp(logits, -100, 100)
        # Create stage mask
        stage_mask = torch.ones(batch_size, max_stages, device=logits.device)
        for b in range(batch_size):
            stage_mask[b, true_num_stages[b]:] = 0.0
        
        for tooth_idx in range(num_teeth):
            tooth_logits = logits[:, :, tooth_idx]  # [batch_size, max_stages]
            tooth_labels = labels[:, :, tooth_idx]  # [batch_size, max_stages]
            
            if self.use_focal:
                # Use torcheval's focal_loss with sum reduction
                tooth_loss = sigmoid_focal_loss(
                    tooth_logits.flatten(),
                    tooth_labels.flatten(),
                    alpha=self.alpha,
                    gamma=self.gamma,
                    reduction='sum'
                )
                # Normalize by number of valid stages
                tooth_loss = tooth_loss / (stage_mask.sum() + 1e-6)
            else:
                tooth_loss = self.bce_loss(tooth_logits, tooth_labels)
                tooth_loss = tooth_loss * stage_mask
                tooth_loss = tooth_loss.sum() / (stage_mask.sum() + 1e-6)
            
            loss += tooth_loss
            num_valid_teeth += 1
        
        loss = self.weight * loss / (num_valid_teeth + 1e-6)
        
        # Compute F1 score
        # Expand stage_mask to include num_teeth dimension
        stage_mask_expanded = stage_mask.unsqueeze(-1).expand(-1, -1, num_teeth)
        preds = torch.sigmoid(logits) * stage_mask_expanded
        labels = labels * stage_mask_expanded
        valid_mask = stage_mask_expanded.flatten()
        valid_preds = preds.flatten()[valid_mask.bool()]
        valid_labels = labels.flatten()[valid_mask.bool()]
        
        if valid_preds.numel() > 0:
            f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5)
        else:
            f1 = torch.tensor(0.0, device=logits.device)
        
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
        
        # Clamp logits for numerical stability
        logits = torch.clamp(logits, -100, 100)
        
        # Create stage mask
        stage_mask = torch.ones(batch_size, max_stages, device=logits.device)
        for b in range(batch_size):
            stage_mask[b, true_num_stages[b]:] = 0.0
        
        # Compute active mask from true tooth_activity_labels
        active_mask = (tooth_activity_labels > 0.5).float().unsqueeze(-1)
        
        for tooth_idx in range(num_teeth):
            tooth_active = active_mask[:, :, tooth_idx, 0]  # [batch_size, max_stages]
            if tooth_active.sum() > 0:
                tooth_logits = logits[:, :, tooth_idx, :]  # [batch_size, max_stages, num_params]
                tooth_labels = labels[:, :, tooth_idx, :]  # [batch_size, max_stages, num_params]
                tooth_loss = 0.0
                for param_idx in range(num_params):
                    param_logits = tooth_logits[:, :, param_idx]  # [batch_size, max_stages]
                    param_labels = tooth_labels[:, :, param_idx]  # [batch_size, max_stages]
                    if self.use_focal:
                        # Use torcheval's focal_loss with sum reduction
                        param_loss = sigmoid_focal_loss(
                            param_logits.flatten(),
                            param_labels.flatten(),
                            alpha=self.alpha,
                            gamma=self.gamma,
                            reduction='sum'
                        )
                        # Normalize by number of valid stages
                        param_loss = param_loss / (stage_mask.sum() + 1e-6)
                    else:
                        param_loss = self.bce_loss(param_logits, param_labels)
                        param_loss = param_loss * stage_mask * tooth_active
                        param_loss = param_loss.sum() / ((stage_mask * tooth_active).sum() + 1e-6)
                    tooth_loss += param_loss
                tooth_loss = tooth_loss / num_params
                loss += tooth_loss
                num_active_teeth += 1
        
        loss = self.weight * loss / (num_active_teeth + 1e-6) if num_active_teeth > 0 else torch.tensor(0.0, device=logits.device)
        
        # Log number of active teeth
        logger.debug(f"ParamActivityLoss: num_active_teeth: {num_active_teeth}, active_mask_sum: {active_mask.sum().item()}")
        
        # Compute F1 score
        stage_mask_expanded = stage_mask.unsqueeze(-1).expand(-1, -1, num_teeth).unsqueeze(-1).expand(-1, -1, -1, num_params)
        active_mask_expanded = active_mask
        preds = torch.sigmoid(logits) * stage_mask_expanded * active_mask_expanded
        labels = labels * stage_mask_expanded * active_mask_expanded
        valid_mask = (stage_mask_expanded * active_mask_expanded).flatten()
        valid_preds = preds.flatten()[valid_mask.bool()]
        valid_labels = labels.flatten()[valid_mask.bool()]
        
        if valid_preds.numel() > 0:
            f1 = binary_f1_score(valid_preds, valid_labels, threshold=0.5)
        else:
            f1 = torch.tensor(0.0, device=logits.device)
        
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