import torch
import torch.nn as nn
import torch.nn.functional as F

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

class StagewiseMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target, activity_mask=None, stage_weights=None):
        loss = self.mse_loss(pred, target)
        if activity_mask is not None:
            loss = loss * activity_mask
        if stage_weights is not None:
            stage_weights_expanded = stage_weights.unsqueeze(-1).unsqueeze(-1)
            loss = loss * stage_weights_expanded
        num_active = (activity_mask * stage_weights_expanded).sum() if activity_mask is not None and stage_weights is not None else loss.numel()
        # num_active = num_active.clamp(min=1e-6)
        return self.weight * loss.sum() / num_active

class ToothActivityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, labels, true_num_stages):
        batch_size, max_stages, num_teeth = logits.shape
        loss = 0.0
        num_valid_teeth = 0
        
        # Create stage mask based on true_num_stages
        stage_mask = torch.ones(batch_size, max_stages, device=logits.device)
        for b in range(batch_size):
            stage_mask[b, true_num_stages[b]:] = 0.0
        
        # Compute BCE loss per tooth
        for tooth_idx in range(num_teeth):
            tooth_logits = logits[:, :, tooth_idx]  # [batch_size, max_stages]
            tooth_labels = labels[:, :, tooth_idx]  # [batch_size, max_stages]
            tooth_loss = self.bce_loss(tooth_logits, tooth_labels)  # [batch_size, max_stages]
            tooth_loss = tooth_loss * stage_mask  # Mask out padded stages
            tooth_loss = tooth_loss.sum() / (stage_mask.sum() + 1e-6)  # Average over valid stages
            loss += tooth_loss
            num_valid_teeth += 1
        
        loss = self.weight * loss / (num_valid_teeth + 1e-6)
        
        # Compute F1-score per tooth, considering only valid stages
        preds = torch.sigmoid(logits) * stage_mask.unsqueeze(-1)  # [batch_size, max_stages, num_teeth]
        labels = labels * stage_mask.unsqueeze(-1)
        tp = (preds > 0.5).float() * labels
        fp = (preds > 0.5).float() * (1 - labels)
        fn = (preds <= 0.5).float() * labels
        tp_sum = tp.sum(dim=(0, 1))  # Sum over batch and stages
        fp_sum = fp.sum(dim=(0, 1))
        fn_sum = fn.sum(dim=(0, 1))
        precision = tp_sum / (tp_sum + fp_sum + 1e-6)
        recall = tp_sum / (tp_sum + fn_sum + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return loss, f1

class ParamActivityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, labels, tooth_activity_mask, true_num_stages):
        batch_size, max_stages, num_teeth, num_params = logits.shape
        loss = 0.0
        num_active_teeth = 0
        
        # Create stage mask based on true_num_stages
        stage_mask = torch.ones(batch_size, max_stages, device=logits.device)
        for b in range(batch_size):
            stage_mask[b, true_num_stages[b]:] = 0.0
        
        # Apply activity mask (active teeth only)
        active_mask = (torch.sigmoid(tooth_activity_mask) > 0.5).float().unsqueeze(-1)  # [batch_size, max_stages, num_teeth, 1]
        
        # Compute BCE loss per tooth
        for tooth_idx in range(num_teeth):
            tooth_active = active_mask[:, :, tooth_idx, 0]  # [batch_size, max_stages]
            if tooth_active.sum() > 0:  # Only compute loss for active teeth
                tooth_logits = logits[:, :, tooth_idx, :]  # [batch_size, max_stages, num_params]
                tooth_labels = labels[:, :, tooth_idx, :]  # [batch_size, max_stages, num_params]
                tooth_loss = self.bce_loss(tooth_logits, tooth_labels)  # [batch_size, max_stages, num_params]
                tooth_loss = tooth_loss * stage_mask.unsqueeze(-1) * tooth_active.unsqueeze(-1)  # Mask padded stages and inactive teeth
                tooth_loss = tooth_loss.sum() / (stage_mask.sum() + 1e-6)  # Average over valid stages
                loss += tooth_loss
                num_active_teeth += 1
        
        loss = self.weight * loss / (num_active_teeth + 1e-6) if num_active_teeth > 0 else torch.tensor(0.0, device=logits.device)
        
        # Compute F1-score per parameter, considering only valid stages and active teeth
        stage_mask_expanded = stage_mask.unsqueeze(-1).unsqueeze(-1)  # [batch_size, max_stages, 1, 1]
        preds = torch.sigmoid(logits) * stage_mask_expanded * active_mask  # [batch_size, max_stages, num_teeth, num_params]
        labels = labels * stage_mask_expanded * active_mask
        tp = (preds > 0.5).float() * labels
        fp = (preds > 0.5).float() * (1 - labels)
        fn = (preds <= 0.5).float() * labels
        tp_sum = tp.sum(dim=(0, 1))  # Sum over batch and stages
        fp_sum = fp.sum(dim=(0, 1))
        fn_sum = fn.sum(dim=(0, 1))
        precision = tp_sum / (tp_sum + fp_sum + 1e-6)
        recall = tp_sum / (tp_sum + fn_sum + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
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