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

class CumulativeLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_cumulative, cumulative_transforms):
        return self.weight * self.mse_loss(pred_cumulative, cumulative_transforms)

class CumulativeZeroLoss(nn.Module):
    def __init__(self, threshold=0.1, weight=1.0):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
    
    def forward(self, pred_cumulative, cumulative_transforms, activity_mask):
        zero_mask = (cumulative_transforms == 0).float() * activity_mask
        non_zero_pred = torch.abs(pred_cumulative) * zero_mask
        loss = torch.relu(non_zero_pred - self.threshold) ** 2
        num_active = zero_mask.sum().clamp(min=1e-6)
        return self.weight * loss.sum() / num_active

class CumulativeSparsityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred_cumulative, param_activity_labels):
        return self.weight * torch.mean((pred_cumulative * (1 - param_activity_labels))**2)

class CumulativeActivityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, activity_logits, activity_labels):
        # Compute BCE per tooth
        batch_size, num_teeth = activity_logits.shape
        loss_per_tooth = F.binary_cross_entropy_with_logits(
            activity_logits, activity_labels, reduction='none'
        ).mean(dim=0)  # Mean over batch for each tooth
        loss = loss_per_tooth.mean()  # Mean over teeth for total loss
        # Compute F1-score per tooth
        f1_scores = compute_f1_score(torch.sigmoid(activity_logits), activity_labels)
        return self.weight * loss, f1_scores

class CumulativeParamActivityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, param_activity_logits, param_activity_labels, activity_labels):
        batch_size, num_teeth, num_params = param_activity_logits.shape
        active_mask = activity_labels.unsqueeze(-1).float()  # [batch_size, num_teeth, 1]
        # Compute BCE per parameter per tooth
        loss = F.binary_cross_entropy_with_logits(
            param_activity_logits, param_activity_labels, reduction='none'
        )
        loss = loss * active_mask  # Apply active mask
        loss_per_param = loss.mean(dim=0)  # Mean over batch for each tooth and parameter
        loss = loss_per_param.mean()  # Mean over teeth and parameters
        # Compute F1-score per parameter
        f1_scores = compute_f1_score(
            torch.sigmoid(param_activity_logits.view(batch_size, -1)),
            param_activity_labels.view(batch_size, -1)
        ).view(num_teeth, num_params).mean(dim=0)  # Reshape and mean over teeth per parameter
        return self.weight * loss, f1_scores