import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=0.5, alpha=10.0, gamma=0.05, epsilon=1e-6, max_value=100.0):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_value = max_value
    
    def forward(self, pred, target, activity_mask=None, stage_weights=None):
        pred = torch.clamp(pred, -self.max_value, self.max_value)
        target = torch.clamp(target, -self.max_value, self.max_value)
        diff = torch.abs(pred - target)
        smooth_l1 = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        weights = torch.exp(-self.alpha * torch.clamp(diff, min=self.epsilon)) + self.gamma
        if activity_mask is not None:
            smooth_l1 = smooth_l1 * activity_mask
            weights = weights * activity_mask
        if stage_weights is not None:
            stage_weights_expanded = stage_weights.unsqueeze(-1).unsqueeze(-1)
            smooth_l1 = smooth_l1 * stage_weights_expanded
            weights = weights * stage_weights_expanded
        num_active = (activity_mask * stage_weights_expanded).sum() if activity_mask is not None and stage_weights is not None else smooth_l1.numel()
        num_active = num_active.clamp(min=self.epsilon)
        return (weights * smooth_l1).sum() / num_active

class SparsityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, param_activity_labels=None):
        if param_activity_labels is not None:
            return self.weight * torch.mean((pred * (1 - param_activity_labels))**2)
        return self.weight * torch.abs(pred).mean()

class SmoothnessLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred):
        diff = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        return self.weight * torch.abs(diff).mean()

class ZeroPredictionLoss(nn.Module):
    def __init__(self, threshold=0.1, weight=1.0):
        super().__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, pred, target, activity_mask, stage_weights):
        zero_mask = (target == 0).float() * activity_mask
        non_zero_pred = torch.abs(pred) * zero_mask
        loss = torch.relu(non_zero_pred - self.threshold) ** 2
        stage_weights_expanded = stage_weights.unsqueeze(-1).unsqueeze(-1)
        num_active = (zero_mask * stage_weights_expanded).sum().clamp(min=1e-6)
        return self.weight * (loss * stage_weights_expanded).sum() / num_active

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

class CumulativeLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred_cumulative, cumulative_transforms):
        return self.weight * self.l1_loss(pred_cumulative, cumulative_transforms)

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