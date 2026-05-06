import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, weight=None, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        focal_weight = (1 - pt) ** self.gamma
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CostSensitiveLoss(nn.Module):
    def __init__(self, base_criterion, fn_penalty=1.0, device='cpu'):
        super().__init__()
        self.base_criterion = base_criterion
        self.fn_penalty = fn_penalty
        self.device = device

    def forward(self, logits, targets):
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        loss = self.base_criterion(logits, targets)

        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        loss = loss.view(-1)
        targets = targets.view(-1)

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            preds = preds.view(-1)
            fn_mask = (preds == 0) & (targets == 1)

        penalty = torch.ones_like(loss)
        fn_mask = fn_mask.view(-1)

        if fn_mask.any():
            penalty[fn_mask] = self.fn_penalty

        weighted_loss = loss * penalty
        return weighted_loss.mean()
