import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, weight=None, smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight,
                             label_smoothing=self.smoothing,
                             reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss