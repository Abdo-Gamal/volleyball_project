import torch
from torch import nn
import torch.nn.functional as F
#use in baseline3 person model
class FocalLoss(nn.Module):
    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

