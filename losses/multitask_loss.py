import torch
from torch import nn
import torch.nn.functional as F
from losses.focal_loss import FocalLoss 
from utils.label_maps import build_targets

class MultiTaskLoss(nn.Module):
    def __init__(self, coarse_weight=0.05, gamma=0.0):
        super().__init__()
        self.motion_loss = nn.CrossEntropyLoss()
        self.action_loss = FocalLoss(gamma)
        self.coarse_weight = coarse_weight

    def forward(self, outputs, labels):
        motion_logits, action_logits = outputs
        motion_targets,action_targets = build_targets(labels)

        loss_action = self.action_loss(action_logits, action_targets)
        loss_motion = self.motion_loss(motion_logits, motion_targets)

        total = (
            loss_action
            + self.coarse_weight * loss_motion
        )

        return total, loss_action,loss_motion, action_targets


