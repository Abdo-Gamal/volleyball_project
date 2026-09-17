
"""
Focal Loss with Class Weights (Alpha) and Modulating Factor (Gamma).

Mathematical Logic & Purpose:
1. Alpha (α): Solves the 'Class Imbalance' (Statistical Rarity) problem.
   - Computes weights based on inverse class frequency.
   - Scales UP the loss for under-represented classes (e.g., jumping).
   - Scales DOWN the loss for dominant classes (e.g., standing).
   - Ensures equal gradient contribution from all classes regardless of their size.

2. Gamma (γ): Solves the 'Feature Difficulty' (Visual Confusion) problem.
   - Targets the model's confidence (pt).
   - Scales DOWN the loss heavily for easy, highly-confident predictions (pt -> 1).
   - Maintains HIGH loss for hard, uncertain predictions (pt -> 0.5).
   - Forces the LSTM to stop ignoring subtle temporal differences (e.g., moving vs. waiting).

Combined, they prevent the model from getting stuck in a local minimum where it 
simply predicts the majority class, forcing it to learn fine-grained action features.
"""


#use in baseline3 and baseline5 person model
import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Tensor of shape [num_classes]

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)        
        loss = ((1 - pt) ** self.gamma * ce)
        
        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t
            
        return loss.mean()

import math
def get_class_weights_smoothed(class_counts: list) -> torch.Tensor:
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    
    weights = []
    for count in class_counts:
        if count == 0:
            weight = 0.0
        else:
            # 1. Calculate raw weight
            raw_weight = total_samples / (num_classes * count)
            # 2. Apply Square Root Smoothing
            smoothed_weight = math.sqrt(raw_weight)
        weights.append(smoothed_weight)
        
    return torch.tensor(weights, dtype=torch.float32)