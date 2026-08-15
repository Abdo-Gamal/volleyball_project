import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = model.fc.in_features

    def forward(self, x):
        x = self.backbone(x)
        return x.flatten(1)