import torch.nn as nn

class Baseline1(nn.Module):
    def __init__(self, backbone, num_classes,drop_p=.5):
        super().__init__()

        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(backbone.out_dim, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

 
 
