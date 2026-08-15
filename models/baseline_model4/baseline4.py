import torch
import torch.nn as nn


class B4ClipModel(nn.Module):

    """
    Baseline 4 — End-to-end clip classification with LSTM.

    Architecture:
        ResNet50 backbone
            → project 2048 → hidden_dim   (follows PersonModel pattern)
            → LSTM over 9 frames
            → dropout + classifier

    Shape flow:
        input  : (B, T, 3, 224, 224)     T = number of frames (9)
        reshape: (B*T, 3, 224, 224)
        backbone: (B*T, out_dim)          out_dim = 2048 for ResNet50
        project: (B*T, hidden_dim)
        reshape: (B, T, hidden_dim)
        lstm   : h_n → (B, hidden_dim)   last hidden state only
        output : (B, num_classes)
    """

    def __init__(self, backbone, hidden_dim=512, num_classes=8,
                 num_layers=1, drop_p=0.3, bidirectional=True):
        super().__init__()

        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(backbone.out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p)
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(
            input_size   = hidden_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout      = drop_p if num_layers > 1 else 0.0
        )

        # hidden_dim * 2 if bidirectional else hidden_dim
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(lstm_out_dim, num_classes)
        )

    def forward(self, x):                        # (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)              # (B*T, 3, 224, 224)
        x = self.backbone(x)                     # (B*T, out_dim)
        x = self.projection(x)                   # (B*T, hidden_dim)
        x = x.view(B, T, -1)                     # (B, T, hidden_dim)
        x = self.norm(x)                         # (B, T, hidden_dim)

        _, (h_n, _) = self.lstm(x)
        
        # bidirectional: concat last forward + last backward
        if self.lstm.bidirectional:
            x = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, hidden_dim*2)
        else:
            x = h_n[-1]                                  # (B, hidden_dim)

        return self.classifier(x)                        # (B, num_classes)