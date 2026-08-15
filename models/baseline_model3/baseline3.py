import torch
import torch.nn as nn
import torch.nn.functional as F



#multitask mode for person 
class PersonModel(nn.Module):
    def __init__(self, backbone, num_actions=9, num_coarse=3, drop_p=0.4):
        super().__init__()
        self.backbone = backbone

        self.shared = nn.Sequential(
            nn.Linear(backbone.out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p)
        )
        self.action_head = nn.Linear(512, num_actions)
        self.coarse_head = nn.Linear(512, num_coarse)
    def forward(self, x, return_feat=False):
        feat = self.backbone(x)
        feat = self.shared(feat)
        if return_feat:
            return feat
        
        return self.coarse_head(feat) , self.action_head(feat)


           ################################      


class GroupModel(nn.Module):
    def __init__(self, person_model, feat_dim=512, num_classes=8, drop_p=0.3):
        super().__init__()

        self.person_model = person_model
        self.person_model.eval()

        # Add spatial info
        self.projection = nn.Sequential(
            nn.Linear(feat_dim + 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )

        # -------- SIMPLE ATTENTION (NEW) --------
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1)
        )

        # Group classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        persons   = x["persons"]     # (B, N, 3, 224, 224)
        positions = x["positions"]   # (B, N, 2)

        B, N = persons.shape[:2]
        valid_mask = (persons.sum(dim=(2, 3, 4)) != 0)

        # 1) Person features
        persons = persons.view(B * N, 3, 224, 224)
        feats = self.person_model(persons, return_feat=True) #(B*12,2048)
        feats = feats.view(B, N, -1)

        # 2) Add position info
        feats = torch.cat([feats, positions], dim=-1)
        feats = self.projection(feats)     # (B, N, D)

        # 3) Attention scores
        attn_logits = self.attention(feats).squeeze(-1)  # (B, N)

        x_pos = positions[..., 0]
        left_mask  = (x_pos < 0) & valid_mask
        right_mask = (x_pos >= 0) & valid_mask

        neg_inf = torch.finfo(attn_logits.dtype).min

        # -------- LEFT ATTENTION POOL --------
        left_logits = attn_logits.masked_fill(~left_mask, neg_inf)
        left_weights = F.softmax(left_logits*.8, dim=1).unsqueeze(-1)
        left_vec = (feats * left_weights).sum(dim=1)

        # -------- RIGHT ATTENTION POOL --------
        right_logits = attn_logits.masked_fill(~right_mask, neg_inf)
        right_weights = F.softmax(right_logits*.8, dim=1).unsqueeze(-1)
        right_vec = (feats * right_weights).sum(dim=1)

        # 4) Group representation
        group_feat = torch.cat([left_vec, right_vec], dim=1)

        return self.classifier(group_feat)

