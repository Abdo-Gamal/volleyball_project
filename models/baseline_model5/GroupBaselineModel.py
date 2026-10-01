import torch 
import torch.nn as nn

class GroupBaselineModel(nn.Module):
    def __init__(self, backbone, num_classes=8, drop_p=0.3, feature_dim=1024):
        # feature_dim is 1024 because the backbone uses a Bidirectional LSTM (512 * 2 = 1024)
        super().__init__() 
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Replace the person-level classifier with an Identity layer.
        # This allows the backbone to return the raw extracted features (1024-d) 
        # instead of the individual action logits.
        self.backbone.classifier = nn.Identity()  
        
        # The final group-level classifier to predict the overall scene activity
        self.classifier = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        # Input x represents a batch of clips.
        # Each clip has N players, and each player has a sequence of T frames.
        # IN:  (B, N, T, C, H, W) -> e.g., (Batch, 12, 9, 3, 224, 224)
        B, N, T, C, H, W = x.shape  
        
        # 1. Merge the Batch (B) and Players (N) dimensions.
        # The backbone expects a 5D tensor (Batch, Frames, Channels, Height, Width).
        # By treating each player in the batch as an independent sample, we can process them in parallel.
        # IN:  (B, N, T, C, H, W)
        # OUT: (B*N, T, C, H, W) -> e.g., (Batch * 12, 9, 3, 224, 224)
        x = x.view(B * N, T, C, H, W)  
        
        # 2. Extract features using the backbone (ResNet50 + Person LSTM).
        # IN:  (B*N, T, C, H, W)
        # OUT: (B*N, feature_dim) -> e.g., (Batch * 12, 1024)
        feats = self.backbone(x)       
        
        # 3. Unmerge the dimensions to separate the batch from the players again.
        # We need to isolate the N dimension to apply pooling over the players.
        # IN:  (B*N, feature_dim)
        # OUT: (B, N, feature_dim) -> e.g., (Batch, 12, 1024)
        feats = feats.view(B, N, -1)   

        # 4. Max Pooling over the N dimension (Players).
        # This aggregates the individual features into a single robust scene representation vector.
        # We use dim=1 because dim=0 is Batch, dim=1 is Players, and dim=2 is Features.
        # IN:  (B, N, feature_dim)
        # OUT: (B, feature_dim) -> e.g., (Batch, 1024)
        group_feats, _ = torch.max(feats, dim=1) 

        # 5. Final Classification for the Group Activity.
        # IN:  (B, feature_dim)
        # OUT: (B, num_classes) -> e.g., (Batch, 8)
        out = self.classifier(group_feats)      
        
        return out