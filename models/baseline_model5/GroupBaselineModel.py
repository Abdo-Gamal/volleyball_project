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
            nn.Linear(self.feature_dim*2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, input_data):
        x=input_data["persons"] #(B,N,T,C,H,W)
        pos=input_data["positions"] #(B,N,T,2)

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

        #pos shape is (B, N,T, 2) where 2 is the x and y coordinates of each player
        # pos[..., 0] shape is (B, N, T) where each value is the x coordinate of each player
        # pos[..., 0].mean(dim=-1) shape is (B, N) where each value is the mean x coordinate of each player across the T frames
        
        mean_x = pos[..., 0].mean(dim=-1) # Extract the x-coordinates of player positions for potential use in attention or pooling.
        valid_mask =(pos.abs().sum(dim=(-1,-2))>0) #(B,N)
       
        left_mask= (mean_x <0) & valid_mask # (B,N) #which players are on the left side 
        right_mask= (mean_x >=0) & valid_mask  #(b,N)

        # make large nigative number for invalid players so that they don't affect the max pooling
        neg_inf = torch.finfo(feats.dtype).min


        # 4. Expand the mask to match the feature tensor's dimensions for Broadcasting.
        # We add a dummy dimension at the end so the mask shape matches the features (B, N, 1024).
        # IN: left_mask shape (B, N).
        # OUT: mask_expanded shape (B, N, 1). During execution, it broadcasts to (B, N, 1024).
        # 5. Apply the mask to the features using masked_fill.
        # Note: masked_fill replaces the value with neg_inf wherever the mask is TRUE.
        # We use ~ (Logical NOT) to invert the mask, meaning: "If the player is NOT in the left team, set their features to -inf".
        # OUT: left_feats shape is (B, N, 1024), where non-left players have -inf features.
        
        left_feats=feats.masked_fill(~left_mask.unsqueeze(-1), neg_inf) #(B,N,feature_dim)
        right_feats=feats.masked_fill(~right_mask.unsqueeze(-1), neg_inf) #(B,N,feature_dim)

      
        # 4. Max Pooling over the N dimension (Players).
        # This aggregates the individual features into a single robust scene representation vector.
        # We use dim=1 because dim=0 is Batch, dim=1 is Players, and dim=2 is Features.
        # IN:  (B, N, feature_dim)
        # OUT: (B, feature_dim) -> e.g., (Batch, 1024)
        left_feats, _ = torch.max(left_feats, dim=1) 
        right_feats, _ = torch.max(right_feats, dim=1) 

        group_feats=torch.cat([left_feats,right_feats],dim=-1) #(B,feature_dim*2)
 
        # 5. Final Classification for the Group Activity.
        # IN:  (B, feature_dim*2)
        # OUT: (B, num_classes) -> e.g., (Batch, 8)
        out = self.classifier(group_feats)      
        
        return out