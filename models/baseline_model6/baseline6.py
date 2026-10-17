import torch 
import torch.nn as nn


class B6Model(nn.Module):
    def __init__(self, backbone, num_classes=8, drop_p=0.3, feature_dim=512):
        # feature_dim is 512 because the backbone return 512 
        super().__init__() 
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.lstm=nn.LSTM(
            input_size=feature_dim*2,
            hidden_size=feature_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        
        )
        
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
        
        # 1. Merge Batch (B), Players (N), and Time (T) to process all crops in parallel
        # IN:  (B, N, T, C, H, W)
        # OUT: (B * N * T, C, H, W)
        x = x.view(B * N*T, C, H, W)  
        
        # 2. Extract features using the backbone
        # IN:  (B * N * T, C, H, W)
        # OUT: (B * N * T, feature_dim) -> e.g., (Batch * 12 * 9, 512)
        feats = self.backbone(x,return_feat=True)    

        # 3. Unflatten to restore individual dimensions
        # OUT: (B, N, T, 512)
        feats = feats.view(B, N, T, self.feature_dim)

        #pos shape is (B, N,T, 2) where 2 is the x and y coordinates of each player
        # pos[..., 0] shape is (B, N, T) where each value is the x coordinate of each player
        # pos[..., 0].mean(dim=-1) shape is (B, N) where each value is the mean x coordinate of each player across the T frames
        
        mean_x = pos[..., 0].mean(dim=-1) # Extract the x-coordinates of player positions for potential use in attention or pooling.
        valid_mask =(pos.abs().sum(dim=(-1,-2))>0) #in (B,N,T,2) --> out:(B,N)
       
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
        
        left_feats=feats.masked_fill(~left_mask.unsqueeze(-1).unsqueeze(-1), neg_inf) #(B,N,T,feature_dim)
        right_feats=feats.masked_fill(~right_mask.unsqueeze(-1).unsqueeze(-1), neg_inf) #(B,N,T,feature_dim)

       # 6. Max Pooling over the N dimension (Players) per frame
        # This creates a single feature vector for each frame.
        # IN:  (B * T, N, 512)
        # OUT: (B * T, 512)
        left_pooling, _ = torch.max(left_feats, dim=1)  #out (B,T,feature_dim)
        right_pooling, _ = torch.max(right_feats, dim=1) 


        group_feats=torch.cat([left_pooling,right_pooling],dim=-1) #(B,feature_dim*2)
        lstm_out,(h_n,c_n)=self.lstm(group_feats) #(B,feature_dim*2)
        final_feat=torch.cat([h_n[-2],h_n[-1]],dim=-1) #(B,feature_dim*2)
 
        # 5. Final Classification for the Group Activity.
        # IN:  (B, feature_dim*2)
        # OUT: (B, num_classes) -> e.g., (Batch, 8)
        out = self.classifier(final_feat)      
        
        return out