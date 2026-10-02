
"""
What we want for Stage 2 (Clip-centric)

sample = {
    "video_id": "0",
    "clip_id": "13286",
    "group_label": "r_set",  #
    "players": [             # 
        {
            "track_id": 1,
            "frames": [
                "/volleyball_data/videos/0/13286/13276.jpg", 
                "/volleyball_data/videos/0/13286/13277.jpg",
                # ... for 20 images
            ],
            "boxes": [
                [446, 157, 580, 13276],  
                [446, 157, 580, 13277],  
                # ... for 20 boxes
            ]
        },
        {
            "track_id": 2,
            "frames": [
                "/volleyball_data/videos/0/13286/13276.jpg", 
                # ... for 20 images
            ],
            "boxes": [
                [100, 200, 150, 300],  
                # ... for 20 boxes
            ]
        },
        # ... (باقي اللاعبين في نفس الكليب، غالباً من 10 لـ 12 لاعب)
    ]
}
"""

import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision import tv_tensors
from dataset.adapters.base_adapter import BaseAdapter

class TrackingAdapter(BaseAdapter):
    def __init__(self, *args, cache_size=1000, max_players=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.max_players = max_players  

    def load_sample(self, idx):
        sample = self.raw[idx]
        player_data = sample["players"]
        
        # 1. Check cache
        if idx in self.cache:
            group_tensor, positions, label_int = self.cache.pop(idx)
            # Fix: Save all three items back to cache
            self.cache[idx] = (group_tensor, positions, label_int) 
            return tv_tensors.Video(group_tensor.clone()), positions, label_int

        # 2. Pre-allocate tensors
        group_tensor = torch.zeros((self.max_players, 9, 3, 224, 224), dtype=torch.uint8)
        group_pos = torch.zeros((self.max_players, 9, 2), dtype=torch.float32)
        label_int = self.label_map[sample["group_label"]]

        for i, player in enumerate(player_data):
            if i >= self.max_players:
                break
                
            frames = player["frames"][5:14]
            boxes = player["boxes"][5:14]
            
            cropped_frames = []
            positions = []
            
            for path, box in zip(frames, boxes):
                img = Image.open(path).convert("RGB")
                
                # Assuming data is [x, y, x2, y2] based on your previous code
                x, y, x2, y2 = map(int, box)
                crop = img.crop((x, y, x2, y2))
                crop = crop.resize((224, 224), Image.BILINEAR)

               # Calculate center: average of x1 and x2
                # Normalize: divide by image width [0, 1]
                # Zero-center: subtract 0.5
                center_x = ((x + x2 )/ 2.0) / img.width - 0.5     
                center_y = ((y + y2) / 2.0) / img.height - 0.5
                positions.append([center_x, center_y])

                # Convert to Tensor and change format to (Channels, H, W)
                crop_tensor = torch.from_numpy(np.array(crop)).permute(2, 0, 1)
                cropped_frames.append(crop_tensor)

            # Stack 9 frames into one tensor (dim=0 by default)
            clip_tensor = torch.stack(cropped_frames)
            
            group_tensor[i] = clip_tensor
            
            # Convert Python list of floats to PyTorch Tensor
            group_pos[i] = torch.tensor(positions, dtype=torch.float32)

        # 3. Update cache
        self.cache[idx] = (group_tensor, group_pos, label_int) 
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return tv_tensors.Video(group_tensor.clone()), group_pos, label_int