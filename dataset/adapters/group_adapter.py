"""
adapters/group_adapter.py
=========================
Used by: Baseline 3 group model.
Returns: (crops_tensor, positions_tensor, group_label_int)
"""

import torch
import numpy as np
from PIL import Image
from dataset.adapters.base_adapter import BaseAdapter

class GroupAdapter(BaseAdapter):
    
    def build_index(self) -> list:
        return []
 
    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            cached_crops_arr, positions, label = self._cache[idx]
            
            crops = []
            for arr in cached_crops_arr:
                crop = Image.fromarray(arr)
                if self.transform:
                    crop = self.transform(crop)
                crops.append(crop)
                
            return (
                torch.stack(crops),
                torch.tensor(positions, dtype=torch.float32),
                label
            )

        sample    = self.raw[idx]
        img       = self.open_image(sample["img"])
        tokens    = sample["ann"]
        
        crops_to_cache = []
        crops          = []
        positions      = []

        i = 0
        while i < len(tokens):
            x, y, w, h = map(int, tokens[i:i + 4])

            crop = img.crop((x, y, x + w, y + h)).resize((256, 256))
            center_x = (x + w / 2) / img.width  - 0.5
            center_y = (y + h / 2) / img.height - 0.5

            crops_to_cache.append(np.array(crop, dtype=np.uint8))
            positions.append([center_x, center_y])

            if self.transform:
                crop = self.transform(crop)

            crops.append(crop)
            i += 5

        label = self.label_map[sample["label"]]

        self._cache[idx] = (crops_to_cache, positions, label)

        return (
            torch.stack(crops),
            torch.tensor(positions, dtype=torch.float32),
            label
        )