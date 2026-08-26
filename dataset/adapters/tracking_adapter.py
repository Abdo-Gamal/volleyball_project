import torch
import numpy as np
from PIL import Image
from dataset.adapters.base_adapter import BaseAdapter

class TrackingAdapter(BaseAdapter):
    
    def load_sample(self, idx: int):
        real_idx = self._index[idx]
        sample = self.raw[real_idx]
        
        if real_idx in self._cache:
            clip_tensor = torch.from_numpy(self._cache[real_idx])
        else:
            frames_list = sample["frames"][5:14]
            boxes_list = sample["boxes"][5:14]
            cropped_frames = []
            
            for img_path, box in zip(frames_list, boxes_list):
                img = self.open_image(img_path)
                
                x1, y1, x2, y2 = map(int, box)
                crop = img.crop((x1, y1, x2, y2)).resize((256, 256))
                
                arr = np.array(crop, dtype=np.uint8)
                crop_tensor = torch.from_numpy(arr).permute(2, 0, 1)
                cropped_frames.append(crop_tensor)
                
            clip_tensor = torch.stack(cropped_frames)
            
            self._cache[real_idx] = clip_tensor.numpy()
        
        label_int = self.label_map[sample["label"]]
        
        return clip_tensor, label_int