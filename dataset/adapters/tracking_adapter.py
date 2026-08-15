import cv2
import torch
from dataset.adapters.base_adapter import BaseAdapter

class TrackingAdapter(BaseAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def build_index(self) -> list:
        return []
        
    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        sample = self.raw[idx]
        
        if idx in self.cache:
            clip_tensor = self.cache[idx]
        else:
            frames_list = sample["frames"][5:14]
            boxes_list = sample["boxes"][5:14]
            
            cropped_frames = []
            
            for img_path, box in zip(frames_list, boxes_list):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                x1, y1, x2, y2 = map(int, box)
                
                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (224, 224))
                
                crop_tensor = torch.from_numpy(crop).permute(2, 0, 1)
                cropped_frames.append(crop_tensor)
                
            clip_tensor = torch.stack(cropped_frames)
            
            self.cache[idx] = clip_tensor
            
        if self.transform:
            clip_tensor = self.transform(clip_tensor)
            
        label_int = self.label_map[sample["label"]]
        
        return clip_tensor, label_int