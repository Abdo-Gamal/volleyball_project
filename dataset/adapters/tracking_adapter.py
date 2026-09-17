# import torch
# import numpy as np
# from PIL import Image
# from torchvision import tv_tensors
# from dataset.adapters.base_adapter import BaseAdapter


# class TrackingAdapter(BaseAdapter):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def load_sample(self, idx):
#         sample = self.raw[idx]
        
#         frames_list = sample["frames"][5:14]
#         boxes_list = sample["boxes"][5:14]

#         cropped_frames = []

#         for img_path, box in zip(frames_list, boxes_list):
#             img = Image.open(img_path).convert("RGB")
            
#             x1, y1, x2, y2 = map(int, box)

#             crop = img.crop((x1, y1, x2, y2))
#             crop = crop.resize((224, 224), Image.BILINEAR)

#             crop_tensor = torch.from_numpy(np.array(crop)).permute(2, 0, 1)
#             cropped_frames.append(crop_tensor)

#         clip_tensor = torch.stack(cropped_frames)
#         label_int = self.label_map[sample["label"]]

#         return tv_tensors.Video(clip_tensor), label_int


#using cache 

import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision import tv_tensors
from dataset.adapters.base_adapter import BaseAdapter


class TrackingAdapter(BaseAdapter):
    def __init__(self, *args, cache_size=1500, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def load_sample(self, idx):
        sample = self.raw[idx]

        # 1. Fetch from cache if available
        if idx in self.cache:
            clip_tensor = self.cache.pop(idx)
            # Reinsert to maintain LRU order
            self.cache[idx] = clip_tensor
            
            label_int = self.label_map[sample["label"]]
            # Return a cloned Video tensor to protect the cache from in-place transforms
            return tv_tensors.Video(clip_tensor.clone()), label_int

        # 2. Read from disk if not in cache
        frames_list = sample["frames"][5:14]
        boxes_list = sample["boxes"][5:14]

        cropped_frames = []

        for img_path, box in zip(frames_list, boxes_list):
            img = Image.open(img_path).convert("RGB")

            x1, y1, x2, y2 = map(int, box)

            # PIL crop safely handles out-of-bounds coordinates
            crop = img.crop((x1, y1, x2, y2))
            crop = crop.resize((224, 224), Image.BILINEAR)

            # Convert PIL to Numpy, then to Tensor [3, 224, 224] 
            # (uint8 is optimal for v2 transforms)
            crop_tensor = torch.from_numpy(np.array(crop)).permute(2, 0, 1)
            
            cropped_frames.append(crop_tensor)

        # Stack 9 frames to form [9, 3, 224, 224]
        clip_tensor = torch.stack(cropped_frames)

        # 3. Save the clean original tensor to cache
        self.cache[idx] = clip_tensor
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        label_int = self.label_map[sample["label"]]

        # Return a cloned Video tensor to protect the cached version
        return tv_tensors.Video(clip_tensor.clone()), label_int
