"""
adapters/frame_adapter.py
=========================
Used by: Baseline 1 (whole-frame classification).
"""
from dataset.adapters.base_adapter import BaseAdapter
import numpy as np
from PIL import Image

class FrameAdapter(BaseAdapter):
    """
    Standard frame-level adapter for Baseline 1 (Whole-frame classification).
        One sample per frame.
        Returns: (frame_tensor, group_label_int)
    
        build_index() is inherited from BaseAdapter (one int per raw sample).
    """

    def load_sample(self, idx: int):
        sample = self.raw[idx]
        path = sample["img"]

        if path not in self._cache:
            img = self.open_image(path).resize((256, 256))
            self._cache[path] = np.array(img, dtype=np.uint8)
            
        img = Image.fromarray(self._cache[path])
        
        action = sample["label"] 
        label = self.label_map[action]

        return img, label