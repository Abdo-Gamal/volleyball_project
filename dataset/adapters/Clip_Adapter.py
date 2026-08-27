"""
adapters/clip_adapter.py
========================
Used by: Baseline 4 (LSTM on frame sequences).

IDEA:
  Extracts a sequence of frames (a clip) centered around a specific annotated keyframe.
  Instead of loading the entire video, it only loads the necessary temporal window.

GOAL:
  Provide the LSTM model with a (T, C, H, W) tensor, where T is the time dimension (frames),
  to understand actions over time.
"""

import os
import torch
import numpy as np
from PIL import Image
from dataset.adapters.base_adapter import BaseAdapter

class ClipAdapter(BaseAdapter):
    """
    Adapter for processing temporal clips.
    """

    def __init__(self, raw_dataset, transform, label_map: dict, n_frames: int = 4):
        """
        GOAL: Initialize the adapter with the temporal window size and a directory cache.
        
        USAGE:
          n_frames: Controls the window size. If n_frames=4, the clip will have:
                    4 frames before + 1 keyframe + 4 frames after = 9 frames total.
        """
        self.n_frames = n_frames
        super().__init__(raw_dataset, transform, label_map)
        
        # IDEA: Cache the list of filenames for each directory.
        # GOAL: Prevent slow disk I/O operations (os.listdir) in every iteration.
        self._dir_cache = {} 

    def build_index(self) -> list:
        """ClipAdapter does not use _index — __getitem__ reads self.raw directly."""
        return []
 
    def __len__(self) -> int:
        """One clip per raw sample."""
        return len(self.raw)
 
    def _get_centered_frames(self, clip_dir: str, keyframe_name: str, n: int = 4) -> list:
        """
        IDEA: 
          Given a directory and a keyframe, return the filenames of the frames 
          surrounding that keyframe.
          
        GOAL & OPTIMIZATION:
          Check if the directory contents are already in memory (_dir_cache).
          If not, read from disk (os.listdir) ONCE and store it.
        """
        # 1. Check if we already read this folder's contents before
        if clip_dir not in self._dir_cache:
            # If not, read it from disk, filter .jpg, sort them, and save to cache
            self._dir_cache[clip_dir] = sorted([
                f for f in os.listdir(clip_dir) if f.endswith(".jpg")
            ])
            
        # 2. Get the full list of frames directly from RAM (super fast)
        all_frames = self._dir_cache[clip_dir]
        
        # 3. Find the exact index of our keyframe
        key_idx = all_frames.index(keyframe_name)

        # 4. Calculate start and end indices (with safety to not go out of bounds)
        start = max(0, key_idx - n)
        end   = min(len(all_frames) - 1, key_idx + n)

        # 5. Return the sliced list of filenames
        return all_frames[start : end + 1]

    def __getitem__(self, idx: int):
        """
        IDEA: 
          Constructs the full clip tensor and applies image caching.
          
        GOAL:
          1. Get frame names using _get_centered_frames.
          2. Open each frame, resize to 256, and cache it as a memory-efficient numpy array.
          3. Apply transforms (which will RandomCrop from 256 to 224).
          4. Stack them into a final (T, C, H, W) tensor.
        """
        sample       = self.raw[idx]
        clip_dir     = os.path.dirname(sample["img"])
        keyframe     = os.path.basename(sample["img"])

        # Fetch the filenames for the temporal window
        frame_names  = self._get_centered_frames(clip_dir, keyframe, n=self.n_frames)

        frames = []
        for name in frame_names:
            img_path = os.path.join(clip_dir, name)
            
            # --- IMAGE CACHING LOGIC ---
            # Read, resize to 256 (for BaseTransform to crop later), and save as uint8 to save RAM
            if img_path not in self._cache:
                img = self.open_image(img_path).resize((256, 256))
                self._cache[img_path] = np.array(img, dtype=np.uint8)
            
            # Retrieve from cache
            img = Image.fromarray(self._cache[img_path])
            
            # Apply Augmentations / Transforms (e.g., RandomCrop 224)
            if self.transform:
                img = self.transform(img)
                
            frames.append(img)

        # Stack into (T, C, H, W)
        clip  = torch.stack(frames)                  
        label = self.label_map[sample["label"]]      

        return clip, label