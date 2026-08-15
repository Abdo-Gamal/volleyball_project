"""
=========================
Used by: Baseline 4 (whole-frame classification).

How to think when adding any new baseline
Every baseline in a deep learning project touches the same 5 layers, always in this order:
DATA  →  MODEL  →  LOSS  →  TRAIN LOOP  →  CONFIG

"""
"""
adapters/clip_adapter.py
========================
Used by: Baseline 4 (LSTM on frame sequences).

Returns: (clip_tensor, group_label_int)
    clip_tensor : FloatTensor[9, 3, 224, 224]
                  9 frames centered on the annotated keyframe
                  (4 before + keyframe + 4 after)

WHY __getitem__ is overridden:
  BaseAdapter.__getitem__ applies transform to ONE image.
  ClipAdapter must apply transform to EACH of 9 frames separately,
  then stack them — same reason GroupAdapter overrides __getitem__.
  open_image(), __len__, __init__ are all still inherited.

FRAME SELECTION — keyframe-centered:
  The annotation keyframe is the exact frame where the action is labeled.
  Taking 4 before + keyframe + 4 after gives the model:
    - buildup context  (before)
    - labeled moment   (keyframe)
    - follow-through   (after)
  This is better than linspace because linspace might skip the keyframe.
"""

"""
        Data out of adapter: (9, 3, 224, 224) per sample
        After collate: (B, 9, 3, 224, 224)
        After backbone loop: (B, 9, feat_dim)
        After LSTM: (B, hidden_dim)
        After classifier: (B, num_classes)

"""

import os
import torch
from dataset.adapters.base_adapter import BaseAdapter

def _get_centered_frames(clip_dir: str, keyframe_name: str, n: int = 4) -> list:
    """
    Returns 2*n+1 frame filenames centered on keyframe_name.

    If keyframe is near the edge of the clip, missing frames are
    padded by repeating the first or last available frame.

    Args:
        clip_dir      : folder that contains all .jpg frames for the clip
        keyframe_name : filename of the annotated keyframe e.g. '73290.jpg'
        n             : number of frames before AND after keyframe

    Returns:
        list of 2*n+1 filenames (always exactly this length)
    """
    all_frames = sorted([
        f for f in os.listdir(clip_dir) if f.endswith(".jpg")
    ])

    key_idx = all_frames.index(keyframe_name)

    start = max(0, key_idx - n)
    end   = min(len(all_frames) - 1, key_idx + n)

    selected = all_frames[start : end + 1]

    return selected   # always 2*n+1 filenames


class ClipAdapter(BaseAdapter):
    """
    One sample per clip.
    Loads 9 frames centered on the annotated keyframe.

    Returns:
        clip  : FloatTensor[9, 3, 224, 224]
        label : int  —  group action label
    """

    def __init__(self, raw_dataset, transform, label_map: dict, n_frames: int = 4):
        """
        Args:
            n_frames : frames before AND after keyframe (default 4 → 9 total)
        """
        self.n_frames = n_frames
        super().__init__(raw_dataset, transform, label_map)
        # NOTE: super().__init__ calls build_index() which uses default
        # (one int per raw sample) — that is correct for ClipAdapter.

    def build_index(self) -> list:
        """ClipAdapter does not use _index — __getitem__ reads self.raw directly."""
        return []          # empty: no wasted memory, no dead code
 
    def __len__(self) -> int:
        """One clip per raw sample — explicit and clear.
            what is the meanig of __len()__ ? how many times can __getitem__ be called
        """
        return len(self.raw)
 
 
    def __getitem__(self, idx: int):
        """
        Overrides BaseAdapter.__getitem__ because transform is applied
        per-frame before stacking, not to the whole clip.
        open_image() is still inherited from BaseAdapter.
        """
        sample       = self.raw[idx]
        clip_dir     = os.path.dirname(sample["img"])
        keyframe     = os.path.basename(sample["img"])   # e.g. '73290.jpg'

        frame_names  = _get_centered_frames(clip_dir, keyframe, n=self.n_frames)

        frames = []
        for name in frame_names:
            img = self.open_image(os.path.join(clip_dir, name))  # inherited
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        clip  = torch.stack(frames)                  # (9, 3, 224, 224)
        label = self.label_map[sample["label"]]      # int

        return clip, label

    # build_index() uses BaseAdapter default → list(range(len(raw)))  ✓
    # load_sample() not needed → __getitem__ is overridden directly   ✓
    # __len__   inherited from BaseAdapter                            ✓
    # open_image inherited from BaseAdapter                           ✓
