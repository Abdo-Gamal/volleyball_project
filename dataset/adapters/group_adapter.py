"""
adapters/group_adapter.py
=========================
Replaces the original group_adapter.py (GroupAdaptor → GroupAdapter, typo fixed).

Used by: Baseline 3 group model.
Returns: (crops_tensor, positions_tensor, group_label_int)

GroupAdapter overrides __getitem__ entirely because it returns 3 values
instead of 2. Everything else (open_image, __len__, __init__) is inherited
from BaseAdapter.
"""

import torch
from dataset.adapters.base_adapter import BaseAdapter

class GroupAdapter(BaseAdapter):
    """
    One sample per frame.
    Parses all person bboxes in the frame and returns them as a batch.

    Returns:
        crops     : FloatTensor[N, 3, 224, 224]  — all persons in the frame
        positions : FloatTensor[N, 2]            — normalized (cx, cy)
        label     : int                          — group action label
    """
    
    """ 
    must  override __getitem__ —  
          1-because transform applies per frame,not once  .
          
          2-because we return 3 values instead of 2  in load_sample.
            BaseAdapter expects this from load_sample():
            crop, label = self.load_sample(idx)       # 2 values
            
            GroupAdapter needs to return:
            crops, positions, label                   # 3 values → CRASH
          
          --------------------------------------
          load_sample() is not needed because we override __getitem__ directly.

    """
    def build_index(self) -> list:
        """_index is not used — __getitem__ reads self.raw directly."""
        return []
 
    def __len__(self) -> int:
        """One sample per raw frame."""
        return len(self.raw)


    def __getitem__(self, idx: int):
        """
        Overrides BaseAdapter.__getitem__ because we return 3 values.
        open_image() is still inherited from BaseAdapter.
        """
        sample    = self.raw[idx]
        img       = self.open_image(sample["img"])
        tokens    = sample["ann"]
        crops     = []
        positions = []

        i = 0
        while i < len(tokens):
            x, y, w, h = map(int, tokens[i:i + 4])

            crop     = img.crop((x, y, x + w, y + h))
            center_x = (x + w / 2) / img.width  - 0.5   # normalized, centered at 0
            center_y = (y + h / 2) / img.height - 0.5

            if self.transform:
                crop = self.transform(crop)

            crops.append(crop)
            positions.append([center_x, center_y])
            i += 5

        return (
            torch.stack(crops),                                      # (N, 3, H, W)
            torch.tensor(positions, dtype=torch.float32),            # (N, 2)
            self.label_map[sample["label"]],                         # int
        )

    # build_index() and load_sample() are not needed because we override
    # __getitem__ directly. BaseAdapter.__len__ is inherited and correct.
