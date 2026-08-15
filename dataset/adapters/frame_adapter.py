"""
adapters/frame_adapter.py
=========================
Used by: Baseline 1 (whole-frame classification).
"""

from base_adapter import BaseAdapter

class FrameAdapter(BaseAdapter):
    """
    One sample per frame.
    Returns: (frame_tensor, group_label_int)

    build_index() is inherited from BaseAdapter (one int per raw sample).
    """

    def load_sample(self, idx: int):
        """
        Args:
            idx: integer index into self.raw

        Returns:
            (PIL.Image, label_int)
        """
        sample = self.raw[idx]
        img    = self.open_image(sample["img"])   
        label  = self.label_map[sample["label"]]
        return img, label