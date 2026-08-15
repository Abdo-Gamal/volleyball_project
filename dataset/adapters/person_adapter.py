"""
adapters/person_adapter.py
==========================
Replaces the original person_adapter.py.

Used by: Baseline 3 person model.
Returns: (person_crop_tensor, action_label_int)

build_index() expands the raw dataset:
  raw has one entry per frame.
  PersonAdapter expands to one entry per PERSON per frame.
  Each index token is (img_id, annotation_offset).
"""

from dataset.adapters.base_adapter import BaseAdapter


class PersonAdapter(BaseAdapter):
    """
    Expands one-entry-per-frame into one-entry-per-person.
    Crops each person bbox with a 15% padding on all sides.
    """

    PAD: float = 0.15

    def build_index(self) -> list:
        """
        Walk every frame in self.raw.
        For each annotation token that has a valid action label,
        store (img_id, annotation_byte_offset) as one index entry.
        """
        index = []
        for img_id in range(len(self.raw)):
            tokens = self.raw[img_id]["ann"]
            i = 0
            while i < len(tokens):
                action = tokens[i + 4].strip().lower()
                if action in self.label_map:
                    index.append((img_id, i))
                i += 5
        return index

    def load_sample(self, idx: int):
        """
        Args:
            idx: index into self._index → (img_id, ann_offset)

        Returns:
            (PIL.Image crop, action_label_int)
        """
        img_id, ann_idx = self._index[idx]
        sample = self.raw[img_id]
        img    = self.open_image(sample["img"])
        tokens = sample["ann"]

        x, y, w, h = map(int, tokens[ann_idx:ann_idx + 4])
        action      = tokens[ann_idx + 4].strip().lower()

        # Pad bbox by PAD fraction on each side
        x1 = int(max(0,          x - w * self.PAD))
        y1 = int(max(0,          y - h * self.PAD))
        x2 = int(min(img.width,  x + w * (1 + self.PAD)))
        y2 = int(min(img.height, y + h * (1 + self.PAD)))

        return img.crop((x1, y1, x2, y2)), self.label_map[action]