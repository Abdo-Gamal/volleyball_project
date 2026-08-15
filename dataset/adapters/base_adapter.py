"""
adapters/base_adapter.py  —  Template Method pattern
=====================================================

ORIGINAL PROBLEM:
  Three adapter classes (FrameAdapter, PersonAdapter, GroupAdaptor) share:
    - __init__ signature (raw_dataset, transform, label_map)
    - __len__
    - Image.open().convert('RGB')
  All three copy these identically.

HOW IT WORKS:
  BaseAdapter owns: __init__, __len__, open_image()
  Subclasses override:
    build_index()   → called once in __init__, returns list of indices
    load_sample()   → returns (image_crop, label) for one index

  __getitem__ applies transform then returns — written once.

EXCEPTION:
  GroupAdaptor overrides __getitem__ entirely because it returns 3 values
  (crops, positions, label) instead of 2. That is valid and intentional.
"""

from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class BaseAdapter(Dataset):
    """
    Template Method base for all dataset adapters.

    Fixed skeleton:
        __init__ → build_index() → stores index
        __getitem__ → load_sample(idx) → apply transform → return

    Subclasses override:
        build_index() → list of index tokens
        load_sample(idx) → (PIL.Image or crop, label_int)

    Subclasses may override __getitem__ if the return signature differs
    (e.g. GroupAdaptor returns 3 values).
    """

    def __init__(self, raw_dataset, transform, label_map: dict):
        self.raw       = raw_dataset
        self.transform = transform
        self.label_map = label_map
        self._index    = self.build_index()   # HOOK — builds the sample list
        self._cache: dict = {}            # for open_image() caching — shared by all subclasses, no need to re-implement


    # ── HOOK 1 — what does the index look like? ───────────────────────────────
    def build_index(self) -> list:
        """Return a list of index tokens.
        Default: one integer per raw sample.
        PersonAdapter overrides this to return (img_id, ann_offset) pairs."""
        return list(range(len(self.raw)))

    # ── HOOK 2 — how to load one sample ──────────────────────────────────────
    def load_sample(self, idx):
        """Return (image_or_crop, label_int).
        Must be overridden by every subclass."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement load_sample()"
        )

    # ── SHARED UTILITY ────────────────────────────────────────────────────────
    def open_image(self, path):
        if path not in self._cache:
            img = Image.open(path).convert("RGB").resize((256, 256))
            self._cache[path] = np.array(img, dtype=np.uint8)
        return Image.fromarray(self._cache[path])
   
    # reads from Drive ONCE, stores in RAM as 256×256, reuses every epoch

    # ── TEMPLATE — never override (unless return type differs) ────────────────
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx):
        crop, label = self.load_sample(idx)
        if self.transform:
            crop = self.transform(crop)
        return crop, label