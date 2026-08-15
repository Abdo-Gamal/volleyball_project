"""
transforms.py  —  Template Method pattern
==========================================

ORIGINAL PROBLEM:
  6 standalone functions. The Normalize line was copy-pasted 6 times.
  One change = 6 edits. A new baseline = 2 more copy-pastes.

WHY TEMPLATE METHOD (not factory function):
  - Transforms always come in a PAIR: train() and val() must match.
  - A class keeps both paired in one object.
  - The pipeline ORDER never changes: resize → crop → [augment] → ToTensor → Normalize.
  - Only the [augment] block differs between baselines.
  → This is exactly what Template Method is for.

HOW IT WORKS:
  BaseTransform owns the full pipeline. Normalize is written ONCE here.
  get_train_augmentations() is the HOOK each subclass overrides.
  train() and val() are TEMPLATES — never override them.

NOTEBOOK usage:
  tfm       = B3FrameTransform()
  train_tfm = tfm.cached_train()   # use this — cache already did Resize
  val_tfm   = tfm.cached_val()     # use this — cache already did Resize
"""

from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────
#  BASE  (owns the skeleton — never change this class)
# ─────────────────────────────────────────────────────────────
class BaseTransform:
    resize_size: int = 256
    crop_size:   int = 224

    def get_train_augmentations(self) -> list:
        """Override in each subclass. Default: no augmentation."""
        return []

    # ── full pipeline (use when NO cache) ─────────────────────
    def train(self) -> transforms.Compose:
        steps = [
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.RandomCrop((self.crop_size, self.crop_size)),
        ]
        steps += self.get_train_augmentations()
        steps += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        return transforms.Compose(steps)

    def val(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.CenterCrop((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # ── short pipeline (use WITH cache — cache already did Resize) ──
    def cached_train(self) -> transforms.Compose:
        """
        Cache stores images at 256×256.
        This pipeline only needs: RandomCrop → augment → ToTensor → Normalize.
        """
        steps = [transforms.RandomCrop(self.crop_size)]
        steps += self.get_train_augmentations()
        steps += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        return transforms.Compose(steps)

    def cached_val(self) -> transforms.Compose:
        """
        Cache stores images at 256×256.
        This pipeline only needs: CenterCrop → ToTensor → Normalize.
        """
        return transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────
#  BASELINE 1
# ─────────────────────────────────────────────────────────────
class Baseline1Transform(BaseTransform):
    def get_train_augmentations(self) -> list:
        return [
            transforms.RandomRotation(3),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        ]


# ─────────────────────────────────────────────────────────────
#  BASELINE 3 — group / frame level
# ─────────────────────────────────────────────────────────────
class B3FrameTransform(BaseTransform):
    def get_train_augmentations(self) -> list:
        return [
            transforms.RandomRotation(3),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        ]


# ─────────────────────────────────────────────────────────────
#  BASELINE 3 — person level
# ─────────────────────────────────────────────────────────────
class PersonTransform(BaseTransform):
    """
    Person crops are tight bboxes — resize directly to 224, no extra crop.
    cached_train/cached_val are NOT used for person (no Resize in cache for persons).
    """
    resize_size: int = 224
    crop_size:   int = 224

    def get_train_augmentations(self) -> list:
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.1),
        ]

    def val(self) -> transforms.Compose:
        # tight person crops: just resize, no center-crop
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────
#  BASELINE 4 — LSTM
# ─────────────────────────────────────────────────────────────
class B4LSTMTransform(BaseTransform):
    """Applied per-frame before stacking into (T, C, H, W)."""
    def get_train_augmentations(self) -> list:
        return [
            transforms.RandomApply([transforms.RandomRotation(degrees=3)], p=0.7),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        ]

# ─────────────────────────────────────────────────────────────
#  BASELINE 5 — LSTM
# ─────────────────────────────────────────────────────────────
from torchvision.transforms import v2  

import torch

class B5PersonTransform:
    resize_size: int = 224
    crop_size:   int = 224

    def get_train_augmentations(self) -> list:
        return [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.RandomRotation(degrees=5)], p=0.7),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            v2.RandomGrayscale(p=0.05),
            v2.RandomAdjustSharpness(sharpness_factor=1.3, p=0.1),
        ]

    def train(self) -> v2.Compose:
        steps = []
        steps += self.get_train_augmentations()
        steps += [
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        return v2.Compose(steps)

    def val(self) -> v2.Compose:
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])