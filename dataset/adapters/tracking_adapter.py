import cv2
import torch

from collections import OrderedDict
from dataset.adapters.base_adapter import BaseAdapter


class TrackingAdapter(BaseAdapter):

    def __init__(
        self,
        *args,
        cache_size=3000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.cache_size = cache_size

        self.cache = OrderedDict()

    def load_sample(self, idx):

        sample = self.raw[idx]

        if idx in self.cache:

            clip_tensor = self.cache.pop(idx)

            self.cache[idx] = clip_tensor

        else:

            frames_list = sample["frames"][5:14]
            boxes_list = sample["boxes"][5:14]

            cropped_frames = []

            for img_path, box in zip(frames_list, boxes_list):

                img = cv2.imread(img_path)

                if img is None:
                    raise FileNotFoundError(
                        f"Could not read image: {img_path}"
                    )

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                x1, y1, x2, y2 = map(int, box)

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    raise ValueError(
                        f"Empty crop for image={img_path}, box={box}"
                    )

                crop = cv2.resize(crop, (224, 224))

                crop_tensor = torch.from_numpy(crop).permute(2, 0, 1)

                cropped_frames.append(crop_tensor)

            # [9, 3, 224, 224]
            clip_tensor = torch.stack(cropped_frames)

            self.cache[idx] = clip_tensor

            if len(self.cache) > self.cache_size:

                self.cache.popitem(last=False)

        label_int = self.label_map[sample["label"]]

        return clip_tensor, label_int