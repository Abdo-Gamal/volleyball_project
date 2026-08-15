import os
from torch.utils.data import Dataset

LABEL_FIX = {
    "r-pass": "r_pass",
    "l-pass": "l_pass",
    "r-set": "r_set",
    "l-set": "l_set",
    "r-spike": "r_spike",
    "l-spike": "l_spike",
}

class VolleyballRawDataset(Dataset):
    def __init__(self, root, video_ids):  # root is vedios path, ids is vedois number  0 to 52
        self.samples = []
        for vid in video_ids:
            vid_path = os.path.join(root, str(vid))
            ann = os.path.join(vid_path, "annotations.txt")
            #ann like root/0/annotations.txt
            if not os.path.exists(ann):
               continue
            
            clips_in_video = set(os.listdir(vid_path))
            #print("work in annotations file ",vid)
            with open(ann) as f:
                for line in f:
                    parts = line.strip().split()
                     # Skip empty lines
                    if not parts:
                        continue
                     # Check format
                    if len(parts) < 2:
                        continue

                    img = parts[0]
                    label = parts[1]
                    clip_folder= img.replace(".jpg","")

                     # Check if clip folder exists
                    if clip_folder not in clips_in_video:
                        continue
                    img_path = os.path.join(vid_path,clip_folder, img )
                    self.samples.append({
                              "img": img_path,
                              "label": LABEL_FIX.get(label,label),
                              "ann": parts[2:]
                          })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
