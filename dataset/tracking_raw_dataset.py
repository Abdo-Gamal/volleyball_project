import os
from torch.utils.data import Dataset



"""
What we want 

sample = {
    "video_id": "0",
    "clip_id": "13286",
    "player_id": 1,
    "label": "standing",  #
    "frames": [
        "/volleyball_data/videos/0/13286/13276.jpg", 
        "/volleyball_data/videos/0/13286/13277.jpg",
        # ...for 20 image
    ],
    "boxes": [
        [446, 157, 580, 13276],  
        [446, 157, 580, 13277],  
        # ... for 20 image
    ]
}

"""

import os
from collections import defaultdict
class TrackingRawDataset:
    def __init__(self, videos_root, tracking_root, video_ids):
        self.samples = []
        
        # Iterate over each video ID in the split (e.g., train/val splits)
        for vid in video_ids: # 0 , 1 , 2 , 3 , ......  for train/test/ val have diffrnet id 
            vid_track_path = os.path.join(tracking_root, str(vid)) #path/0
            if not os.path.exists(vid_track_path):
                continue
            
            # Iterate over each clip folder inside the video directory
            for clip_id in os.listdir(vid_track_path):  # list all clip in vedios
                txt_file = os.path.join(vid_track_path, clip_id, f"{clip_id}.txt") #path/0/1232/1232.txt
                if not os.path.exists(txt_file):
                    continue
                
                # 2D / Nested dictionary to group data per player (track_id)
                # Structure: {track_id: {"frames": [...], "boxes": [...], "labels": [...]}}
                player_data = defaultdict(lambda: {"frames": [], "boxes": [], "labels": []})
                
                # Read the tracking annotation text file line by line
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 10: 
                            continue
                        
                        # Parse tracking data fields based on format:
                        # [track_id, xmin, ymin, xmax, ymax, frame_id, flag1, flag2, flag3, label]
                        track_id = int(parts[0])
                        box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                        frame_name = f"{parts[5]}.jpg"
                        label = parts[9]
                        
                        # Build absolute path to the actual image file in the raw video directory
                        img_path = os.path.join(videos_root, str(vid), clip_id, frame_name)
                        
                        # Append data into the nested structure for this specific track_id
                        player_data[track_id]["frames"].append(img_path)
                        player_data[track_id]["boxes"].append(box)
                        player_data[track_id]["labels"].append(label)
                
                # Flatten the nested dictionary into individual samples (one sample per player sequence)
                for trk_id, data in player_data.items():
                    if len(data["frames"]) > 0:
                        #dominant_label = max( set(data["labels"]), key=data["labels"].count)
                        middle_index = len(data["labels"]) // 2
                        self.samples.append({
                            "video_id": vid,
                            "clip_id": clip_id,
                            "track_id": trk_id,
                            "frames": data["frames"],       # List of 20 image file paths
                            "boxes": data["boxes"],         # List of 20 bounding boxes [xmin, ymin, xmax, ymax]
                            "label":  data["labels"][middle_index]      # Dominant action label for this sequence
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    