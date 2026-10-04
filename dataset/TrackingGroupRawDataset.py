
"""
What we want for Stage 2 (Clip-centric)

sample = {
    "video_id": "0",
    "clip_id": "13286",
    "group_label": "r_set",  #
    "players": [             # 
        {
            "track_id": 1,
            "frames": [
                "/volleyball_data/videos/0/13286/13276.jpg", 
                "/volleyball_data/videos/0/13286/13277.jpg",
                # ... for 20 images
            ],
            "boxes": [
                [446, 157, 580, 13276],  
                [446, 157, 580, 13277],  
                # ... for 20 boxes
            ]
        },
        {
            "track_id": 2,
            "frames": [
                "/volleyball_data/videos/0/13286/13276.jpg", 
                # ... for 20 images
            ],
            "boxes": [
                [100, 200, 150, 300],  
                # ... for 20 boxes
            ]
        },
        # ... (باقي اللاعبين في نفس الكليب، غالباً من 10 لـ 12 لاعب)
    ]
}
"""
import os
from collections import defaultdict
from torch.utils.data import Dataset

LABEL_FIX = {
    "r-pass": "r_pass",
    "l-pass": "l_pass",
    "r-set": "r_set",
    "l-set": "l_set",
    "r-spike": "r_spike",
    "l-spike": "l_spike",
}

class TrackingRawDataset(Dataset):
    def __init__(self, videos_root, tracking_root, video_ids):

        self.group_labels = self._load_group_labels(videos_root, video_ids)
        self.samples = []
        
        # Iterate over each video ID in the split
        for vid in video_ids: 
            vid_track_path = os.path.join(tracking_root, str(vid))
            if not os.path.exists(vid_track_path):
                continue

            # Iterate over each clip folder inside the video directory
            for clip_id in os.listdir(vid_track_path):  
                txt_file = os.path.join(vid_track_path, clip_id, f"{clip_id}.txt")
                if not os.path.exists(txt_file):
                    continue
                
                # Dictionary to group data per player (track_id)
                # 2D / Nested dictionary to group data per player (track_id)
                # Structure: {track_id: {"frames": [...], "boxes": [...] }}

                player_data = defaultdict(lambda: {"frames": [], "boxes": []})


                # Read the tracking annotation text file line by line
                #this loop make list of all player in the clip and for each player make list of 20 image and boxes and labels
                # Read the tracking annotation text file line by line

                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 10: 
                            continue
                        
                        track_id = int(parts[0])
                        box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                        frame_name = f"{parts[5]}.jpg"
                        
                        # Build absolute path
                        img_path = os.path.join(videos_root, str(vid), clip_id, frame_name)
                        
                        # Append data
                        player_data[track_id]["frames"].append(img_path)
                        player_data[track_id]["boxes"].append(box)
                        
                player_list = []
                for trk_id, data in player_data.items():
                    if len(data["frames"]) > 0:
                        player_list.append({ 
                            "track_id": trk_id,
                            "frames": data["frames"],       
                            "boxes": data["boxes"],         
                        })

                if len(player_list) > 0:
                    dict_key = f"{vid}_{clip_id}"
                    self.samples.append({
                        "video_id": vid,
                        "clip_id": clip_id,  
                        "group_label": self.group_labels.get(dict_key, "unknown"),  
                        "players": player_list,         
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _load_group_labels(self, root, video_ids):
        """
        Loads group labels for the specified video IDs.
        """
        clip_labels = {}
        for vid in video_ids:
            vid_path = os.path.join(root, str(vid))
            ann = os.path.join(vid_path, "annotations.txt")
            
            if not os.path.exists(ann):
                continue
            
            with open(ann, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                
                    img = parts[0]
                    label = parts[1]
                    clip_folder = img.replace(".jpg", "")
                    
                    clean_label = LABEL_FIX.get(label, label)
                    
                    clip_labels[f"{vid}_{clip_folder}"] = clean_label
                    
        return clip_labels


