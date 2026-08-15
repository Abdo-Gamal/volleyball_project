import torch
from utils.label_maps import PERSON_ACTION_TO_IDX ,GROUP_ACTION_TO_IDX 


################################# baseline3

#use this to minmize number of berson class 
def person_collate(batch):
    all_crops, all_labels = [], []
    # Strict cap for the "Bullies"
    LIMITS = {PERSON_ACTION_TO_IDX["standing"]: 4}
    STANDING_COUNT = 0
    for crop, label in batch:
        if label in LIMITS:
            if STANDING_COUNT < LIMITS[label]:
                all_crops.append(crop)
                all_labels.append(label)
                STANDING_COUNT += 1
        else:
            all_crops.append(crop)
            all_labels.append(label)
    return torch.stack(all_crops), torch.tensor(all_labels)


          ########################################

def group_collate(batch):
    """
     batch is  list of list 

    batch element:
      crops     -> list of tensors [(3,224,224), ...]
      positions -> list of [x,y]
      label     -> int
    """
    
    B = len(batch)
    max_n = max(len(item[0]) for item in batch)

    # Allocate tensors
    persons_batch = torch.zeros(B, max_n, 3, 224, 224)
    pos_batch = torch.zeros(B, max_n, 2)
    labels = torch.zeros(B, dtype=torch.long)

    for i, (crops, positions, label) in enumerate(batch):
        n = len(crops)
        # Stack crops (they belong together)
        persons_batch[i, :n] = crops
        # Convert positions list → tensor
        pos_batch[i, :n] =positions
        labels[i] = label
    
    x={"persons":persons_batch,
    "positions":pos_batch }
    return x, labels

################################### baseline....


