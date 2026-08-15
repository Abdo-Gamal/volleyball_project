
GROUP_LABELS = [
    "r_pass", "l_pass", "r_set", "l_set",
    "r_spike", "l_spike", "r_winpoint", "l_winpoint"
]

PERSON_ACTIONS = [
    "standing", "moving", "jumping", "spiking",
    "blocking", "digging", "setting", "falling",'waiting'
]

GROUP_ACTION_TO_IDX = {k: i for i, k in enumerate(GROUP_LABELS)}
GROUP_IDX_TO_ACTION = { i:k for i, k in enumerate(GROUP_LABELS)}

PERSON_ACTION_TO_IDX = {k: i for i, k in enumerate(PERSON_ACTIONS)}
PERSON_IDX_TO_ACTION = {i: k for i, k in enumerate(PERSON_ACTIONS)}



########################## use in baseline3 person model
import torch 

# MOTION_CLASSES
MOTION_CLASSES = ["idle", "defense", "attack"]
MOTION_TO_IDX = {c: i for i, c in enumerate(MOTION_CLASSES)}


# Map action → motion
ACTION_TO_MOTION = {
    "standing": "idle",
    "waiting":  "idle",
    "moving":   "attack",
    "digging":  "defense",
    "spiking":  "attack",
    "blocking": "defense",
    "jumping":  "attack",
    "falling":  "attack",
    "setting":  "attack",
}

ACTION_TO_MOTION_IDX = torch.tensor([                     #[0,0,2,1,2,1,2,2,2]
    MOTION_TO_IDX[ACTION_TO_MOTION[PERSON_IDX_TO_ACTION[i]]]
    for i in range(len(PERSON_ACTION_TO_IDX))
])


def build_targets(action_labels):
    """
    action_labels: Tensor[B]
    """
    action_targets = action_labels.clone()
    motion_targets = ACTION_TO_MOTION_IDX.to(action_labels.device)[action_labels]

    return motion_targets,action_targets
                   #################################