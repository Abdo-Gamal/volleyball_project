#!/usr/bin/env python
# coding: utf-8

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()
import sys
import yaml

# =========================================================================
# 1. Detect Machine (HPC vs Local) & Setup Paths
# =========================================================================
ON_HPC = os.path.exists("/nfs/slurm/assu002")

if ON_HPC:
    # We are on HPC
    PROJECT_ROOT = "/nfs/slurm/assu002/projects/volleyball_project/"
else:
    PROJECT_ROOT = "/home/abdulrahmangamal/volleyball_project/"

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load configurations
base_cfg = load_yaml("configs/base.yaml")
b1_cfg   = load_yaml("configs/baseline1.yaml")

if  not ON_HPC:
    base_cfg['dataset']['root'] = "/home/abdulrahmangamal/volleyball_data/videos"
    b1_cfg['output']['root'] = "/home/abdulrahmangamal/outputs"


# =========================================================================
# 2. Imports
# =========================================================================
from torch.utils.data import DataLoader
from torch import optim

from dataset.raw_dataset import VolleyballRawDataset
from dataset.adapters.frame_adapter import FrameAdapter

from dataset.transforms import Baseline1Transform

from models.backbones.resnet50 import ResNet50
from models.baseline_model1.baseline1 import Baseline1 

from trainers.base_trainer import BaseTrainer 

from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed
from utils.metrics import accuracy, f1_calc
from utils.label_maps import GROUP_ACTION_TO_IDX 

# =========================================================================
# 3. Initialization & Hyperparameters
# =========================================================================
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root        = base_cfg["dataset"]["root"]
split_train = base_cfg["splits"]["train"]
split_val   = base_cfg["splits"]["val"]

batch_size   = b1_cfg["train"]["batch_size"]
epochs       = b1_cfg["train"]["epochs"]
lr           = b1_cfg["train"]["lr"]
weight_decay = b1_cfg["train"]["weight_decay"]
eta_min      = b1_cfg["train"]["eta_min"]
output_dir   = b1_cfg["output"]["root"]

num_workers = os.cpu_count() if not ON_HPC else base_cfg.get('dataloader', {}).get('num_workers', 4)

# =========================================================================
# 4. Datasets & DataLoaders
# =========================================================================
tfm = Baseline1Transform()
transform_train = tfm.train()
transform_val   = tfm.val()

train_raw = VolleyballRawDataset(root, split_train)
val_raw   = VolleyballRawDataset(root, split_val)

print(f"Total Train Videos: {len(train_raw)}")
print(f"Total Val Videos: {len(val_raw)}")


Adapter = FrameAdapter
train_ds = Adapter(train_raw, transform_train, GROUP_ACTION_TO_IDX)
val_ds   = Adapter(val_raw, transform_val, GROUP_ACTION_TO_IDX)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# =========================================================================
# 5. Model, Optimizer, and Scheduler
# =========================================================================
num_classes = len(GROUP_ACTION_TO_IDX)

model = Baseline1(backbone=ResNet50(), num_classes=num_classes)

if torch.cuda.device_count() > 1:
    print(f"=== Using {torch.cuda.device_count()} GPUs! ===")
    model = torch.nn.DataParallel(model)

model = model.to(device)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=weight_decay
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, 
    T_max=epochs,
    eta_min=eta_min
)

loss_fn = torch.nn.CrossEntropyLoss()

os.makedirs(output_dir, exist_ok=True)

# =========================================================================
# 6. Training with New BaseTrainer
# =========================================================================
trainer = BaseTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    accuracy=accuracy,
    f1_score=f1_calc,
    save_checkpoint=save_checkpoint,
    device=device,
    epochs=epochs,
    output_dir=output_dir,
    class_map=GROUP_ACTION_TO_IDX,         
    print_perclass=False,                  
    checkpoint_name="baseline1_best.pth"  
)

print("Starting Baseline 1 Training...")
trainer.train()
print("Training Completed Successfully!")