#!/usr/bin/env python
# coding: utf-8
# Your script, with small changes marked  [CHANGED n]

import sys
import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

local_rank  = int(os.environ["LOCAL_RANK"])   # number of this process INSIDE its node (0..3)
global_rank = int(os.environ["RANK"])         # [CHANGED 1] number of this process among ALL processes; 0 = the "leader"
dist.init_process_group(backend="gloo")       # gloo: the only backend that works with MIG slices

torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()

# [CHANGED 2] every process prints ONE line, so you can see that each one got its own GPU slice
print(f"[rank {global_rank}] cuda:{local_rank} -> {torch.cuda.get_device_name(local_rank)}", flush=True)


# detect which machine we are on
if os.path.exists("/nfs/slurm/assu002"):
    # we are on HPC                  
    PROJECT_ROOT = "/nfs/slurm/assu002/projects/volleyball_project/"
else:
    # we are on local PC
    PROJECT_ROOT = "/home/abdulrahmangamal/volleyball_project/"

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

base_cfg = load_yaml("configs/base_ddp.yaml")
b5_cfg   = load_yaml("configs/baseline5_ddp.yaml")


# ── fix data path based on machine ──────────────
ON_HPC = os.path.exists("/nfs/slurm/assu002")

if  not ON_HPC:
    base_cfg['dataset']['root'] = "/home/abdulrahmangamal/volleyball_data/videos"
    base_cfg['dataset']['tracking_annotation'] = "/home/abdulrahmangamal/volleyball_data/volleyball_tracking_annotation"
    b5_cfg['output']['root'] = "/home/abdulrahmangamal/outputs"


from torch import nn
from torch.utils.data import DataLoader
from torch import optim


root=base_cfg["dataset"]['root']
tracking_annotation=base_cfg["dataset"]["tracking_annotation"]
num_classes=base_cfg['dataset']['action_person']

train_videos=base_cfg['splits']['train']
val_videos=base_cfg['splits']['val']
num_workers=base_cfg['dataloader']['num_workers'] 
pre_fetch_factor=base_cfg['dataloader']['pre_fetch_factor'] 

###################################################################
freeze_backbone=b5_cfg['model']['freeze_backbone']

batch_size=b5_cfg['train']['batch_size']
epochs=b5_cfg['train']['epochs']
lr=b5_cfg['train']['lr']
weight_decay=b5_cfg['train']['weight_decay']
eta_min=b5_cfg['train']['eta_min']
drop_p=b5_cfg['train']['drop_p']
gamma=b5_cfg['train']['gamma']
hidden_dim=b5_cfg['train']['hidden_dim']
lstm_num_layer=b5_cfg['train']['lstm_num_layer']
print_perclass=b5_cfg['train']['print_PerClass']    
output_dir=b5_cfg['output']['root']


from utils.seed import  set_seed

from dataset.tracking_raw_dataset import TrackingRawDataset 
from dataset.adapters.tracking_adapter   import TrackingAdapter

from dataset.transforms  import B5PersonTransform 
from dataset.data_loader import build_dataloader
##############
from models.backbones.resnet50 import ResNet50
from models.baseline_model5.baseline5 import  B5Model


from trainers.DDP_base_trainer  import BaseTrainer

from utils.checkpoint import save_checkpoint
from utils.label_maps import PERSON_ACTION_TO_IDX
from utils.metrics import accuracy ,f1_calc

from losses.focal_loss import FocalLoss


set_seed()

tfm       = B5PersonTransform()
train_transforms = tfm.train()
val_transforms   = tfm.val()


train_raw_sample=TrackingRawDataset(root,tracking_annotation,train_videos)
val_raw_sample=TrackingRawDataset(root,tracking_annotation,val_videos)

train_dataset=TrackingAdapter(train_raw_sample,train_transforms,PERSON_ACTION_TO_IDX)
val_dataset=TrackingAdapter(val_raw_sample,val_transforms,PERSON_ACTION_TO_IDX)

# [CHANGED 3] print the sizes once (before: every process printed them -> 4 copies)
if global_rank == 0:
    print("train raw / val raw:", len(train_raw_sample), len(val_raw_sample))
    print("train / val samples:", len(train_dataset), len(val_dataset))


train_sampler = DistributedSampler(train_dataset,shuffle=True)
train_loader=build_dataloader(train_dataset,batch_size,num_workers,shuffle=False,sampler=train_sampler,drop_last=False,prefetch_factor=pre_fetch_factor )

val_sampler = DistributedSampler(val_dataset)
val_loader=build_dataloader(val_dataset,batch_size,num_workers,shuffle=False,sampler=val_sampler,drop_last=False,prefetch_factor=pre_fetch_factor )


device = torch.device(f"cuda:{local_rank}")

# [CHANGED 4] build the backbone ONCE, and really apply freeze_backbone from the yaml
#             (before: the flag was read but never used, so the backbone was always trainable)
backbone = ResNet50()

# [CHANGED 5] the variable is called `model` (before: `B5Model = B5Model(...)` replaced the class by the object)
model = B5Model(backbone,num_classes=num_classes,drop_p=drop_p,hidden_dim=hidden_dim,num_layers=lstm_num_layer,bidirectional=True)

model = model.to(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


optimizer=optim.AdamW(
    filter(lambda p :p.requires_grad,model.parameters()),
    lr=lr,
    weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,T_max=epochs,
    eta_min=eta_min
    )

loss_fn = FocalLoss(gamma=gamma)


trainer=BaseTrainer(
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
    class_map=PERSON_ACTION_TO_IDX,
    print_perclass=print_perclass,
)
trainer.train()


dist.destroy_process_group()