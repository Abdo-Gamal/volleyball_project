#!/usr/bin/env python
# coding: utf-8

# # project    
# 
# 
# 

# In[101]:
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()



import sys
import os
# detect which machine we are on
if os.path.exists("/nfs/slurm/assu002"):
    # we are on HPC                  
    PROJECT_ROOT = "/nfs/slurm/assu002/projects/volleyball_project/"
else:
    # we are on local PC
    PROJECT_ROOT = "/home/abdulrahmangamal/volleyball_project/"

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# In[102]:


import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

base_cfg = load_yaml("configs/base.yaml")
b5_cfg   = load_yaml("configs/baseline5.yaml")


# In[103]:



# ── fix data path based on machine ──────────────
ON_HPC = os.path.exists("/nfs/slurm/assu002")

if  not ON_HPC:
    base_cfg['dataset']['root'] = "/home/abdulrahmangamal/volleyball_data/videos"
    base_cfg['dataset']['tracking_annotation'] = "/home/abdulrahmangamal/volleyball_data/volleyball_tracking_annotation"
    b5_cfg['output']['root'] = "/home/abdulrahmangamal/outputs"


# In[104]:


import yaml
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim


# In[105]:


root=base_cfg["dataset"]['root']
tracking_annotation=base_cfg["dataset"]["tracking_annotation"]
num_classes=base_cfg['dataset']['action_person']

train_videos=base_cfg['splits']['train']
val_videos=base_cfg['splits']['val']
num_workers=base_cfg['dataloader']['num_workers'] 
pre_fetch_factor=base_cfg['dataloader']['pre_fetch_factor'] 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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




# In[106]:


from utils.seed import  set_seed

from dataset.tracking_raw_dataset import TrackingRawDataset 
from dataset.adapters.tracking_adapter   import TrackingAdapter

from dataset.transforms  import B5PersonTransform 
from dataset.data_loader import build_dataloader
##############
from models.backbones.resnet50 import ResNet50
from models.baseline_model5.baseline5 import  B5Model


from trainers.base_trainer  import BaseTrainer

from utils.checkpoint import save_checkpoint,load_checkpoint
from utils.label_maps import PERSON_ACTION_TO_IDX,PERSON_IDX_TO_ACTION
from utils.metrics import accuracy ,f1_calc
from utils.visualization import visualize_samples

from losses.focal_loss import FocalLoss


# In[107]:


set_seed()

tfm       = B5PersonTransform()
train_transfroms = tfm.train()
val_transforms   = tfm.val()

# In[108]:


train_raw_sample=TrackingRawDataset(root,tracking_annotation,train_videos)
val_raw_sample=TrackingRawDataset(root,tracking_annotation,val_videos)
print(len(train_raw_sample))
print(len(val_raw_sample))

# In[109]:


train_dataset=TrackingAdapter(train_raw_sample,train_transfroms,PERSON_ACTION_TO_IDX)
val_dataset=TrackingAdapter(val_raw_sample,val_transforms,PERSON_ACTION_TO_IDX)
print(len(train_dataset))
print(len(val_dataset))


# In[110]:


trainloader=build_dataloader(train_dataset,batch_size,num_workers,shuffle=True,sampler=None,drop_last=False,prefetch_factor=pre_fetch_factor )
valloader=build_dataloader(val_dataset,batch_size,num_workers,shuffle=False,sampler=None,drop_last=False,prefetch_factor=pre_fetch_factor )



# In[111]:


backbone=ResNet50()

# 1. build person model
backbone = ResNet50()
B5Model = B5Model(backbone,num_classes=num_classes,drop_p=drop_p,hidden_dim=hidden_dim,num_layers=lstm_num_layer,bidirectional=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print(f"=== Using {torch.cuda.device_count()} GPUs! ===")
    B5Model = torch.nn.DataParallel(B5Model)

B5Model = B5Model.to(device)


optimizer=optim.AdamW(
    filter(lambda p :p.requires_grad,B5Model.parameters()),
    lr=lr,
    weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,T_max=epochs,
    eta_min=eta_min
    )

loss_fn = FocalLoss(gamma=gamma)


# In[112]:


visualize_samples(train_dataset, label_map=PERSON_IDX_TO_ACTION)


# In[ ]:



trainer=BaseTrainer(
    model=B5Model,
    train_loader=trainloader,
    val_loader=valloader,
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

# In[ ]:




# In[ ]:



