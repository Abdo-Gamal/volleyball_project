#!/usr/bin/env python
# coding: utf-8

# # project    
# 
# 
# 

# In[82]:


import sys
import os
# detect which machine we are on
if os.path.exists("/nfs/slurm/assu002"):
    # we are on HPC
    PROJECT_ROOT = "/nfs/slurm/assu002/projects/volleyball_project/"
else:
    # we are on local PC
    PROJECT_ROOT = "/home/abdulrahmangamal/Desktop/volleyball_project"

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# In[83]:


import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

base_cfg = load_yaml("configs/base.yaml")
b3_cfg   = load_yaml("configs/baseline3_person.yaml")


# In[84]:



# ── fix data path based on machine ──────────────
ON_HPC = os.path.exists("/nfs/slurm/assu002")
if  not ON_HPC:
    base_cfg['dataset']['root'] = "/home/abdulrahmangamal/Desktop/volleyball_data/videos"
    b3_cfg['output']['root'] = "/home/abdulrahmangamal/Desktop/volleyball_project/outputs/baseline3"


# In[85]:


import yaml
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim


# In[ ]:


root=base_cfg["dataset"]['root']
action_person=base_cfg['dataset']['action_person']
num_calsses=base_cfg['dataset']['num_classes']

train_vedios=base_cfg['splits']['train']
val_vedios=base_cfg['splits']['val']
num_workers=base_cfg['dataloader']['num_workers'] 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

freeze_backbone=b3_cfg['model']['freeze_backbone']

batch_size=b3_cfg['train']['batch_size']
epochs=b3_cfg['train']['epochs']
lr=b3_cfg['train']['lr']
weight_decay=b3_cfg['train']['weight_decay']
eta_min=b3_cfg['train']['eta_min']
drop_p=b3_cfg['train']['drop_p']
gamma=b3_cfg['train']['gamma']
coarse_weight=b3_cfg['train']['coarse_weight']   ###
coarse_num=b3_cfg['train']['coarse_num']
print_perclass=b3_cfg['train']['print_evl_perclass']  
output_dir=b3_cfg['output']['root']



# In[87]:


from utils.seed import  set_seed

from dataset.raw_dataset import VolleyballRawDataset
from dataset.transforms  import PersonTransform 
from dataset.adapters.person_adapter import PersonAdapter
from dataset.data_loader import build_dataloader
from dataset.collect import person_collate ,group_collate

from models.backbones.resnet50 import ResNet50
from models.baseline_model3.baseline3 import  PersonModel


from trainers.person_trainer  import PersonTrainer 


from utils.checkpoint import save_checkpoint,load_checkpoint
from utils.label_maps import GROUP_ACTION_TO_IDX,GROUP_IDX_TO_ACTION, PERSON_ACTION_TO_IDX, PERSON_IDX_TO_ACTION
from utils.label_maps import MOTION_CLASSES,MOTION_TO_IDX,ACTION_TO_MOTION
from utils.metrics import accuracy ,f1_calc
from utils.visualization import visualize_samples

from losses.focal_loss import FocalLoss
from losses.multitask_loss import  MultiTaskLoss


# In[88]:


set_seed()

tfm       = PersonTransform()
train_transfroms = tfm.train()
val_transforms   = tfm.val()

# In[89]:


train_raw_sample=VolleyballRawDataset(root,train_vedios)
val_raw_sample=VolleyballRawDataset(root,val_vedios)
print(len(train_raw_sample))
print(len(val_raw_sample))

# In[90]:


train_dataset=PersonAdapter(train_raw_sample,train_transfroms,PERSON_ACTION_TO_IDX)
val_dataset=PersonAdapter(val_raw_sample,val_transforms,PERSON_ACTION_TO_IDX)

print(len(train_dataset))
print(len(val_dataset))


# In[91]:


trainloader=build_dataloader(train_dataset,batch_size,num_workers,collate_fn=person_collate,shuffle=True )
valloader=build_dataloader(val_dataset,batch_size,num_workers,shuffle=False)

# In[92]:


backbone=ResNet50()

model=PersonModel(backbone,
                  num_actions=action_person,
                  num_coarse=coarse_num,
                  drop_p=drop_p)


optimizer=optim.AdamW(
    filter(lambda p :p.requires_grad,model.parameters()),
    lr=lr,
    weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,T_max=epochs,
    eta_min=eta_min
    )

loss_fn = MultiTaskLoss(coarse_weight=coarse_weight,gamma=gamma)


# In[93]:


visualize_samples(train_dataset)

# In[ ]:


trainer = PersonTrainer(
    model=model, 
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
    print_perclass=print_perclass,            # was: print_evl_perclass=True
)
trainer.train()  


# In[ ]:




# In[ ]:



