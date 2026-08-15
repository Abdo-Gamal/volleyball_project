#!/usr/bin/env python
# coding: utf-8

# # project    
# 
# 
# 

# In[21]:


import sys
import os
# detect which machine we are on
if os.path.exists("/nfs/slurm/assu002"):
    # we are on HPC                  
    PROJECT_ROOT = "/nfs/slurm/assu002/projects/volleyball_project/"
else:
    # we are on local PC
    PROJECT_ROOT = "/home/abdulrahmangamal/Desktop/volleyball_project/"

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# In[22]:


import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

base_cfg = load_yaml("configs/base.yaml")
b3_cfg   = load_yaml("configs/baseline3_group.yaml")


# In[23]:



# ── fix data path based on machine ──────────────
ON_HPC = os.path.exists("/nfs/slurm/assu002")

if  not ON_HPC:
    base_cfg['dataset']['root'] = "/home/abdulrahmangamal/Desktop/volleyball_data/videos"
    b3_cfg['output']['root'] = "/home/abdulrahmangamal/Desktop/outputs"
    b3_cfg['model']['path'] = "/home/abdulrahmangamal/Desktop/volleyball_project/outputs/baseline3/PersonModel_best.pth"


# In[24]:


import yaml
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim


# In[25]:


root=base_cfg["dataset"]['root']
action_person=base_cfg['dataset']['action_person']
num_calsses=base_cfg['dataset']['num_classes']

train_vedios=base_cfg['splits']['train']
val_vedios=base_cfg['splits']['val']
num_workers=base_cfg['dataloader']['num_workers'] 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
###################################################################
freeze_backbone=b3_cfg['model']['freeze_backbone']
model_path=b3_cfg['model']['path']

batch_size=b3_cfg['train']['batch_size']
epochs=b3_cfg['train']['epochs']
lr=b3_cfg['train']['lr']
weight_decay=b3_cfg['train']['weight_decay']
eta_min=b3_cfg['train']['eta_min']
drop_p=b3_cfg['train']['drop_p']
gamma=b3_cfg['train']['gamma']
print_perclass=b3_cfg['train']['print_evl_perclass']    
output_dir=b3_cfg['output']['root']




# In[26]:


from utils.seed import  set_seed

from dataset.raw_dataset import VolleyballRawDataset
from dataset.transforms  import B3FrameTransform 
from dataset.adapters.group_adapter import GroupAdapter
from dataset.data_loader import build_dataloader
from dataset.collect import  group_collate

from models.backbones.resnet50 import ResNet50
from models.baseline_model3.baseline3 import  GroupModel,PersonModel


from trainers.group_trainer  import GroupTrainer


from utils.checkpoint import save_checkpoint,load_checkpoint
from utils.label_maps import GROUP_ACTION_TO_IDX,GROUP_IDX_TO_ACTION, PERSON_ACTION_TO_IDX, PERSON_IDX_TO_ACTION
from utils.metrics import accuracy ,f1_calc
from utils.visualization import visualize_samples

from losses.focal_loss import FocalLoss


# In[27]:


set_seed()

tfm       = B3FrameTransform()
train_transfroms = tfm.train()
val_transforms   = tfm.val()

# In[28]:


train_raw_sample=VolleyballRawDataset(root,train_vedios)
val_raw_sample=VolleyballRawDataset(root,val_vedios)
print(len(train_raw_sample))
print(len(val_raw_sample))

# In[29]:


train_dataset=GroupAdapter(train_raw_sample,train_transfroms,GROUP_ACTION_TO_IDX)
val_dataset=GroupAdapter(val_raw_sample,val_transforms,GROUP_ACTION_TO_IDX)
print(len(train_dataset))
print(len(val_dataset))


# In[30]:


trainloader=build_dataloader(train_dataset,batch_size,num_workers,shuffle=True,collate_fn=group_collate,sampler=None,drop_last=False )
valloader=build_dataloader(val_dataset,batch_size,num_workers,shuffle=False,collate_fn=group_collate,sampler=None,drop_last=False )



# In[31]:


backbone=ResNet50()

# 1. build person model
backbone = ResNet50()
person_model = PersonModel(backbone)

# 2. load trained weights
ckpt = torch.load(model_path, map_location=device)
person_model.load_state_dict(ckpt["model_state"])
# 3. use inside group model
model = GroupModel(person_model,drop_p=drop_p)


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


# In[32]:


#visualize_samples(train_dataset)

# In[33]:



trainer=GroupTrainer(
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
    class_map=GROUP_ACTION_TO_IDX,
    print_perclass=print_evl_perclass,
)
trainer.train()

# In[ ]:




# In[ ]:



