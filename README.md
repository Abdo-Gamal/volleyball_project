A Hierarchical Deep Temporal Model for Group Activity RecognitionAn end-to-end PyTorch implementation and modern extension of the CVPR 2016 paper by Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, and Greg Mori.

---## 📖 Introduction & Problem Overview

Recognizing group activity in multi-agent sports scenes (e.g., volleyball) requires jointly modeling:1. Individual Person Dynamics: What each player is doing over time (e.g., spiking, setting, jumping).2. Collective Group Dynamics: How individual behaviors and team court positions compose the macro-level team activity (e.g., "Right Set", "Left Spike").

This repository implements a progressive benchmark of models from single-frame classifiers up to two-stage hierarchical spatio-temporal deep networks. The architecture pairs deep Convolutional Neural Networks (CNNs) with Long Short-Term Memory (LSTM) networks to aggregate player representations temporally and spatially into scene-level decisions.<p align="center">
<img src="images/fig1.png" alt="High-level Hierarchical Model" width="800">
<br>
<em>Figure 1: High-level hierarchical architecture. Individual dynamics are tracked temporally, then aggregated into a higher-level scene network.</em></p>

---## 🧠 Architectural Concepts & Mechanics### 1. Person vs. Group Temporal Dynamics<p align="center">
<img src="images/fig2.png" alt="Group vs Person Dynamics" width="800">
<br>
<em>Figure 2: Decoupled modeling of atomic person actions and high-level team activity.</em></p>### 2. End-to-End Hierarchical Pipeline<p align="center">
<img src="images/fig3.png" alt="Detailed Model Pipeline" width="800">
<br>
<em>Figure 3: Feature extraction from player tracklets, person-level LSTM (LSTM 1), pooling over all participants, and final activity classification via group-level LSTM (LSTM 2).</em></p>### 3. Spatial Court-Aware Pooling<p align="center">
<img src="images/fig4.png" alt="Spatial Pooling" width="800">
<br>
<em>Figure 4: Dividing players by court side (left vs. right) to preserve court formation geometry before temporal aggregation.</em></p>

---## 🏐 Dataset Specifications

The project evaluates on the expanded Volleyball Dataset:* Scale: 55 video matches (4,830 annotated keyframes across 55 videos).* Train / Val / Test Splits: Split at the video level to prevent sequence overlap and data leakage:  * Train Videos (24): 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54  * Validation Videos (15): 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51  * Test Videos (16): 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47* Temporal Window: 41 raw frames centered on the labeled frame ($t-20$ to $t+20$). Sequences utilize 9 frames ($t-4$ to $t+4$).* Resolutions: Videos 2, 37, 38, 39, 40, 41, 44, 45 are 1920×1080; all others are 1280×720.### Class Cardinality* 8 Group Activity Classes: Right set (644), Right spike (623), Right pass (801), Right winpoint (295), Left winpoint (367), Left pass (826), Left spike (642), Left set (633).* 9 Atomic Action Classes: Waiting (3,601), Setting (1,332), Digging (2,333), Falling (1,241), Spiking (1,216), Blocking (2,458), Jumping (341), Moving (5,121), Standing (38,696).

---## 📊 Experimental Benchmarks & Progression

Baseline

Architecture / Pipeline Summary

Original Paper Accuracy

Our Implementation

B1-Tuned

ResNet-50 fine-tuned on middle frame (whole scene)

66.7%

70.0% Acc

B3

Multi-task Person Model $\rightarrow$ Spatial Pos + Left/Right Attention Pooling

68.1%

83.0% Macro F1

B4

9-Frame Sequence $\rightarrow$ ResNet-50 $\rightarrow$ BiLSTM Frame Model

63.1%

82.2% Macro F1

B5

Tracklet-level BiLSTM $\rightarrow$ 2-Court Max Pooling $\rightarrow$ Group Classifier

67.6%

87.0% Macro F1

B6

Frame-by-frame person pooling $\rightarrow$ Sequence LSTM

74.7%

In Progress

B7

Full Two-Stage Model (Player LSTMs $\rightarrow$ Global Pool $\rightarrow$ Group LSTM)

80.2%

In Progress

B8

Full Two-Stage Model (Player LSTMs $\rightarrow$ 2-Team Spatial Pool $\rightarrow$ Group LSTM)

81.9%

In Progress

---## 🏗️ Software Architecture & Design Patterns

The codebase is built with modular OOP design patterns to decouple data management, feature modeling, and distributed execution.
volleyball_project/
├── configs/                     # Hyperparameter & runtime configurations (.yaml)
├── dataset/
│   ├── adapters/                # Template-pattern dataset adapters
│   │   ├── base_adapter.py      # BaseAdapter with image caching
│   │   ├── frame_adapter.py     # Frame extraction for Baseline 1
│   │   ├── group_adapter.py     # Person crop extraction with normalized spatial coords
│   │   ├── Clip_Adapter.py      # Frame sequence window extractor with directory cache
│   │   └── tracking_adapter.py  # Tracklet sequence adapters
│   ├── collect.py               # Custom collate functions with class balancing
│   ├── data_loader.py           # DataLoader builder with prefetching & pin_memory
│   ├── raw_dataset.py           # Raw image/annotation parsing
│   ├── TrackingGroupRawDataset.py# Multi-player clip-level trajectory parsing
│   ├── tracking_raw_dataset.py  # Single-player trajectory parsing
│   └── transforms.py            # Paired train/val transformations
├── models/
│   ├── backbones/resnet50.py   # Feature extractor backbone
│   ├── baseline_model1/         # B1 scene classifier
│   ├── baseline_model3/         # PersonModel & Attention GroupModel
│   ├── baseline_model4/         # B4 sequence BiLSTM
│   └── baseline_model5/         # B5 tracklet BiLSTM & GroupBaselineModel
├── trainers/
│   ├── base_trainer.py          # BaseTrainer with AMP, lr scheduling, and hooks
│   ├── DDP_base_trainer.py      # Multi-GPU trainer with cross-rank reduction
│   ├── person_trainer.py        # Multi-task loss tracking trainer
│   └── group_trainer.py         # Multi-input dictionary trainer (Single GPU & DDP)
├── runs/                        # Training execution entry points
├── run_volleyball.sh            # Slurm single-GPU execution script
└── run_volleyball_ddp.sh        # Slurm multi-GPU DDP execution script

1. Template Method Pattern (Data Adapters & Transforms)

BaseAdapter (dataset/adapters/base_adapter.py):
Defines the fixed execution skeleton: __init__ triggers build_index(), while __getitem__ coordinates loading, on-the-fly caching, and transforms. Derived classes (FrameAdapter, ClipAdapter, GroupAdapter) only implement build_index() and load_sample().

BaseTransform (dataset/transforms.py):
Enforces a strict transformation order (Resize $\rightarrow$ Crop $\rightarrow$ Augmentations $\rightarrow$ ToTensor $\rightarrow$ Normalize). Subclasses override get_train_augmentations() while evaluation code (val()) remains consistent and deterministic.

2. Hook Pattern (Training Pipelines)

BaseTrainer (trainers/base_trainer.py):
Orchestrates the training lifecycle (Mixed Precision torch.amp, gradient scaling, epoch loops, model checkpoints). Specialized trainers customize behavior by overriding specific hooks:

move_input(x): Overridden in GroupTrainer to route dictionary inputs ({"persons": ..., "positions": ...}).

compute_loss(outputs, y): Overridden in PersonTrainer to unpack multi-task objectives (coarse motion vs. fine action).

3. Spatial Court-Partitioned Pooling (models/baseline_model5/GroupBaselineModel.py)

Player coordinates are normalized relative to frame dimensions. The horizontal coordinate (mean_x) partitions players into court sides:
$$\text{Left Team} = {p \mid \text{mean}(x_p) < 0}, \quad \text{Right Team} = {p \mid \text{mean}(x_p) \ge 0}$$
Features are masked with $-\infty$ using masked_fill, max-pooled per team, and concatenated:
$$\mathbf{f}{\text{group}} = \left[ \max{p \in \text{Left}}(\mathbf{h}p) ;\Vert{}; \max{p \in \text{Right}}(\mathbf{h}_p) \right]$$

⚡ Performance & Engineering Optimizations

Class Balancing (dataset/collect.py):
The "standing" class accounts for >55% of all bounding boxes. person_collate() enforces a strict threshold (LIMITS = {"standing": 4}) to prevent optimizer domination.

Directory Listing Cache (Clip_Adapter.py):
Caches directory file structures in self._dir_cache to eliminate repeated OS-level os.listdir() calls during sequence construction.

RAM-Efficient uint8 Image Caching:
Pre-resizes raw crops to $256 \times 256$ and stores them in memory as np.uint8 arrays, cutting memory footprint by $4\times$ relative to 32-bit floats.

Dimension Collapsing for GPU Parallelism:
Processes player tracklets of shape $(B, N, T, C, H, W)$ by folding dimensions into $(B \cdot N, T, C, H, W)$, allowing full temporal CNN-LSTM processing in a single vectorized pass.

Zero-Redundancy Multi-GPU Synchronization (DDP_base_trainer.py):

Computes per-class counters and validation metrics across all ranks via dist.all_reduce and dist.all_gather_object.

Strips the DDP container before serializing checkpoints (model.module unwrapping) to ensure saved weights load cleanly in standard single-GPU environments.

🖥️ HPC Infrastructure & Slurm Setup

The project includes batch scripts optimized for compute nodes with shared network storage (NFS):

Local Node Caching: Automatically unpacks datasets directly into $LOCAL_TMP (/tmp/$USER) to avoid NFS I/O bottlenecks and prevent exceeding network drive quotas.

Pre-cached Model Weights: Sets TORCH_HOME to local scratch storage and downloads pretrained ResNet-50 weights prior to launching distributed workers, preventing multi-process race conditions.

Low-Overhead GPU Profiling: Runs an asynchronous background monitoring process via pynvml to log GPU compute and memory utilization to CSV every 10 seconds.

Submitting Slurm Jobs

Single-GPU Sequential Run:

sbatch run_volleyball.sh
Distributed Multi-GPU Run (DDP via torchrun):

Bash

sbatch run_volleyball_ddp.sh
🚀 Getting Started
1. Environment Setup
Bash

conda create -n vision_env python=3.10 -y
conda activate vision_env# Install PyTorch with CUDA support
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install pillow numpy pyyaml scikit-learn pynvml kaggle
2. Dataset Preparation
Ensure the Volleyball dataset is organized as follows:

volleyball_data/
├── videos/
│   ├── 0/
│   │   ├── annotations.txt
│   │   ├── 13286/
│   │   │   ├── 13276.jpg
│   │   │   └── ...
├── volleyball_tracking_annotation/
│   ├── 0/
│   │   └── 13286/
│   │       └── 13286.txt
3. Running Experiments Locally
Train Baseline 1 (Whole-Image ResNet-50):

Bash

python runs/baseline1/Run_Baseline1.py
Train Baseline 3 (Person Multi-Task & Attention Group Model):

Bash

python runs/baseline3/Run_Baseline3_person_model.py
python runs/baseline3/Run_Baseline3_group_model.py
Train Baseline 4 (Frame-Level BiLSTM):

Bash

python runs/baseline4/Run_Baseline4.py
Train Baseline 5 (Player Tracklet BiLSTM + Group Spatial Pooling):

Bash

# Distributed Multi-GPU Execution
torchrun --standalone --nproc_per_node=2 runs/baseline5/Run_GroupBaseline5_ddp.py
📜 Citations
Code snippet

@inproceedings{ibrahim2016hierarchical,
  title={A hierarchical deep temporal model for group activity recognition},
  author={Ibrahim, Mostafa S and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1971--1980},
  year={2016}
}

@article{ibrahim2018hierarchical,
  title={Hierarchical relational networks for group activity recognition and retrieval},
  author={Ibrahim, Mostafa S and Mori, Greg},
  journal={arXiv preprint arXiv:1811.09319},
  year={2018}
}
need put alll of this info and section in one markdown  ewdme file 