# A Hierarchical Deep Temporal Model for Group Activity Recognition

> **An end-to-end PyTorch implementation and modern extension of the CVPR 2016 paper by Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, and Greg Mori.**

---

## 📖 Introduction & Problem Overview

Recognizing group activity in multi-agent sports scenes such as volleyball requires jointly modeling two levels of behavior:

1. **Individual Person Dynamics** — what each player is doing over time, such as spiking, setting, jumping, blocking, or falling.
2. **Collective Group Dynamics** — how individual behaviors and team court positions combine to form a high-level group activity, such as **Right Set**, **Left Spike**, or **Right Winpoint**.

This repository implements a progressive benchmark of models, starting from single-frame classifiers and progressing toward hierarchical spatio-temporal deep networks. The architecture combines deep **Convolutional Neural Networks (CNNs)** with **Long Short-Term Memory (LSTM)** networks to aggregate player representations temporally and spatially into scene-level decisions.

<p align="center">
  <img src="images/fig1.png" alt="High-level Hierarchical Model" width="600">
  <br>
  <em>Figure 1 — High-level hierarchical architecture. Individual dynamics are tracked temporally, then aggregated into a higher-level scene network.</em>
</p>

---

## 🧠 Architectural Concepts & Mechanics

### 1. Person vs. Group Temporal Dynamics

<p align="center">
  <img src="images/fig2.png" alt="Group vs Person Dynamics" width="600">
  <br>
  <em>Figure 2 — Decoupled modeling of atomic person actions and high-level team activity.</em>
</p>

The problem is decomposed into two related levels:

- **Person level:** learn temporal representations of individual player actions.
- **Group level:** aggregate information from multiple players to recognize the collective activity.

This hierarchical formulation allows the model to capture both local player behavior and global team behavior.

### 2. End-to-End Hierarchical Pipeline

<p align="center">
  <img src="images/fig3.png" alt="Detailed Model Pipeline" width="600">
  <br>
  <em>Figure 3 — Feature extraction from player tracklets, person-level LSTM (LSTM 1), pooling over participants, and final activity classification via group-level LSTM (LSTM 2).</em>
</p>

The hierarchical pipeline follows the sequence:

```text
Player Tracklets
      ↓
CNN Feature Extraction
      ↓
Person-Level LSTM (LSTM 1)
      ↓
Spatial / Participant Pooling
      ↓
Group-Level LSTM (LSTM 2)
      ↓
Group Activity Classification
```

### 3. Spatial Court-Aware Pooling

<p align="center">
  <img src="images/fig4.png" alt="Spatial Pooling" width="600">
  <br>
  <em>Figure 4 — Dividing players by court side (left vs. right) to preserve court formation geometry before temporal aggregation.</em>
</p>

Instead of treating all players as an unordered collection, the model uses normalized player coordinates to distinguish the two sides of the volleyball court. This preserves useful spatial information about team formation.

---

# 🏐 Dataset Specifications

The project evaluates on the expanded **Volleyball Dataset**.

## Dataset Scale

- **55 video matches**
- **4,830 annotated keyframes** across the 55 videos
- Split at the **video level** to prevent sequence overlap and data leakage

## Train / Validation / Test Split

### Training Videos — 24

```text
1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31,
32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54
```

### Validation Videos — 15

```text
0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51
```

### Test Videos — 16

```text
4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47
```

## Temporal Sampling

Each labeled frame is associated with a temporal window of **41 raw frames**, centered on the labeled frame:

```text
t - 20  ...  t - 1  [ t ]  t + 1  ...  t + 20
```

The implemented sequence models use **9 frames** centered around the target frame:

```text
t - 4  ...  t - 1  [ t ]  t + 1  ...  t + 4
```

## Video Resolution

The following videos have resolution **1920 × 1080**:

```text
2, 37, 38, 39, 40, 41, 44, 45
```

All other videos have resolution **1280 × 720**.

## Class Cardinality

### 8 Group Activity Classes

| Group Activity | Count |
|---|---:|
| Right set | 644 |
| Right spike | 623 |
| Right pass | 801 |
| Right winpoint | 295 |
| Left winpoint | 367 |
| Left pass | 826 |
| Left spike | 642 |
| Left set | 633 |

### 9 Atomic Action Classes

| Atomic Action | Count |
|---|---:|
| Waiting | 3,601 |
| Setting | 1,332 |
| Digging | 2,333 |
| Falling | 1,241 |
| Spiking | 1,216 |
| Blocking | 2,458 |
| Jumping | 341 |
| Moving | 5,121 |
| Standing | 38,696 |

> **Note:** The strong imbalance of the atomic-action labels, particularly the `Standing` class, is explicitly addressed by the training pipeline through class balancing.

---

# 📊 Experimental Benchmarks & Progression

The project follows a progressive benchmark from simple single-frame recognition toward hierarchical spatio-temporal models.

| Baseline | Architecture / Pipeline Summary | Original Paper Accuracy | Our Implementation |
|:---|:---|---:|---:|
| **B1-Tuned** | ResNet-50 fine-tuned on middle frame (whole scene) | 66.7% | **70.0% Acc** |
| **B3** | Multi-task Person Model → Spatial Position + Left/Right Attention Pooling | 68.1% | **83.0% Macro F1** |
| **B4** | 9-Frame Sequence → ResNet-50 → BiLSTM Frame Model | 63.1% | **82.2% Macro F1** |
| **B5** | Tracklet-level BiLSTM → 2-Court Max Pooling → Group Classifier | 67.6% | **87.0% Macro F1** |
| **B6** | Frame-by-frame person pooling → Sequence LSTM | 74.7% | *In Progress* |
| **B7** | Full Two-Stage Model (Player LSTMs → Global Pool → Group LSTM) | 80.2% | *In Progress* |
| **B8** | Full Two-Stage Model (Player LSTMs → 2-Team Spatial Pool → Group LSTM) | 81.9% | *In Progress* |

---

# 🏗️ Software Architecture & Design Patterns

The codebase is built with modular **Object-Oriented Programming (OOP)** design patterns to decouple data management, feature modeling, training, and distributed execution.

## Project Structure

```text
volleyball_project/
│
├── configs/                         # Hyperparameter & runtime configurations (.yaml)
│
├── dataset/
│   ├── adapters/                    # Template-pattern dataset adapters
│   │   ├── base_adapter.py          # BaseAdapter with image caching
│   │   ├── frame_adapter.py         # Frame extraction for Baseline 1
│   │   ├── group_adapter.py         # Person crop extraction with normalized spatial coords
│   │   ├── Clip_Adapter.py          # Frame sequence window extractor with directory cache
│   │   └── tracking_adapter.py      # Tracklet sequence adapters
│   │
│   ├── collect.py                   # Custom collate functions with class balancing
│   ├── data_loader.py               # DataLoader builder with prefetching & pin_memory
│   ├── raw_dataset.py               # Raw image/annotation parsing
│   ├── TrackingGroupRawDataset.py   # Multi-player clip-level trajectory parsing
│   ├── tracking_raw_dataset.py      # Single-player trajectory parsing
│   └── transforms.py                # Paired train/val transformations
│
├── models/
│   ├── backbones/resnet50.py        # Feature extractor backbone
│   ├── baseline_model1/             # B1 scene classifier
│   ├── baseline_model3/             # PersonModel & Attention GroupModel
│   ├── baseline_model4/             # B4 sequence BiLSTM
│   └── baseline_model5/             # B5 tracklet BiLSTM & GroupBaselineModel
│
├── trainers/
│   ├── base_trainer.py              # BaseTrainer with AMP, LR scheduling, and hooks
│   ├── DDP_base_trainer.py          # Multi-GPU trainer with cross-rank reduction
│   ├── person_trainer.py            # Multi-task loss tracking trainer
│   └── group_trainer.py             # Multi-input dictionary trainer (Single GPU & DDP)
│
├── runs/                            # Training execution entry points
├── run_volleyball.sh                 # Slurm single-GPU execution script
└── run_volleyball_ddp.sh             # Slurm multi-GPU DDP execution script
```

---

## 1. Template Method Pattern — Data Adapters & Transforms

### `BaseAdapter` — `dataset/adapters/base_adapter.py`

Defines the fixed execution skeleton:

- `__init__` triggers `build_index()`.
- `__getitem__` coordinates sample loading, on-the-fly caching, and transforms.
- Derived classes such as `FrameAdapter`, `ClipAdapter`, and `GroupAdapter` implement `build_index()` and `load_sample()`.

### `BaseTransform` — `dataset/transforms.py`

Enforces a strict transformation order:

```text
Resize
  ↓
Crop
  ↓
Augmentations
  ↓
ToTensor
  ↓
Normalize
```

Training subclasses override `get_train_augmentations()`, while evaluation through `val()` remains consistent and deterministic.

---

## 2. Hook Pattern — Training Pipelines

### `BaseTrainer` — `trainers/base_trainer.py`

`BaseTrainer` orchestrates the main training lifecycle, including:

- Mixed Precision using `torch.amp`
- Gradient scaling
- Epoch loops
- Model checkpointing
- Training hooks

Specialized trainers customize behavior by overriding specific hooks.

### `move_input(x)`

Overridden in `GroupTrainer` to route dictionary inputs such as:

```python
{"persons": ..., "positions": ...}
```

### `compute_loss(outputs, y)`

Overridden in `PersonTrainer` to unpack the multi-task objectives, including **coarse motion** and **fine-grained action** predictions.

---

## 3. Spatial Court-Partitioned Pooling

Implemented in:

```text
models/baseline_model5/GroupBaselineModel.py
```

Player coordinates are normalized relative to the frame dimensions. The horizontal coordinate `mean_x` is then used to partition players into the two court sides:

$$
\text{Left Team} = \{p \mid \text{mean}(x_p) < 0\}
$$

$$
\text{Right Team} = \{p \mid \text{mean}(x_p) \ge 0\}
$$

The features are masked with $-\infty$ using `masked_fill`, max-pooled independently for each team, and then concatenated:

$$
\mathbf{f}_{\text{group}}
=
\left[
\max_{p \in \text{Left}}(\mathbf{h}_p)
\;\Vert\;
\max_{p \in \text{Right}}(\mathbf{h}_p)
\right]
$$

This preserves the distinction between the left and right sides of the volleyball court before group-level classification.

---

# ⚡ Performance & Engineering Optimizations

## 1. Class Balancing — `dataset/collect.py`

The `Standing` class accounts for more than **55% of all bounding boxes**. To reduce optimizer domination by this highly frequent class, `person_collate()` enforces a strict threshold:

```python
LIMITS = {"standing": 4}
```

## 2. Directory Listing Cache — `Clip_Adapter.py`

Directory structures are cached in:

```python
self._dir_cache
```

This eliminates repeated operating-system-level `os.listdir()` calls during sequence construction.

## 3. RAM-Efficient `uint8` Image Caching

Raw crops are pre-resized to:

```text
256 × 256
```

and stored in memory as `np.uint8` arrays. This reduces the memory footprint by approximately **4×** compared with storing the same image data as 32-bit floating-point values.

## 4. Dimension Collapsing for GPU Parallelism

Player tracklets are represented as:

```text
(B, N, T, C, H, W)
```

where the dimensions correspond to batch, number of players, temporal sequence, channels, height, and width.

The implementation folds the first two dimensions into:

```text
(B × N, T, C, H, W)
```

This allows the CNN-LSTM pipeline to process player tracklets in a single vectorized pass.

## 5. Zero-Redundancy Multi-GPU Synchronization — `DDP_base_trainer.py`

The distributed trainer:

- Computes per-class counters and validation metrics across all ranks using `dist.all_reduce` and `dist.all_gather_object`.
- Removes the DDP wrapper before checkpoint serialization by unwrapping `model.module`.

This ensures that saved checkpoints can be loaded cleanly in standard single-GPU environments.

---

# 🖥️ HPC Infrastructure & Slurm Setup

The project includes batch scripts optimized for compute nodes using shared network storage (**NFS**).

## Local Node Caching

Datasets are automatically unpacked into:

```text
$LOCAL_TMP
```

which maps to local temporary storage such as:

```text
/tmp/$USER
```

This helps avoid NFS I/O bottlenecks and prevents unnecessary pressure on network-drive quotas.

## Pre-cached Model Weights

`TORCH_HOME` is configured to use local scratch storage. Pretrained ResNet-50 weights are downloaded before launching distributed workers to prevent multiple processes from racing to download the same weights.

## Low-Overhead GPU Profiling

An asynchronous background monitoring process using `pynvml` records GPU compute and memory utilization to CSV every **10 seconds**.

---

# 🚀 Getting Started

## 1. Environment Setup

Create and activate the Conda environment:

```bash
conda create -n vision_env python=3.10 -y
conda activate vision_env
```

Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:

```bash
pip install pillow numpy pyyaml scikit-learn pynvml kaggle
```

---

## 2. Dataset Preparation

Ensure the Volleyball dataset follows this structure:

```text
volleyball_data/
├── videos/
│   ├── 0/
│   │   ├── annotations.txt
│   │   ├── 13286/
│   │   │   ├── 13276.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
└── volleyball_tracking_annotation/
    ├── 0/
    │   └── 13286/
    │       └── 13286.txt
    └── ...
```

---

## 3. Running Experiments Locally

### Baseline 1 — Whole-Image ResNet-50

```bash
python runs/baseline1/Run_Baseline1.py
```

### Baseline 3 — Person Multi-Task & Attention Group Model

Train the person model:

```bash
python runs/baseline3/Run_Baseline3_person_model.py
```

Then train the group model:

```bash
python runs/baseline3/Run_Baseline3_group_model.py
```

### Baseline 4 — Frame-Level BiLSTM

```bash
python runs/baseline4/Run_Baseline4.py
```

### Baseline 5 — Player Tracklet BiLSTM + Group Spatial Pooling

For distributed multi-GPU execution:

```bash
torchrun --standalone --nproc_per_node=2 \
    runs/baseline5/Run_GroupBaseline5_ddp.py
```

---

# 🖥️ Submitting Slurm Jobs

## Single-GPU Sequential Run

```bash
sbatch run_volleyball.sh
```

## Distributed Multi-GPU Run — DDP via `torchrun`

```bash
sbatch run_volleyball_ddp.sh
```

---

# 📜 Citations

If you use this implementation or build upon the referenced work, please cite the original publications.

## Original CVPR 2016 Paper

```bibtex
@inproceedings{ibrahim2016hierarchical,
  title={A hierarchical deep temporal model for group activity recognition},
  author={Ibrahim, Mostafa S and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1971--1980},
  year={2016}
}
```

## Hierarchical Relational Networks

```bibtex
@article{ibrahim2018hierarchical,
  title={Hierarchical relational networks for group activity recognition and retrieval},
  author={Ibrahim, Mostafa S and Mori, Greg},
  journal={arXiv preprint arXiv:1811.09319},
  year={2018}
}
```

---

# 📌 Summary of the Repository

This project provides a progressive implementation of hierarchical group-activity recognition for volleyball scenes. It moves from a **single-frame whole-scene CNN baseline** toward increasingly structured models that incorporate:

```text
Whole Scene
    ↓
Individual Players
    ↓
Temporal Player Dynamics
    ↓
Spatial / Court-Aware Pooling
    ↓
Group Temporal Dynamics
    ↓
Group Activity Recognition
```

The repository therefore serves both as an experimental benchmark and as a modular PyTorch implementation for studying how **individual actions, temporal dynamics, spatial relationships, and team-level context** contribute to group activity recognition.
