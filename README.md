# A Hierarchical Deep Temporal Model for Group Activity Recognition

**Based on the CVPR 2016 paper by Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, and Greg Mori.**

---

## 📖 Introduction & Problem Description

When watching a sports game, how do we know what play a team is executing? We usually figure out the **whole team's activity** by observing the **actions of individual players** over time. 

This project solves the problem of **Group Activity Recognition** by building a deep learning model that mimics this exact human logic. We introduce a **2-stage deep temporal model** powered by Long Short-Term Memory (LSTM) networks:
1. **Stage 1 (Person-Level):** An LSTM model is designed to represent the dynamic actions of individual people in a sequence.
2. **Stage 2 (Group-Level):** A second LSTM model aggregates this individual person-level information to understand the entire scene's activity.

<img src="images/fig1.png" alt="High-level hierarchical model" width="800">
*Figure 1: A high-level look at our hierarchical model. Each person's movement is tracked individually to capture their dynamics, and these models are then integrated into a higher-level network to recognize the full scene's activity.*

---

## 🧠 How the Architecture Works

Our model is designed to be highly intuitive. Here is a simple breakdown of how data flows through the architecture:

### 1. Capturing Dynamics
<img src="images/fig2.png" alt="Group vs Person Dynamics" width="800">
*Figure 2: We separate the problem into two distinct parts—understanding individual person dynamics (like a player setting or standing) and understanding the overarching group dynamics.*

### 2. The Detailed Pipeline
<img src="images/fig3.png" alt="Detailed Model Pipeline" width="800">
*Figure 3: The step-by-step process. First, we feed individual player tracklets into a Convolutional Neural Network (CNN), followed by a Person-level LSTM (LSTM 1) to understand what each player is doing. We then pool everyone's features together and feed them into a second Group-level LSTM (LSTM 2) to classify the final team activity (e.g., identifying a "Right Set").*

### 3. Adding Spatial Awareness
<img src="images/fig4.png" alt="Spatial Pooling" width="800">
*Figure 4: Where players are located on the court matters! While basic models drop spatial information, our updated model uses a 2-group pooling strategy to capture the spatial arrangements and formations of the players.*

---

## 🏐 The Expanded Volleyball Dataset

To train and evaluate our model, we collected a massive new **Volleyball Dataset** using publicly available YouTube videos. This expanded version is **3 times larger** than the original CVPR submission!

* **Total Videos:** 55 videos (8 videos are 1920x1080 resolution, the rest are 1280x720).
* **Total Frames:** 4,830 handpicked annotated frames (3,493 for training, 1,337 for testing).
* **Train/Test Split:** Performed strictly at the *video level* (not frame level) to ensure the model's evaluation is convincing and prevents data leakage.

### 🏷️ Labels & Classes
We labeled the data at both the team and individual levels:
* **8 Group Activity Classes:** Right set (644), Right spike (623), Right pass (801), Right winpoint (295), Left winpoint (367), Left pass (826), Left spike (642), Left set (633).
* **9 Individual Action Classes:** Waiting (3601), Setting (1332), Digging (2333), Falling (1241), Spiking (1216), Blocking (2458), Jumping (341), Moving (5121), Standing (38696).

### 🔗 Downloads & Updates
* **[Main Dataset Download Link](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS)** *(Combined Google Drive folder).*
*(Note: Refer to the official dataset source for additional manual annotations and detector files if not using our automated DataLoader).*

---

## 🛠️ Implemented Baselines & Progression

To benchmark and understand the contribution of each temporal and spatial component, we implement a step-by-step hierarchy of baselines moving from static individual classifiers up to two-stage spatial-temporal models.

### Baseline Descriptions

* **Baseline 1 (B1-Tuned — Whole Image Classifier):** A standard scene-classification baseline using a modern CNN backbone (ResNet-50) fine-tuned directly on the center frame of each clip to classify the 8 group activities without explicit person modeling.
* **Baseline 3 (B3 — Static Person Feature Pooling):**
  * **Step A:** Fine-tune a CNN classifier on individual cropped players.
  * **Step B:** Extract features for every player in an image and max-pool them into a single frame-level representation.
  * **Step C:** Train a feedforward network on the pooled features to classify the group activity.
* **Baseline 4 (B4 — Frame-Level Temporal Modeling):** Introduces temporal modeling on the global scene level. Sequences of whole-image feature representations are extracted and passed into an LSTM.
* **Baseline 5 (B5 — Player-Level Temporal Modeling):** Introduces temporal modeling on individual players. LSTMs operate over player tracklet sequences. The final hidden states of all players in a frame are max-pooled into an overall scene vector and classified.
* **Baseline 6 (B6 — Pooled Spatial Features into Temporal LSTM):** Combines B3 and B4. Person crops are extracted and max-pooled for each individual frame, and the sequence of pooled embeddings is fed into a frame-level LSTM.
* **Baseline 7 (B7 — Hierarchical Two-Stage Model with Global Pooling):** The standard full two-stage model (Player-level LSTMs → Max Pooling → Group-level LSTM).
* **Baseline 8 (B8 — Hierarchical Two-Stage Model with Team/Court Pooling):** Extends B7 by preserving court spatial arrangements. Players are partitioned by court side (Team 1 vs. Team 2), pooled separately, and concatenated.

---

## 📈 Experimental Results

### Our Implementation Performance

| Baseline Model | Input / Pipeline Representation | Original Paper Accuracy | Our Result |
| :--- | :--- | :--- | :--- |
| **B1-Tuned** | Middle Frame / ResNet-50 Scene Classifier | 66.7% | **70.0% Acc** |
| **B3** | Player Crops → CNN Features → Max Pool → Classifier | 68.1% | **83.0% F1** |
| **B4** | 9-Frame Image Sequence → Frame LSTM | 63.1% | **82.2% F1** |
| **B5** | 9-Frame Player Crops → Player LSTM → Max Pool | 67.6% | **87.0% F1** |
| **B6** | Frame-by-Frame Person Pool → Sequence LSTM | 74.7% | *In Progress* |
| **B7** | Two-Stage Hierarchical Model (Global Pooling) | 80.2% | *In Progress* |
| **B8** | Two-Stage Hierarchical Model (2-Team Spatial Pooling) | 81.9% | *In Progress* |

---

## 🏗️ Code Architecture & Design Patterns

The repository is structured with a strict separation of concerns, heavily utilizing Object-Oriented Programming (OOP) design patterns to eliminate boilerplate and ensure reproducibility.

### Modular Project Structure
* **`configs/`**: YAML files managing hyperparameters for each baseline and Distributed Data Parallel (DDP) runs.
* **`dataset/`**: Contains raw data loaders and an `adapters/` module for formatting specific inputs (frames, person crops, or group tracks).
* **`models/`**: Separates CNN backbones (`resnet50.py`) from baseline-specific architectures (`baseline1.py`, `baseline5.py`).
* **`trainers/`**: Houses the training loops, including a flexible `base_trainer.py` and a `DDP_base_trainer.py` engineered for multi-GPU environments.
* **`runs/`**: Executable Python scripts linking configs, models, and trainers for specific experiments.

### 📐 The Template Method Pattern
To manage the complexity of multiple data inputs and transformation pipelines without rewriting code, we rely on the **Template Method Pattern**. This defines the skeleton of an algorithm in a base class while letting subclasses override specific steps.

**1. Data Adapters (`adapters/base_adapter.py`)**
* **Problem:** Loading a full frame, a single person crop, or a spatial group required duplicating basic Dataset logic.
* **Solution:** `BaseAdapter` defines the immutable skeleton (`__init__`, shared `open_image()` cache, and `__getitem__`). Subclasses only override `build_index()` and `load_sample()`.
* **Advantage:** Guarantees uniformity, drastically reduces code duplication, and centralizes image caching.

**2. Transformation Pipelines (`transforms.py`)**
* **Problem:** Hardcoding transforms for multiple baselines leads to dangerous copy-pasting, especially for normalization and resizing.
* **Solution:** `BaseTransform` locks in the immutable pipeline order: `Resize` → `Crop` → `[Augmentations]` → `ToTensor` → `Normalize`. 
* **Advantage:** Baseline-specific transform classes only override `get_train_augmentations()` (e.g., injecting `RandomRotation`). The `train()` and `val()` methods remain consistent, ensuring validation sets are never accidentally augmented.

### 🔄 Customizable Training Hooks (`trainers/base_trainer.py`)
Instead of writing a new training loop for every baseline, `BaseTrainer` handles the core epoch iteration, Automatic Mixed Precision (AMP) scaling, learning rate scheduling, and checkpoint saving. Subclasses override specific hooks like `compute_loss()` or `move_input()` to dictate how complex nested data flows into the model.

---

## 🚀 HPC & Distributed Training (Slurm)

Training hierarchical deep temporal models requires significant compute. The repository includes optimized Slurm batch scripts engineered specifically for High-Performance Computing (HPC) clusters.

* **Smart Local Caching:** To avoid I/O bottlenecks and protect Network File System (NFS) storage quotas, the training scripts dynamically download and cache the dataset directly to the compute node's local `/tmp` storage.
* **Distributed Data Parallel (DDP):** Baselines requiring heavy temporal unrolling utilize PyTorch DDP scripts (e.g., `runs/baseline5/Run_Baseline5_ddp.py`) for efficient multi-GPU scaling.
* **Background GPU Profiling:** Slurm scripts include a detached `pynvml` Python process that continuously logs GPU utilization, memory usage, and timestamps to a CSV file during training, safely bypassing standard `nvidia-smi` restrictions on cluster nodes.

### Running a Distributed Baseline
To submit a multi-GPU baseline job using the provided Slurm scripts:

```bash
sbatch run_volleyball_ddp.sh