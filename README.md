# A Hierarchical Deep Temporal Model for Group Activity Recognition

**Based on the CVPR 2016 paper by Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, and Greg Mori.**

---

## 📖 Introduction & Problem Description

When watching a sports game, how do we know what play a team is executing? We usually figure out the **whole team's activity** by observing the **actions of individual players** over time. 

This project solves the problem of **Group Activity Recognition** by building a deep learning model that mimics this exact human logic. We introduce a **2-stage deep temporal model** powered by Long Short-Term Memory (LSTM) networks:
1. **Stage 1 (Person-Level):** An LSTM model is designed to represent the dynamic actions of individual people in a sequence.
2. **Stage 2 (Group-Level):** A second LSTM model aggregates this individual person-level information to understand the entire scene's activity.

![High-level hierarchical model](images/fig1.png)

*Figure 1: A high-level look at our hierarchical model. Each person's movement is tracked individually to capture their dynamics, and these models are then integrated into a higher-level network to recognize the full scene's activity.*

---

## 🧠 How the Architecture Works

Our model is designed to be highly intuitive. Here is a simple breakdown of how data flows through the architecture:

### 1. Capturing Dynamics
![Group vs Person Dynamics](images/fig2.png)

*Figure 2: We separate the problem into two distinct parts—understanding individual person dynamics (like a player setting or standing) and understanding the overarching group dynamics.*[cite: 2]

### 2. The Detailed Pipeline
![Detailed Model Pipeline](images/fig3.png)

*Figure 3: The step-by-step process. First, we feed individual player tracklets into a Convolutional Neural Network (CNN), followed by a Person-level LSTM (LSTM 1) to understand what each player is doing. We then pool everyone's features together and feed them into a second Group-level LSTM (LSTM 2) to classify the final team activity (e.g., identifying a "Right Set").*

### 3. Adding Spatial Awareness
![Spatial Pooling](images/fig4.png)

*Figure 4: Where players are located on the court matters! While basic models drop spatial information, our updated model uses a 2-group pooling strategy to capture the spatial arrangements and formations of the players.*

---

## 🏐 The Expanded Volleyball Dataset

To train and evaluate our model, we collected a massive new **Volleyball Dataset** using publicly available YouTube videos. This expanded version is **3 times larger** than our original CVPR submission!

* **Total Videos:** 55 videos (8 videos are 1920x1080 resolution, the rest are 1280x720).
* **Total Frames:** 4,830 handpicked annotated frames (3,493 for training, 1,337 for testing).
* **Train/Test Split:** Performed strictly at the *video level* (not frame level) to ensure the model's evaluation is convincing and prevents data leakage.
    * **Train Videos:** 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54
    * **Validation Videos:** 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51
    * **Test Videos:** 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47

### 🏷️ Labels & Classes
We labeled the data at both the team and individual levels:
* **8 Group Activity Classes:** Right set (644), Right spike (623), Right pass (801), Right winpoint (295), Left winpoint (367), Left pass (826), Left spike (642), Left set (633).
* **9 Individual Action Classes:** Waiting (3601), Setting (1332), Digging (2333), Falling (1241), Spiking (1216), Blocking (2458), Jumping (341), Moving (5121), Standing (38696).

### 📁 Directory Structure & Format
* The dataset is organized into folders by unique Video IDs (`0` to `54`).
* Inside each video directory, frames are grouped by the target frame ID (e.g., `volleyball/39/29885`).
* Because scenes change rapidly in volleyball, each frame directory contains a tight window of **41 images** (20 images before the target frame, the target frame itself, and 20 after). *Note: In our work, we specifically used 5 frames before and 4 frames after.*
* **Annotations:** Stored in an `annotations.txt` file per video. 
    * Format: `{Frame ID} {Frame Activity Class} {Player Annotation 1} {Player Annotation 2} ...`
    * Player Annotation Format: `{Action Class} X Y W H` (tight bounding box).

### 🔗 Downloads & Updates
* **[Main Dataset Download Link](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS)** *(Combined Google Drive folder).*

---

## 📊 Original Results & Performance

Our Two-stage Hierarchical Model was rigorously tested and significantly outperforms standard baselines.

![Performance Table](images/table1.png)

*Table 1: A comparison of team activity recognition performance on the Volleyball Dataset. Our Two-stage Hierarchical Model achieves a leading accuracy of **81.9%**, successfully outperforming basic image/person classification methods and Improved Dense Trajectories (IDTF) approaches.*[cite: 5]

---

## 🛠️ Implemented Baselines & Progression

To benchmark and understand the contribution of each temporal and spatial component, we implement a step-by-step hierarchy of baselines moving from static individual classifiers up to two-stage spatial-temporal models.

### Baseline Descriptions

* **Baseline 1 (B1-Tuned — Whole Image Classifier):**
  A standard scene-classification baseline. Instead of older backbones (e.g., AlexNet), a modern CNN backbone (ResNet-50) is fine-tuned directly on the center frame of each clip (or the 9-frame window) to classify the 8 group activities without explicit person modeling.
* **Baseline 3 (B3 — Static Person Feature Pooling):**
  * **Step A (Action Pretraining):** Fine-tune a CNN classifier on individual cropped players across the 9 action classes.
  * **Step B (Feature Extraction & Pooling):** Extract high-dimensional feature vectors (e.g., 2048-D) for every player bounding box in an image and max-pool them into a single frame-level representation.
  * **Step C (Classification):** Train a feedforward network on top of the pooled features to classify the 8 group activities (no temporal modeling).
* **Baseline 4 (B4 — Frame-Level Temporal Modeling):**
  Introduces temporal modeling on the global scene level. Sequences of whole-image feature representations across 9 frames are extracted using the B1 backbone and passed into an LSTM to classify group activity over time.
* **Baseline 5 (B5 — Player-Level Temporal Modeling):**
  Introduces temporal modeling on individual players. LSTMs operate over player tracklet sequences across 9 frames. The final hidden states of all players in a frame are max-pooled into an overall scene vector and classified using a feedforward classifier (temporal person dynamics without a second group LSTM).
* **Baseline 6 (B6 — Pooled Spatial Features into Temporal LSTM):**
  Combines B3 and B4. Person crops are extracted and max-pooled for each individual frame across a 9-frame sequence, and the resulting sequence of pooled frame embeddings is fed into a frame-level LSTM.
* **Baseline 7 (B7 — Hierarchical Two-Stage Model with Global Pooling):**
  The standard full two-stage model:
  1. Player-level LSTMs process cropped player tracklets over time.
  2. Features across all players are max-pooled per time step.
  3. A group-level LSTM processes the pooled sequence to capture scene dynamics.
* **Baseline 8 (B8 — Hierarchical Two-Stage Model with Team/Court Pooling):**
  Extends B7 by preserving court spatial arrangements. Instead of pooling across all 12 players globally, players are partitioned by court side (Team 1 vs. Team 2). Features are pooled per team and concatenated `[Team 1; Team 2]` before entering the group-level LSTM.

---

## 📈 Experimental Results

### Our Implementation Performance

| Baseline Model | Input / Pipeline Representation | Paper Accuracy  | Our Result |
| :--- | :--- | :--- | :--- |
| **B1-Tuned** | Middle Frame / ResNet-50 Scene Classifier | 66.7% | **70.0% Acc** |
| **B3** | Player Crops → CNN Features → Max Pool → Classifier | 68.1% | **83.0% F1** |
| **B4** | 9-Frame Image Sequence → Frame LSTM | 63.1% | **82.2% F1** |
| **B5** | 9-Frame Player Crops → Player LSTM → Max Pool | 67.6% | **87.0% F1** |
| **B6** | Frame-by-Frame Person Pool → Sequence LSTM | 74.7%[cite: 5] | *In Progress* |
| **B7** | Two-Stage Hierarchical Model (Global Pooling) | 80.2%  | *In Progress* |
| **B8** | Two-Stage Hierarchical Model (2-Team Spatial Pooling) | 81.9% | *In Progress* |


---
*Note: The first version of this work was accepted at CVPR 2016. The provided dataset is the expanded version. Please use and compare against this version.*