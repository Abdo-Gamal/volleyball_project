# Group Activity Recognition

A modern implementation of the CVPR 2016 paper: [*A Hierarchical Deep Temporal Model for Group Activity Recognition*](https://arxiv.org/abs/1607.02643) by Mostafa S. Ibrahim et al.

This project tackles the complex task of group activity recognition by inferring the temporal dynamics of a whole activity based on the individual dynamics of the people involved. It employs a two-stage hierarchical architecture (LSTM/GRU) to capture both individual-level actions and group-level temporal interactions.

## 📑 Table of Contents

* [Abstract & Key Updates](#abstract--key-updates)

* [Dataset Overview](#dataset-overview)

* [Model Architecture](#model-architecture)

* [Ablation Study & Baselines](#ablation-study--baselines)

* [Performance & Results](#performance--results)

* [Interesting Observations](#interesting-observations)

* [Installation & Usage](#installation--usage)

* [Citation](#citation)

## 🚀 Abstract & Key Updates

**Abstract:** In group activity recognition, the temporal dynamics of the whole activity can be inferred based on the dynamics of the individual people representing the activity. We present a 2-stage deep temporal model where an initial sequence model (e.g., LSTM/GRU) is designed to represent action dynamics of individual people, and a secondary sequence model aggregates this person-level information for comprehensive scene-level activity understanding.

**Key Updates in this Implementation:**

* **Modern Backbones:** Replaced the original AlexNet with **ResNet-50** and **ResNet-34** for superior feature extraction.

* **End-to-End Training:** Implemented a unified end-to-end version (Baseline 9) using GRUs instead of LSTMs to reduce complexity and mitigate overfitting.

* **Higher Performance:** Achieved higher accuracy and F1 scores across *every* model baseline compared to the original paper.

* **Modern Framework:** Full implementation in Python/PyTorch (original was in Caffe).

* **Extensive Ablation Studies:** Detailed analysis of model components to understand the effect of spatial pooling and temporal modeling.

## 📊 Dataset Overview

We utilize the expanded **Volleyball Dataset**, collected from publicly available YouTube videos. The dataset contains **4,830 frames** handpicked from **55 videos**, featuring 2 levels of annotation: **9 player action labels** and **8 team activity labels**.

### Annotations Visualized


*Figure: A frame labeled as "Left Spike" with bounding boxes around players demonstrating team activity annotations.*


*Figure: For each visible player, an individual action label is annotated (e.g., Spiking, Blocking, Standing).*

### Dataset Statistics

**Train-Test Split:**

* **Training Set:** 3,493 frames (Train Videos: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54)

* **Testing Set:** 1,337 frames (Test/Val Videos: 0, 2, 4, 5, 8, 9, 11, 12, 14, 17, 19, 20, 21, 24-30, 33-35, 37, 43-47, 49, 51)

* *Note: The train-test split is performed at the video level to ensure convincing model evaluation.*

| Group Activity Class | Instances |  | Player Action Class | Instances | 
| ----- | ----- | ----- | ----- | ----- | 
| Right set | 644 |  | Waiting | 3,601 | 
| Right spike | 623 |  | Setting | 1,332 | 
| Right pass | 801 |  | Digging | 2,333 | 
| Right winpoint | 295 |  | Falling | 1,241 | 
| Left winpoint | 367 |  | Spiking | 1,216 | 
| Left pass | 826 |  | Blocking | 2,458 | 
| Left spike | 642 |  | Jumping | 341 | 
| Left set | 633 |  | Moving | 5,121 | 
|  |  |  | Standing | 38,696 | 

## 🧠 Model Architecture

The core of this project is a Hierarchical Deep Temporal Model.


*Figure: Given tracklets of K-players, we feed each tracklet into a CNN, followed by a person-level LSTM to represent individual actions. Features are then pooled and fed into a secondary team-level LSTM to identify the whole team's activity.*

### 1. Player Activity Temporal Classifier

* **Spatial Backbone:** A pretrained ResNet-50/ResNet-34 extracts spatial features from image crops of individual players.

* **Temporal Modeling:** An LSTM/GRU processes the sequence of features (tracklets) across multiple frames to capture individual player dynamics.

### 2. Group Activity Temporal Classifier (Team Pooling)

* **Spatial Pooling:** Instead of pooling all people blindly, players are grouped into two teams (e.g., players 1–6 for Team 1, players 7–12 for Team 2).

* **Hierarchical Integration:** Adaptive max-pooling aggregates features within each team independently. Features from both teams are then concatenated.

* **Scene Level Modeling:** A second sequence model (LSTM/GRU) processes these concatenated team features over time to classify the final group activity.


*Figure: Previous basic models dropped spatial information. In our updated model (Baseline 8/9), 2-group pooling captures the spatial arrangements of opposing teams.*

## 🔬 Ablation Study & Baselines

To thoroughly assess the impact of various components (feature extraction, temporal modeling, pooling strategies), we conducted extensive experimentation by systematically adding or removing features.

* **Baseline B1 (Image Classification):** A simple baseline. Fine-tunes a ResNet-50 image classifier over the 8 scene classes using only a single frame (the middle frame) from a video clip. No temporal information.

* **Baseline B3 (Fine-tuned Person Classification):**

  * **Train:** Fine-tune a classifier over 9 actions using cropped persons.

  * **Inference:** Extract 2048 features per person crop, max pool all features to create an image representation, and train a Neural Network over the 8 group classes.

* **Baseline B4 (Temporal Model with Image Features):** Uses the B1 classifier to extract sequence representations for clips (9 frames per clip). Trains an LSTM on these sequences without explicit individual feature extraction.

* **Baseline B5 (Temporal on Crops - LSTM on Player Level):** Extracts features per person temporally (LSTM on player level). The last hidden states represent each player. These are max-pooled for all 12 players, followed by a standard NN classifier for the scene (no scene-level temporal info).

* **Baseline B6 (Two-stage Model without LSTM 1):** Similar to B3, but applies an LSTM at the image/scene level to sequences of individual pooled features.

* **Baseline B7 (Two-stage Model without LSTM 2 - Full Model V1):** Trains an LSTM on crop-level data (9 steps per player). A single max-pooling operation is applied to all players in the frame, and a second LSTM (LSTM 2) is trained on the frame level.

* **Baseline B8 (Two-stage Hierarchical Model with Team Pooling):** Same as B7, but the scene representation is **not** a blind pool of all players. It pools Team 1 (6 players) and Team 2 (6 players) independently, then concatenates them. This preserves crucial spatial arrangements.

* **Baseline B9 (Unified Hierarchical End-to-End Model):** Integrates person-level and group-level losses into a single, unified end-to-end training pipeline with shared gradient flow. Uses **ResNet-34** and **GRU** to reduce complexity and overfitting.

## 📈 Performance & Results

Our modern implementation outperforms the original CVPR 2016 baselines significantly.

### My Scores (Accuracy and F1 Scores)

| Baseline Model | Accuracy | F1 Score | 
| ----- | ----- | ----- | 
| **Baseline 1** | 72.66% | 72.63% | 
| **Baseline 3** | 80.25% | 80.24% | 
| **Baseline 4** | 76.59% | 76.67% | 
| **Baseline 5** | 77.04% | 77.07% | 
| **Baseline 6** | 84.52% | 83.99% | 
| **Baseline 7** | 89.15% | 89.14% | 
| **Baseline 8** | 92.30% | 92.29% | 
| **Baseline 9 (End-to-End)** | **93.12%** | **93.11%** | 

*(For comparison, the original paper's top model achieved an accuracy of around 81.9% on the validation set).*

## 💡 Interesting Observations

**The Effect of Team Independent Pooling**

A key discovery in our ablation study was observed when transitioning from the individual level to the frame level.

In Baselines 5 and 6, when all 12 players from both teams were pooled together into a single representation, the model lost valuable geometric and spatial information. This resulted in frequent confusion between mirrored actions, such as:

* Right winpoint vs. Left winpoint

* Right pass vs. Left pass

* Right set vs. Left set

When teams are grouped and processed individually before concatenation (as introduced in **Baseline 8** and perfected in **Baseline 9**), the player position information is retained. This careful handling of spatial arrangements drastically reduces confusion and boosts model accuracy to >92%.
