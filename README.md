Group Activity Recognition: A Hierarchical Deep Temporal Model

A modern implementation of the CVPR 2016 paper: A Hierarchical Deep Temporal Model for Group Activity Recognition. This project tackles the complex challenge of understanding collective group activities by analyzing the individual actions of people over time.

📊 1. Data Understanding & Dataset Overview

In practice, the first step in any robust machine learning pipeline is to fully understand the data: checking quality and quantity, visualizing samples, and noting down biases.

Dataset Details

We utilize the Volleyball Dataset, consisting of publicly available YouTube videos. This dataset is excellent for this problem because it features structured team interactions.

Size: ~60GB dataset. Download Link Here (Or start with a sample of 2 videos, each with 2 clips).

Annotations: This dataset has 2 distinct levels of annotation:

9 Person Actions (Individual Level): Waiting, Setting, Digging, Falling, Spiking, Blocking, Jumping, Moving, Standing.

8 Scene Classes (Group Activity Level): Right set, Right spike, Right pass, Right winpoint, Left winpoint, Left pass, Left spike, Left set.

Example Annotations:

Figure: A frame labeled as "Left Spike," with bounding boxes around each player, demonstrating individual and team activity annotations.

🔬 2. Ablation Study: Understanding System Components

An Ablation Study is a method used to assess the impact of various components of a system on its overall performance by experimenting by removing them one by one.

Example: Assume your system consists of 4 enhancement features (A, B, C, D).

A and B are extra losses, C is a 2nd LSTM layer, D is a complex backbone.

You run experiments like ABC, ACD, ABD. Each combination tells you the effect of the single component that was removed.

In a standard classifier: This could mean removing certain layers, disabling data augmentation, or using different feature extraction methods.

In a self-driving car: It's akin to removing specific sensors to see the effect on motion prediction.

🏗️ 3. Model Baselines (B1 to B8)

To prove the effectiveness of the final Hierarchical Model, we build and test several baselines. Each experiment teaches us something vital (e.g., naive image classification doesn't help much, but temporal information significantly boosts results).

Baseline B1-tuned (Image Classification)

Concept: Don't try anything that doesn't fine-tune well.

Implementation: For each clip, use the middle image only (or feel free to use 5 before and 4 after). Fine-tune an image classifier (e.g., ResNet50, upgrading from the original paper's AlexNet) over the 8 scene classes. Compute the results. This is your first model.

Baseline B3 (Fine-tuned Person Classification)

Train (A): Fine-tune an image classifier over 9 individual actions. The input is a cropped person.

Inference (B): For an image, get all the person crops. Extract features for each crop (e.g., 2048 features). Apply Max Pool across all features to create a single image representation.

Train (C): Do standard Neural Network (NN) training on these pooled features over the 8 group classes.

Baseline B4 (Temporal Model with Image Features)

Implementation #1: Use the classifier from B1-tuned to extract a representation per clip. Using 9 frames per image, you create a sequence of 9 steps for each clip. Train an LSTM on these sequences.

Implementation #2: Alternatively, extend the classifier network directly with an LSTM layer followed by classification. This avoids explicit feature extraction steps.

Baseline B5 (Temporal Model with Person Features)

Concept: Temporal on crops (LSTM on a player level).

Implementation: Build a representation per person. Represent each clip with the last hidden state of the LSTM. Max pool all players' representations (up to 12 per image). Then, apply the NN classifier exactly like in B3 (on images). Note: The features classifier here has no temporal info at the group level.

Baseline B6 (Two-stage Model without LSTM 1)

Implementation: Follow B3 steps A and B. However, for step B, extract representations for each clip of 9 frames. For step C, apply an LSTM on the sequences from step B.

Result: This is a model where the LSTM is applied on the image (scene) level only, not on the individual players.

Baseline B7 (Two-stage Model without LSTM 2 / Full Model V1)

Implementation:

Train LSTM on the crops level (LSTM on a player).

Extract clips: a sequence of 9 steps per player.

For each frame, properly max pool its players into one representation.

Train LSTM 2 on the frame level.

Baseline B8 (Two-stage Hierarchical Model - Full Model V2)

Concept: Retaining Spatial & Team Information.

Implementation: Same as B7, but the scene representation is not a pool of all players mixed together.

X = Pool Team 1 (6 players)

Y = Pool Team 2 (6 players)

Scene Representation = Concatenation of X and Y.

Using one representation per team reduces confusion (e.g., Left vs. Right) and significantly enhances results.


Figure 1: High-level architecture. Each person is modeled using a temporal model that captures their dynamics, integrated into a higher-level model for scene activity.


Figure 2: Detailed model view showing individual LSTMs feeding into a group LSTM.


Figure 3: Highlighting the 2-group pooling (Baseline B8) to capture the spatial arrangements of players.

📈 4. Results & Performance

As shown in the ablation study results below, adding temporal models (LSTM) and properly pooling teams (B8) incrementally improves the model's accuracy.


Figure: The original paper's baseline scores demonstrating the impact of each component, culminating in 81.9% accuracy for the Two-Stage Hierarchical Model.

🚀 5. Implementation Challenges & Future Work

The End-to-End Challenge

A major implementation challenge (aside from memory constraints) is building the 2-stage model as a single end-to-end network:

Input: 12 cropped users (ensuring each person is tracked consistently).

Architecture: Need 3 modules -> LSTM 1 (Person) -> Proper Team Pooling -> LSTM 2 (Scene).

Optimization: Calculating simultaneous losses on person action and scene classification to share the gradient flow.

Expanding the Network

Graph Neural Networks (GNN): You can extend this network by applying GNNs. (See the 2018 paper: Hierarchical relational networks for group activity recognition and retrieval).

Literature Review: The field is vast. Studying similar techniques applied to different domains, such as motion prediction for cars in self-driving, can yield significant insights into how these temporal and spatial ideas advance.
