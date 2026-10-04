# Group Activity Recognition

## A Hierarchical Deep Temporal Model for Group Activity Recognition

```{=html}
<p align="center">
```
`<img src="docs/images/hierarchical_model.png" width="850">`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<b>`{=html}Understanding group activities from individual player
actions, temporal dynamics, and team-level interactions.`</b>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## Table of Contents

-   [1. Overview](#1-overview)
-   [2. Problem Definition](#2-problem-definition)
-   [3. Why Group Activity Recognition Is
    Difficult](#3-why-group-activity-recognition-is-difficult)
-   [4. Hierarchical View of the
    Problem](#4-hierarchical-view-of-the-problem)
-   [5. Volleyball Dataset](#5-volleyball-dataset)
-   [6. Dataset Statistics](#6-dataset-statistics)
-   [7. Group Activity Classes](#7-group-activity-classes)
-   [8. Individual Player Action
    Classes](#8-individual-player-action-classes)
-   [9. Dataset Organization](#9-dataset-organization)
-   [10. Annotation Format](#10-annotation-format)
-   [11. Temporal Window](#11-temporal-window)
-   [12. Train / Validation / Test
    Split](#12-train--validation--test-split)
-   [13. Original Hierarchical
    Architecture](#13-original-hierarchical-architecture)
-   [14. Person-Level Temporal
    Modeling](#14-person-level-temporal-modeling)
-   [15. Group-Level Temporal
    Modeling](#15-group-level-temporal-modeling)
-   [16. Team-Aware Pooling](#16-team-aware-pooling)
-   [17. Baseline Experiments](#17-baseline-experiments)
-   [18. Baseline B1](#18-baseline-b1--image-classification)
-   [19. Baseline B3](#19-baseline-b3--fine-tuned-person-classification)
-   [20. Baseline
    B4](#20-baseline-b4--temporal-model-with-image-features)
-   [21. Baseline
    B5](#21-baseline-b5--temporal-model-with-person-features)
-   [22. Baseline B6](#22-baseline-b6--two-stage-model-without-lstm-1)
-   [23. Baseline B7](#23-baseline-b7--two-stage-model-without-lstm-2)
-   [24. Baseline B8](#24-baseline-b8--two-stage-hierarchical-model)
-   [25. Original Results](#25-original-results)
-   [26. Main Insight](#26-main-insight)
-   [27. Project Scope](#27-project-scope)
-   [28. References](#28-references)

------------------------------------------------------------------------

# 1. Overview

**Group Activity Recognition (GAR)** is the task of recognizing the
activity performed by a group of people in a video.

This project is based on the problem introduced in the CVPR 2016 paper:

> **A Hierarchical Deep Temporal Model for Group Activity Recognition**\
> Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat,
> Greg Mori\
> IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
> 2016.

The central idea is that a group activity can be understood by first
modeling the temporal behavior of individual people and then aggregating
these individual representations to understand the dynamics of the whole
group.

The problem can therefore be viewed as:

``` text
Video Frames
     ↓
Individual Players
     ↓
Player Actions
     ↓
Person-Level Temporal Dynamics
     ↓
Team / Group Representation
     ↓
Group-Level Temporal Dynamics
     ↓
Group Activity
```

The original work introduced a hierarchical architecture based on two
levels of temporal modeling:

1.  **Person-level LSTM** --- models the temporal dynamics of each
    player.
2.  **Group-level LSTM** --- models the temporal dynamics of the whole
    scene after aggregating the players.

An important extension is **two-group pooling**, where players from the
two volleyball teams are processed separately before their
representations are concatenated.

------------------------------------------------------------------------

# 2. Problem Definition

Traditional image classification attempts to answer questions such as:

``` text
"What is in this image?"
```

or:

``` text
"What action is happening?"
```

Group activity recognition is more complex because multiple people may
be performing different actions at the same time.

For example:

``` text
Player 1 → Standing
Player 2 → Moving
Player 3 → Jumping
Player 4 → Blocking
Player 5 → Setting
```

These are **individual actions**.

However, the complete scene may correspond to a higher-level activity
such as:

``` text
Left Spike
```

Therefore, the model must understand how the individual actions and
their temporal evolution combine into a group activity.

The task can be summarized as:

``` text
Individual Actions
        +
Temporal Dynamics
        +
Spatial / Team Structure
        ↓
Group Activity Recognition
```

------------------------------------------------------------------------

# 3. Why Group Activity Recognition Is Difficult

## 3.1 Multiple People

A volleyball scene contains multiple players, and each player may have a
different role.

The model therefore cannot rely on a single object or a single person's
appearance.

------------------------------------------------------------------------

## 3.2 Different Actions at the Same Time

Several actions can occur simultaneously.

For example:

``` text
Player A → Standing
Player B → Moving
Player C → Setting
Player D → Jumping
Player E → Blocking
```

The model must combine these actions to understand the activity of the
entire group.

------------------------------------------------------------------------

## 3.3 Temporal Dynamics

A single frame may not contain enough information to distinguish
activities.

A player may appear to be standing in one frame and then move, jump, or
block in subsequent frames.

For example:

``` text
t-2 → Standing
t-1 → Moving
t  → Jumping
t+1 → Blocking
```

The sequence is therefore more informative than an isolated image.

This motivates the use of recurrent temporal models such as LSTMs.

------------------------------------------------------------------------

## 3.4 Spatial and Team Information

Volleyball naturally contains two opposing teams located on different
sides of the court.

Therefore:

``` text
Left Spike  ≠ Right Spike
Left Set    ≠ Right Set
Left Pass   ≠ Right Pass
Left Winpoint ≠ Right Winpoint
```

If all players are pooled into one unordered representation, the model
may lose information about which players belong to which team.

This observation motivates **team-aware pooling**.

------------------------------------------------------------------------

# 4. Hierarchical View of the Problem

The complete problem can be viewed as a hierarchy:

``` text
                         GROUP ACTIVITY
                               ↑
                         Group Dynamics
                               ↑
                       Team Representation
                               ↑
                      Player Representations
                               ↑
                       Person Dynamics
                               ↑
                      Individual Actions
                               ↑
                           Video Frames
```

The motivation is that a group activity is not independent of the people
performing it.

Instead:

``` text
Video
  ↓
Player-level visual information
  ↓
Person-level temporal dynamics
  ↓
Team-level aggregation
  ↓
Group-level temporal dynamics
  ↓
Group activity
```

------------------------------------------------------------------------

# 5. Volleyball Dataset

The experiments use the **Volleyball Dataset**, collected from publicly
available YouTube volleyball videos.

The dataset was specifically designed for group activity recognition and
contains annotations at both the individual-player level and the
group-activity level.

### Dataset Summary

  Property                      Value
  --------------------------- -------
  Videos                           55
  Annotated frames              4,830
  Individual action classes         9
  Group activity classes            8

The dataset contains:

-   Video sequences.
-   Annotated target frames.
-   Player bounding boxes.
-   Individual action labels.
-   Group activity labels.
-   Temporal context around each annotated frame.

------------------------------------------------------------------------

## Example: Group Activity Annotation

```{=html}
<p align="center">
```
`<img src="docs/images/group_activity_annotation.png" width="750">`{=html}
```{=html}
</p>
```
**Figure 1.** Example frame labeled as **Left Spike**, with bounding
boxes around the players.

------------------------------------------------------------------------

## Example: Individual Player Annotations

```{=html}
<p align="center">
```
`<img src="docs/images/person_action_annotation.png" width="750">`{=html}
```{=html}
</p>
```
**Figure 2.** Example of individual player annotations such as
**Standing**, **Setting**, and **Blocking**.

------------------------------------------------------------------------

# 6. Dataset Statistics

The dataset contains two different annotation levels:

``` text
                    Volleyball Dataset
                           │
             ┌─────────────┴─────────────┐
             │                           │
      Individual Actions          Group Activities
             │                           │
          9 Classes                  8 Classes
```

The individual-level annotations describe what each visible player is
doing.

The group-level annotation describes what the overall team/group is
doing.

------------------------------------------------------------------------

# 7. Group Activity Classes

There are **8 group activity classes**.

  Group Activity     Instances
  ---------------- -----------
  Right Set                644
  Right Spike              623
  Right Pass               801
  Right Winpoint           295
  Left Winpoint            367
  Left Pass                826
  Left Spike               642
  Left Set                 633

The labels encode both the **type of activity** and the **side of the
court**.

``` text
Set
├── Left Set
└── Right Set

Spike
├── Left Spike
└── Right Spike

Pass
├── Left Pass
└── Right Pass

Winpoint
├── Left Winpoint
└── Right Winpoint
```

Therefore, the model must recognize not only *what* activity is
happening but also *which side* is performing it.

------------------------------------------------------------------------

# 8. Individual Player Action Classes

Each visible player can be assigned one of **9 action classes**.

  Player Action     Instances
  --------------- -----------
  Waiting               3,601
  Setting               1,332
  Digging               2,333
  Falling               1,241
  Spiking               1,216
  Blocking              2,458
  Jumping                 341
  Moving                5,121
  Standing             38,696

These labels provide fine-grained information about individual behavior.

For example:

``` text
Player 1 → Setting
Player 2 → Jumping
Player 3 → Blocking
Player 4 → Moving
Player 5 → Standing
```

These individual actions provide evidence for recognizing the
higher-level group activity.

------------------------------------------------------------------------

## Class Imbalance

The individual action labels are highly imbalanced.

For example:

``` text
Standing → 38,696
Jumping  →    341
```

This imbalance is important when training an individual action
classifier because a model can become biased toward frequent classes.

------------------------------------------------------------------------

# 9. Dataset Organization

The dataset contains **55 videos**, indexed from `0` to `54`.

Conceptually:

``` text
volleyball/
│
├── 0/
│   ├── annotations.txt
│   └── ...
│
├── 1/
│   ├── annotations.txt
│   └── ...
│
├── ...
│
└── 54/
    ├── annotations.txt
    └── ...
```

Each video directory contains annotated frames.

For example:

``` text
volleyball/39/29885/
```

means:

``` text
Video ID = 39
Frame ID = 29885
```

------------------------------------------------------------------------

# 10. Annotation Format

Each annotation line contains:

``` text
{Frame ID} {Frame Activity Class} {Player Annotation} {Player Annotation} ...
```

Each player annotation has the form:

``` text
{Action Class} X Y W H
```

where:

  Field          Meaning
  -------------- ---------------------------
  Action Class   Individual player action
  X              Bounding-box x-coordinate
  Y              Bounding-box y-coordinate
  W              Bounding-box width
  H              Bounding-box height

Therefore, each player annotation tells the model:

``` text
WHERE is the player?
        +
WHAT is the player doing?
```

------------------------------------------------------------------------

# 11. Temporal Window

Each annotated frame has temporal context.

For example, for target frame:

``` text
29885
```

the available temporal neighborhood is:

``` text
29865 ... 29884
29885
29886 ... 29905
```

This corresponds to:

``` text
20 frames before
        +
Target frame
        +
20 frames after
```

for a total of **41 frames**.

However, the original experiments use:

``` text
5 frames before
+
Target frame
+
4 frames after
```

giving a sequence of **9 frames**.

``` text
t-5  t-4  t-3  t-2  t-1   t   t+1  t+2  t+3
 │    │    │    │    │    │    │    │    │
 └────┴────┴────┴────┴────┴────┴────┴────┴──→ Time
```

The relatively short temporal window is important because volleyball
scenes can change rapidly.

------------------------------------------------------------------------

# 12. Train / Validation / Test Split

The dataset is split at the **video level**, rather than randomly
splitting frames.

This prevents highly similar frames from the same video from appearing
in both training and evaluation sets.

## Training Videos

``` text
1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23,
31, 32, 36, 38, 39, 40, 41, 42, 48, 50,
52, 53, 54
```

**24 videos**

## Validation Videos

``` text
0, 2, 8, 12, 17, 19, 24, 26, 27, 28,
30, 33, 46, 49, 51
```

**15 videos**

## Test Videos

``` text
4, 5, 9, 11, 14, 20, 21, 25, 29, 34,
35, 37, 43, 44, 45, 47
```

**16 videos**

------------------------------------------------------------------------

## Annotated Frame Split

The original dataset description reports:

  Split        Annotated Frames
  ---------- ------------------
  Training                3,493
  Testing                 1,337
  Total                   4,830

The important point is that the split is performed by **video**, not by
independently sampling frames.

------------------------------------------------------------------------

# 13. Original Hierarchical Architecture

The original paper proposes a two-stage hierarchical temporal model.

```{=html}
<p align="center">
```
`<img src="docs/images/original_hierarchical_architecture.png" width="850">`{=html}
```{=html}
</p>
```
**Figure 3.** High-level view of the hierarchical approach: individual
people are modeled using temporal models and their representations are
integrated into a higher-level group model.

The architecture contains two temporal levels:

``` text
                    VIDEO
                      │
             ┌────────┴────────┐
             │                 │
          Player 1          Player N
             │                 │
            CNN               CNN
             │                 │
       Person LSTM       Person LSTM
             │                 │
             └────────┬────────┘
                      │
                   Pooling
                      │
              Group Representation
                      │
                  Group LSTM
                      │
                      ↓
               Group Activity
```

------------------------------------------------------------------------

# 14. Person-Level Temporal Modeling

The first stage models the temporal dynamics of each individual player.

For each player, a sequence of crops is extracted from consecutive
frames.

``` text
Player Crop t-5
Player Crop t-4
Player Crop t-3
Player Crop t-2
Player Crop t-1
Player Crop t
Player Crop t+1
Player Crop t+2
Player Crop t+3
```

Each crop is passed through a CNN:

``` text
Player Crop
     ↓
    CNN
     ↓
Visual Feature
```

The sequence of visual features is then processed by an LSTM:

``` text
Feature t-5 ─┐
Feature t-4 ─┤
Feature t-3 ─┤
Feature t-2 ─┤
Feature t-1 ─┤
Feature t   ─┤
Feature t+1 ─┤──→ Person LSTM
Feature t+2 ─┤
Feature t+3 ─┘
                  ↓
          Player Representation
```

The Person LSTM therefore learns the temporal evolution of an individual
player.

------------------------------------------------------------------------

# 15. Group-Level Temporal Modeling

After obtaining a temporal representation for every player, the model
aggregates these representations.

``` text
Player 1 → Person Feature
Player 2 → Person Feature
Player 3 → Person Feature
...
Player N → Person Feature
```

The player features are pooled to form a group representation:

``` text
Player Features
      ↓
   Pooling
      ↓
Group Representation
```

A second LSTM then models the temporal evolution of the group:

``` text
Group Representation t-5
Group Representation t-4
Group Representation t-3
...
Group Representation t+3
              ↓
          Group LSTM
              ↓
       Group Activity
```

This produces the hierarchical structure:

``` text
Person Dynamics
      ↓
Player Representations
      ↓
Group Aggregation
      ↓
Group Dynamics
      ↓
Group Activity
```

------------------------------------------------------------------------

# 16. Team-Aware Pooling

A key observation in the original work is that pooling all players
together can remove important spatial information.

Consider the two teams:

``` text
Team 1
├── Player 1
├── Player 2
├── Player 3
├── Player 4
├── Player 5
└── Player 6

Team 2
├── Player 7
├── Player 8
├── Player 9
├── Player 10
├── Player 11
└── Player 12
```

### Global Pooling

If all players are pooled together:

``` text
12 Players
     ↓
Global Pooling
     ↓
One Group Representation
```

the model can lose information about the side of the court.

This can lead to confusion such as:

``` text
Left Spike  ↔ Right Spike
Left Set    ↔ Right Set
Left Pass   ↔ Right Pass
Left Winpoint ↔ Right Winpoint
```

------------------------------------------------------------------------

## Two-Group Pooling

The improved model pools the two teams independently:

``` text
                 Player Features
                       │
              ┌────────┴────────┐
              │                 │
           Team 1            Team 2
              │                 │
          Max Pool           Max Pool
              │                 │
        Team Feature 1   Team Feature 2
              │                 │
              └────────┬────────┘
                       │
                  Concatenation
                       │
                 Group Feature
                       │
                   Group LSTM
                       │
                       ↓
                Group Activity
```

Mathematically:

``` text
Team_1 = MaxPool(Player_1, ..., Player_6)

Team_2 = MaxPool(Player_7, ..., Player_12)

Group = Concatenate(Team_1, Team_2)
```

This preserves the distinction between the two teams before the final
group-level temporal model.

------------------------------------------------------------------------

# 17. Baseline Experiments

The baseline experiments progressively add components to understand
their contribution.

The progression is:

``` text
B1
 ↓
B3
 ↓
B4
 ↓
B5
 ↓
B6
 ↓
B7
 ↓
B8
```

The baselines investigate:

-   Single-frame recognition.
-   Person-level representation.
-   Temporal information.
-   Person-level temporal dynamics.
-   Group-level temporal dynamics.
-   Hierarchical temporal modeling.
-   Team-aware spatial pooling.

------------------------------------------------------------------------

# 18. Baseline B1 --- Image Classification

B1 is the simplest baseline.

A single frame is used to directly classify the group activity.

``` text
Single Frame
     ↓
    CNN
     ↓
Group Activity
```

No explicit temporal modeling is used.

The model therefore learns:

``` text
Image → Group Activity
```

### Question

> Can the group activity be recognized from a single frame?

------------------------------------------------------------------------

# 19. Baseline B3 --- Fine-Tuned Person Classification

B3 introduces explicit individual-player information.

Player crops are extracted using the annotated bounding boxes.

``` text
Full Frame
    ↓
Player Bounding Boxes
    ↓
Player Crops
    ↓
CNN
    ↓
Player Features
```

For multiple players:

``` text
Player 1 → CNN → Feature 1
Player 2 → CNN → Feature 2
Player 3 → CNN → Feature 3
...
Player N → CNN → Feature N
```

The features are then pooled and classified:

``` text
Player Features
      ↓
   Pooling
      ↓
Scene Representation
      ↓
Group Classifier
      ↓
Group Activity
```

### Question

> Does explicit modeling of individual players improve group activity
> recognition?

------------------------------------------------------------------------

# 20. Baseline B4 --- Temporal Model with Image Features

B4 introduces temporal information at the scene level.

Instead of using one frame, a sequence of 9 frames is processed.

``` text
Frame t-5 → CNN → Feature
Frame t-4 → CNN → Feature
Frame t-3 → CNN → Feature
Frame t-2 → CNN → Feature
Frame t-1 → CNN → Feature
Frame t   → CNN → Feature
Frame t+1 → CNN → Feature
Frame t+2 → CNN → Feature
Frame t+3 → CNN → Feature
                    ↓
                  LSTM
                    ↓
             Group Activity
```

### Question

> Does temporal information improve group activity recognition?

------------------------------------------------------------------------

# 21. Baseline B5 --- Temporal Model with Person Features

B5 moves temporal modeling to the individual-player level.

For each player:

``` text
Player Crop Sequence
        ↓
       CNN
        ↓
Feature Sequence
        ↓
    Person LSTM
        ↓
Player Temporal Feature
```

The temporal representation of each player is then pooled to recognize
the group activity.

The main difference is:

``` text
B4:
Frame-level temporal modeling

B5:
Player-level temporal modeling
```

### Question

> Is it better to model the temporal dynamics of individual players?

------------------------------------------------------------------------

# 22. Baseline B6 --- Two-Stage Model Without LSTM 1

B6 uses individual player features but does not use the first LSTM.

``` text
Player
  ↓
 CNN
  ↓
Player Feature
  ↓
Pooling
  ↓
Frame Representation
  ↓
Group LSTM
  ↓
Group Activity
```

The temporal modeling happens only at the group level.

### Question

> Is group-level temporal modeling sufficient without a person-level
> temporal model?

------------------------------------------------------------------------

# 23. Baseline B7 --- Two-Stage Model Without LSTM 2

B7 introduces the hierarchical temporal structure.

Each player is modeled over time:

``` text
Player Crop Sequence
        ↓
       CNN
        ↓
   Person LSTM
        ↓
Player Temporal Feature
```

The player representations are then pooled:

``` text
Player Temporal Features
          ↓
      Global Pooling
          ↓
   Group Representation
          ↓
      Group LSTM
          ↓
    Group Activity
```

The model therefore contains two temporal stages:

``` text
Person LSTM
     ↓
Group LSTM
```

### Question

> Does hierarchical temporal modeling improve over using only a single
> temporal level?

------------------------------------------------------------------------

# 24. Baseline B8 --- Two-Stage Hierarchical Model

B8 extends the hierarchical model by preserving the two-team structure.

Instead of:

``` text
All Players
     ↓
Global Pooling
     ↓
Group LSTM
```

the model uses:

``` text
Team 1 Players
     ↓
Max Pooling
     ↓
Team 1 Feature
          \
           → Concatenation → Group LSTM
          /
Team 2 Feature
     ↑
Max Pooling
     ↑
Team 2 Players
```

The full architecture is:

``` text
                     Player Crops
                          │
                          ↓
                 CNN Feature Extraction
                          │
                          ↓
                    Person LSTM
                          │
                          ↓
              Player Temporal Features
                          │
              ┌───────────┴───────────┐
              │                       │
           Team 1                   Team 2
              │                       │
         Max Pooling             Max Pooling
              │                       │
        Team Feature 1          Team Feature 2
              │                       │
              └───────────┬───────────┘
                          │
                     Concatenation
                          │
                          ↓
                  Group Representation
                          │
                          ↓
                     Group LSTM
                          │
                          ↓
                  Group Classifier
                          │
                          ↓
                   Group Activity
```

### Question

> Does preserving the spatial/team structure improve group activity
> recognition?

------------------------------------------------------------------------

# 25. Original Results

The original work reports the following baseline performance on the
Volleyball Dataset.

  Baseline   Method                                   Accuracy
  ---------- ------------------------------------- -----------
  **B1**     Image Classification                    **66.7%**
  **B2**     Person Classification                   **64.6%**
  **B3**     Fine-Tuned Person Classification        **68.1%**
  **B4**     Temporal Model with Image Features      **63.1%**
  **B5**     Temporal Model with Person Features     **67.6%**
  **B6**     Two-Stage Model without LSTM 1          **74.7%**
  **B7**     Two-Stage Model without LSTM 2          **80.2%**
  **B8**     Two-Stage Hierarchical Model            **81.9%**

The results show the benefit of progressively modeling:

``` text
Individual Players
        +
Temporal Dynamics
        +
Hierarchical Structure
        +
Team-Level Spatial Information
```

------------------------------------------------------------------------

# 26. Main Insight

The central insight behind the model is:

> **A group activity can be inferred from the temporal dynamics of the
> individuals participating in the activity.**

Therefore, simply analyzing the complete image is not enough.

A stronger representation is obtained by following the hierarchy:

``` text
Video Frames
     ↓
Player Crops
     ↓
CNN Features
     ↓
Person-Level LSTM
     ↓
Team-Aware Pooling
     ↓
Group-Level LSTM
     ↓
Group Activity
```

This architecture captures two complementary types of information:

### Individual Dynamics

``` text
What is each player doing?
How is that player's action changing over time?
```

### Group Dynamics

``` text
How do the players interact?
How does the team-level configuration evolve over time?
What group activity does this interaction represent?
```

The problem therefore combines:

-   Computer Vision
-   CNN feature extraction
-   Multi-person representation
-   Temporal modeling
-   LSTM networks
-   Spatial reasoning
-   Team-aware aggregation
-   Group activity classification

------------------------------------------------------------------------

# 27. Project Scope

This repository focuses on implementing and studying the group activity
recognition problem through a sequence of baselines, beginning with
simple image-level classification and progressing toward hierarchical
temporal models.

The implementation may differ from the original Caffe-based
implementation in several engineering and architectural details. The
**problem definition, dataset, task formulation, and baseline
progression** are based on the original work.

The goal is to make the problem understandable from the dataset level
all the way to the hierarchical temporal architecture.

The implementation is organized around the following conceptual
pipeline:

``` text
Raw Volleyball Videos
        ↓
Annotated Frames
        ↓
Player Bounding Boxes
        ↓
Player Crops
        ↓
Visual Feature Extraction
        ↓
Person-Level Modeling
        ↓
Player / Team Pooling
        ↓
Group-Level Temporal Modeling
        ↓
Group Activity Prediction
```

------------------------------------------------------------------------

# 28. References

## Original CVPR Paper

M. S. Ibrahim, S. Muralidharan, Z. Deng, A. Vahdat, and G. Mori,

**"A Hierarchical Deep Temporal Model for Group Activity Recognition,"**

*Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR)*, 2016.

## Extended Work

M. S. Ibrahim, S. Muralidharan, Z. Deng, A. Vahdat, and G. Mori,

**"Hierarchical Deep Temporal Models for Group Activity Recognition,"**

arXiv:1607.02643, 2016.

## Original Repository

https://github.com/mostafa-saad/deep-activity-rec

## Citation

``` bibtex
@inproceedings{Ibrahim_2016_CVPR,
    author    = {Ibrahim, Mostafa S. and
                 Muralidharan, Srikanth and
                 Deng, Zhiwei and
                 Vahdat, Arash and
                 Mori, Greg},
    title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and
                 Pattern Recognition (CVPR)},
    year      = {2016}
}

@article{Ibrahim2016Hierarchical,
    author  = {Ibrahim, Mostafa S. and
               Muralidharan, Srikanth and
               Deng, Zhiwei and
               Vahdat, Arash and
               Mori, Greg},
    title   = {Hierarchical Deep Temporal Models for Group Activity Recognition},
    journal = {arXiv preprint arXiv:1607.02643},
    year    = {2016}
}
```
