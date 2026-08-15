# 🏐 Volleyball Activity Recognition & Multi-Task Learning Pipeline

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![HPC](https://img.shields.io/badge/Slurm-HPC-blue?style=for-the-badge)

## 📌 Project Overview
This repository contains a high-performance deep learning pipeline designed for **Group Activity Recognition (GAR)** using the Volleyball dataset. The project evolves from a standard single-task classification baseline into a robust **Multi-Task Learning (MTL)** architecture, simultaneously predicting individual player actions and overarching team activities.

The codebase is highly optimized for High-Performance Computing (HPC) environments (e.g., NVIDIA A100 GPUs), featuring intelligent data caching, dynamic memory management, and clean software architecture.

---

## 🏗️ Software Architecture & Design Patterns

To ensure maximum scalability and maintainability, the project strictly adheres to solid software engineering principles:

* **Adapter Design Pattern:** The data loading pipeline heavily utilizes the Adapter pattern (e.g., `BaseAdapter`, `TrackingAdapter`). This decouples the raw dataset format from the PyTorch `DataLoader`, allowing seamless switching between different datasets or preprocessing strategies without altering the core training loops.
* **Modular Pipeline:** The architecture separates concerns into distinct modules: `datasets`, `models`, `adapters`, and `training` scripts.
* **Dynamic RAM Caching:** Implemented an On-the-Fly Memory Cache within the Adapter using OpenCV (`cv2`), resolving CPU bottlenecks during sequential bounding box cropping and drastically reducing epoch times while supporting multi-processing (`persistent_workers=True`).

---

## 🗂️ Dataset Description & Structure

The project utilizes a structured representation of the **Volleyball Dataset**.

* **Input Data:** Sequences of video frames (typically 9-frame clips, e.g., frames 5 to 14 around the key event).
* **Annotations:** 
  * Bounding boxes `[x1, y1, x2, y2]` for tracking active players.
  * Individual action labels (e.g., spiking, blocking, setting).
  * Group activity labels (e.g., right set, left spike).
* **Data Flow:** The `TrackingAdapter` dynamically loads frames, applies bounding box crops using OpenCV, resizes them to `(224, 224)`, and stacks them into a unified clip tensor of shape `(T, C, H, W)`.

---

## 🧠 Multi-Task Learning (MTL)

A core innovation in this repository is the implementation of a Multi-Task Learning architecture. Instead of solely predicting the final group activity, the model shares underlying feature representations (via a ResNet50 backbone) to simultaneously optimize for:
1. **Primary Task:** Group Activity Recognition (GAR).
2. **Auxiliary Task:** Individual Player Action Recognition.

**Benefits achieved:**
* Improved feature generalization and reduced overfitting.
* Richer spatial-temporal context extraction.
* A shared representation that forces the network to pay attention to specific player movements that dictate the overall team play.

---

## 🚀 Baselines & Experiments

The project was developed iteratively through several baselines, each introducing architectural or pipeline improvements. 

| Baseline | Description | F1-Score |
| :--- | :--- | :---: |
| **Baseline 1** | Initial setup using standard CNN-RNN sequential processing. Basic bounding box cropping and single-task objective. | **70.0** |
| **Baseline 2** | Addressed structural data-loading bottlenecks and integrated initial transformations. | - |
| **Baseline 3** | **(Best Performance)** Integrated advanced architectures and heavy augmentation (`torchvision.transforms.v2`). Reached highest stability and accuracy before MTL overhead. | **83.0** |
| **Baseline 4** | Introduced the **Multi-Task Learning** formulation (Person-level + Group-level). Slightly more complex optimization landscape but highly robust generalized features. | **82.0** |
| **Baseline 5** | *(Active Development)* Focused on HPC scaling, A100 MIG optimizations, solving VRAM fragmentation (`expandable_segments:True`), and RAM-based OpenCV caching for ultra-fast throughput. | *TBD* |

---

## 📂 Project Directory Structure

```text
volleyball_project/
├── dataset/
│   ├── adapters/
│   │   ├── base_adapter.py      # Abstract base class for adapters
│   │   └── tracking_adapter.py  # OpenCV & RAM Caching implementation
│   └── dataset.py               # Main PyTorch Dataset definition
├── models/
│   ├── backbone.py              # ResNet50 definitions
│   └── mtl_model.py             # Multi-Task Learning heads (GAR + Individual)
├── training/
│   ├── Run_BaselineX.py         # Entry points for specific baselines
│   └── trainer.py               # Core training/validation loops
├── scripts/
│   └── sbatch_run.sh            # Slurm cluster submission scripts
└── README.md