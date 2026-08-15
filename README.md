# Volleyball Group Activity Recognition

This repository contains the codebase for training and evaluating deep learning models for group activity recognition using the Volleyball Dataset. The current architecture leverages a ResNet50 backbone combined with an LSTM sequence model (Baseline 5) to process video frames and predict collective activities.

## Project Structure
* `models/`: Contains the model architectures (e.g., ResNet backbone, Baseline 5).
* `trainers/`: Includes training loops, evaluation logic, and loss calculations.
* `runs/`: Execution scripts for different baselines.
* `logs/`: Slurm output logs and training progress.

## Environment Setup
The project uses a Conda environment (`vision_env`). 
Dependencies include PyTorch (CUDA-enabled) and torchvision.

## Dataset
The dataset used is the [Group Activity Recognition Volleyball Dataset].
Ensure the dataset is downloaded and extracted into the appropriate directory before running the training scripts. The data loader handles videos and `annot_all.pkl` annotations.

## Training on Slurm Cluster
To submit a training job on the HPC cluster using Slurm:
```bash
sbatch runs/baseline5/run_baseline5.sh
