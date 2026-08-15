#!/bin/bash
#SBATCH --job-name=volleyball
#SBATCH --output=/nfs/slurm/assu002/projects/volleyball_project/logs/baseline5.txt
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_1g.20gb:4
##SBATCH --nodelist=gpu2

echo "========================================"
echo " VOLLEYBALL PROJECT TRAINING"
echo " Started : $(date)"
echo " Node    : $(hostname)"
echo "========================================"

# ── activate conda ─────────────────────────
source /nfs/slurm/$USER/miniconda3/etc/profile.d/conda.sh
conda activate vision_env

# ── check GPU ─────────────────────────────
python3 -c "
import torch
print('CUDA :', torch.cuda.is_available())
print('VRAM :', torch.cuda.get_device_properties(0).total_memory//1024**3, 'GB')
"

# ── smart download ─────────────────────────
DATA_DIR="/tmp/assu002/volleyball_data"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/videos" ] && [ "$(ls -A $DATA_DIR/videos)" ]; then
    echo "=== Dataset exists — skipping download ==="
else
    echo "=== Downloading dataset ==="
    /nfs/slurm/$USER/miniconda3/bin/kaggle datasets download \
        -d sherif31/group-activity-recognition-volleyball \
        --path /tmp/assu002 --unzip
    mv /tmp/assu002/videos        $DATA_DIR/videos        2>/dev/null
    mv /tmp/assu002/annot_all.pkl $DATA_DIR/annot_all.pkl 2>/dev/null
    mv /tmp/assu002/volleyball_tracking_annotation $DATA_DIR/volleyball_tracking_annotation 2>/dev/null
fi

echo "=== Data ready ==="
ls -lh $DATA_DIR

# ── create output folders ──────────────────
#mkdir -p /nfs/slurm/$USER/outputs/volleyball_project

# ── run training ───────────────────────────
cd /nfs/slurm/$USER/projects/volleyball_project

echo "=== Starting Training ==="
python3 -u /nfs/slurm/$USER/projects/volleyball_project/runs/baseline5/Run_Baseline5.py

echo ""
echo "=== Saved outputs ==="
ls -lh /nfs/slurm/$USER/projects/volleyball_project/outputs/baseline5

echo " DONE: $(date)"
echo "========================================"
