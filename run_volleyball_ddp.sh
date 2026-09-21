#!/bin/bash
#SBATCH --job-name=volleyball
#SBATCH --output=/nfs/slurm/assu002/projects/volleyball_project/logs/baseline5.txt
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:a100_1g.20gb:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu1

# [CHANGED 0] --nodes=1 : all 4 GPU slices come from ONE node (--gres is counted per node). If one node cannot give 4,
#             the job waits in the queue; it is never split over two nodes.
# [CHANGED 1] %j in the log name = job number, so a new job does not overwrite the old log
# [CHANGED 2] --nodelist=gpu1 : the dataset is in gpu1's /tmp (a local disk), so the job must run on gpu1.
#             Remove this line and the job may land on another node, which has no data there.

echo "========================================"
echo " VOLLEYBALL PROJECT TRAINING"
echo " Started : $(date)"
echo " Node    : $(hostname)"
echo "========================================"
# do not need when use gloo
#export NCCL_P2P_DISABLE=1
# ── activate conda ─────────────────────────
source /nfs/slurm/$USER/miniconda3/etc/profile.d/conda.sh
conda activate vision_env

# ── check GPU ─────────────────────────────
python3 -c "
import torch
print('CUDA :', torch.cuda.is_available())
print('VRAM :', torch.cuda.get_device_properties(0).total_memory//1024**3, 'GB')
"

# ── local /tmp on THIS node (a local disk: no NFS quota, but not shared between nodes) ──
LOCAL_TMP="/tmp/$USER"
DATA_DIR="$LOCAL_TMP/volleyball_data"
mkdir -p "$DATA_DIR"

# Force Kaggle to use local /tmp for caching to protect your 30GB NFS quota
export KAGGLE_CACHE_DIR="$LOCAL_TMP/kaggle_cache"
export KAGGLE_CONFIG_DIR="$LOCAL_TMP/kaggle_config"

mkdir -p "$KAGGLE_CACHE_DIR"

# [CHANGED 4] same idea for PyTorch: the pretrained ResNet50 weights (98 MB) go to the local disk.
#             Before, they went to /nfs/.../.cache/torch and the job died with "Disk quota exceeded".
export TORCH_HOME="$LOCAL_TMP/torch_cache"
mkdir -p "$TORCH_HOME"

# ── smart download ─────────────────────────

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

# [CHANGED 5] download the ResNet50 weights ONCE here, before the 4 processes start.
#             Before, all 4 processes downloaded the same file at the same time into the same place.
python3 -c "
import torchvision
torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
print('ResNet50 weights ready')
"

# ── run training ───────────────────────────
cd /nfs/slurm/$USER/projects/volleyball_project

echo "=== Starting Training ==="

# [CHANGED 3] --standalone = "everything is on this one node": no rendezvous address, no port to pick
torchrun --standalone --nproc_per_node=3 /nfs/slurm/assu002/projects/volleyball_project/runs/baseline5/Run_Baseline5_ddp.py

echo ""
echo "=== Saved outputs ==="
ls -lh /nfs/slurm/$USER/projects/volleyball_project/outputs/baseline5

echo " DONE: $(date)"
echo "========================================"