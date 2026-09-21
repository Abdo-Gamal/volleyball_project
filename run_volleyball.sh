#!/bin/bash
#SBATCH --job-name=volleyball
#SBATCH --output=/nfs/slurm/assu002/projects/volleyball_project/logs/baseline5.txt
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100_1g.20gb:1

# -- UNCOMMENT THE NEXT LINE IF YOU ONLY WANT TO USE GPU1 (WHERE DATA ALREADY EXISTS) --
##SBATCH --nodelist=gpu1

echo "========================================"
echo " VOLLEYBALL PROJECT TRAINING"
echo " Started : $(date)"
echo " Node    : $(hostname)"
echo "========================================"

# 1. Activate conda environment
source /nfs/slurm/$USER/miniconda3/etc/profile.d/conda.sh
conda activate vision_env

# 2. Check GPU details
python3 -c "
import torch
print('CUDA :', torch.cuda.is_available())
print('VRAM :', torch.cuda.get_device_properties(0).total_memory//1024**3, 'GB')
"

# 3. Setup local /tmp directory for the current node
LOCAL_TMP="/tmp/$USER"
DATA_DIR="$LOCAL_TMP/volleyball_data"

mkdir -p "$DATA_DIR"

# 4. Force Kaggle to use local /tmp for caching to protect your 30GB NFS quota
export KAGGLE_CACHE_DIR="$LOCAL_TMP/kaggle_cache"
mkdir -p "$KAGGLE_CACHE_DIR"

# 5. Smart download logic
if [ -d "$DATA_DIR/videos" ] && [ "$(ls -A "$DATA_DIR/videos")" ]; then
    echo "=== Dataset exists on this node — skipping download ==="
else
    echo "=== Downloading dataset to local /tmp of $(hostname) ==="

    # Clean any corrupted partial downloads first
    rm -f "$LOCAL_TMP/group-activity-recognition-volleyball.zip"

    /nfs/slurm/$USER/miniconda3/bin/kaggle datasets download \
        -d sherif31/group-activity-recognition-volleyball \
        --path "$LOCAL_TMP" --unzip

    mv "$LOCAL_TMP/videos" "$DATA_DIR/videos" 2>/dev/null
    mv "$LOCAL_TMP/annot_all.pkl" "$DATA_DIR/annot_all.pkl" 2>/dev/null
    mv "$LOCAL_TMP/volleyball_tracking_annotation" "$DATA_DIR/volleyball_tracking_annotation" 2>/dev/null

    # Clean up the cache to free space on the local drive
    rm -rf "$KAGGLE_CACHE_DIR"
fi

echo "=== Data ready ==="
ls -lh "$DATA_DIR"

# 6. Run training script
cd /nfs/slurm/$USER/projects/volleyball_project

echo "=== Starting Training ==="
python3 -u runs/baseline5/Run_Baseline5_seq.py

echo " DONE: $(date)"
echo "========================================"




# with profiling

# #!/bin/bash
# #SBATCH --job-name=volleyball
# #SBATCH --output=/nfs/slurm/assu002/projects/volleyball_project/logs/baseline5.txt
# #SBATCH --time=10:00:00
# #SBATCH --mem=64G
# #SBATCH --cpus-per-task=5
# #SBATCH --gres=gpu:a100_1g.20gb:1

# # -- UNCOMMENT THE NEXT LINE IF YOU ONLY WANT TO USE GPU1 (WHERE DATA ALREADY EXISTS) --
# ##SBATCH --nodelist=gpu1

# echo "========================================"
# echo " VOLLEYBALL PROJECT TRAINING (with profiling)"
# echo " Started : $(date)"
# echo " Node    : $(hostname)"
# echo "========================================"

# # 1. Activate conda environment
# source /nfs/slurm/$USER/miniconda3/etc/profile.d/conda.sh
# conda activate vision_env

# # Make sure pynvml is available (one-time install, safe to run every time - no-op if present)
# pip install --quiet --user pynvml 2>/dev/null

# # 2. Print system RAM status
# echo "=== System RAM Status ==="
# free -h

# # 3. Print detailed GPU hardware info
# echo "=== PyTorch GPU Details ==="
# python3 -c "
# import torch
# print('CUDA :', torch.cuda.is_available())
# if torch.cuda.is_available():
#     print('Device :', torch.cuda.get_device_name(0))
#     print('VRAM   :', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
# "

# # 4. Setup local /tmp directory for the current node
# LOCAL_TMP="/tmp/$USER"
# DATA_DIR="$LOCAL_TMP/volleyball_data"

# mkdir -p "$DATA_DIR"

# # 5. Force Kaggle to use local /tmp for caching to protect your 30GB NFS quota
# export KAGGLE_CACHE_DIR="$LOCAL_TMP/kaggle_cache"
# mkdir -p "$KAGGLE_CACHE_DIR"

# # 6. Smart download logic (checks THIS node's local /tmp)
# if [ -d "$DATA_DIR/videos" ] && [ "$(ls -A "$DATA_DIR/videos")" ]; then
#     echo "=== Dataset exists on this node ($(hostname)) — skipping download ==="
# else
#     echo "=== Downloading dataset to local /tmp of $(hostname) ==="

#     rm -f "$LOCAL_TMP/group-activity-recognition-volleyball.zip"

#     /nfs/slurm/$USER/miniconda3/bin/kaggle datasets download \
#         -d sherif31/group-activity-recognition-volleyball \
#         --path "$LOCAL_TMP" --unzip

#     mv "$LOCAL_TMP/videos" "$DATA_DIR/videos" 2>/dev/null
#     mv "$LOCAL_TMP/annot_all.pkl" "$DATA_DIR/annot_all.pkl" 2>/dev/null
#     mv "$LOCAL_TMP/volleyball_tracking_annotation" "$DATA_DIR/volleyball_tracking_annotation" 2>/dev/null

#     rm -rf "$KAGGLE_CACHE_DIR"
# fi

# echo "=== Data ready ==="
# ls -lh "$DATA_DIR"

# # 7. Start continuous GPU profiling in the background using pynvml (NOT nvidia-smi directly,
# #    which is blocked by the cluster's job filter)
# PROFILER_LOG="/nfs/slurm/$USER/projects/volleyball_project/logs/gpu_profile.csv"
# mkdir -p "$(dirname "$PROFILER_LOG")"

# echo "=== Starting GPU Profiler via pynvml (saving to $PROFILER_LOG) ==="
# python3 - "$PROFILER_LOG" <<'PYEOF' &
# import sys, time, csv, datetime
# import pynvml

# log_path = sys.argv[1]
# interval_sec = 10

# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# name = pynvml.nvmlDeviceGetName(handle)
# if isinstance(name, bytes):
#     name = name.decode()

# with open(log_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["timestamp", "name", "utilization.gpu (%)", "utilization.memory (%)",
#                       "memory.used (MiB)", "memory.total (MiB)"])
#     f.flush()
#     while True:
#         util = pynvml.nvmlDeviceGetUtilizationRates(handle)
#         mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         writer.writerow([
#             datetime.datetime.now().isoformat(timespec="seconds"),
#             name,
#             util.gpu,
#             util.memory,
#             mem.used // 1024**2,
#             mem.total // 1024**2,
#         ])
#         f.flush()
#         time.sleep(interval_sec)
# PYEOF
# PROFILER_PID=$!

# # Make sure the profiler is always killed, even if training crashes or the job is cancelled
# cleanup() {
#     echo "=== Stopping GPU Profiler (PID $PROFILER_PID) ==="
#     kill "$PROFILER_PID" 2>/dev/null
# }
# trap cleanup EXIT

# # 8. Run training script
# cd /nfs/slurm/$USER/projects/volleyball_project

# echo "=== Starting Training ==="
# python3 -u runs/baseline5/Run_Baseline5_seq.py

# echo " DONE: $(date)"
# echo "========================================"

