#!/bin/bash
#SBATCH --job-name=mopo-replication
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x.%A_%a.out
#SBATCH --array=1-3
# Checks the MOPO port against the scores OfflineRL-Kit publishes.
# One array task = one D4RL task, all at seed 0. Submit with:
#   sbatch /scratch/boateng/masterarbeit_tools/mopo_replication.sbatch
# Download the datasets on the frontend first, see the README section.
set -euo pipefail

cd /scratch/boateng/imitation_learning_and_data_control_augmentation

# uv lives in the home directory, not in the system path.
source "$HOME/.local/bin/env"

# Caches default to the home directory, which is mirrored and nearly full.
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export WANDB_CACHE_DIR="/scratch/$USER/wandb-cache"
export WANDB_DATA_DIR="/scratch/$USER/wandb-staging"
export WANDB_ENTITY="bryanboateng-team"

# MuJoCo picks a rendering backend at import even when nothing is drawn.
# There is no display on a compute node, so name a headless one.
export MUJOCO_GL=egl

# torch would otherwise start one thread per core of the whole node rather
# than per core of this allocation, and the threads then fight each other.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TASKS=(
    "walker2d-medium-v2"
    "halfcheetah-medium-v2"
    "walker2d-medium-replay-v2"
)
TASK="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "task ${TASK}, seed ${SEED:-0}"

# --no-sync: the environment is built once on the frontend. Compute nodes
# must not resolve or install anything.
uv run --no-sync python -m scripts.policies.mopo.replication.run \
    --wandb-project=mopo-replication \
    --task="${TASK}" \
    --seed="${SEED:-0}"
