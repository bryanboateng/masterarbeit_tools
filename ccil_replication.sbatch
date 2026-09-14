#!/bin/bash
#SBATCH --job-name=ccil-replication
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/%x.%A_%a.out
#SBATCH --array=1-80
# Trains the policies that check the CCIL port. Evaluation happens on hal,
# where the CCIL environment with mujoco-py and metaworld lives.
#
# Any card rather than an a100: this trains a dynamics model and a two-layer
# policy without torch.compile, so the P100 nodes are enough.
#
# One array task = one task at one seed, which is three policies: plain
# behavior cloning, noise, and CCIL itself. Submit with:
#   sbatch /scratch/boateng/masterarbeit_tools/ccil_replication.sbatch
# Copy the CCIL config and data directories to /scratch/$USER/CCIL first.
set -euo pipefail

cd /scratch/boateng/imitation_learning_and_data_control_augmentation

# uv lives in the home directory, not in the system path.
source "$HOME/.local/bin/env"

# Caches default to the home directory, which is mirrored and nearly full.
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export WANDB_CACHE_DIR="/scratch/$USER/wandb-cache"
export WANDB_DATA_DIR="/scratch/$USER/wandb-staging"
export WANDB_ENTITY="bryanboateng-team"

# torch would otherwise start one thread per core of the whole node rather
# than per core of this allocation, and the threads then fight each other.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TASKS=(
    "hopper"
    "walker2d"
    "halfcheetah"
    "ant"
    "coffee_pull"
    "coffee_push"
    "button"
    "drawer_close"
)
SEEDS=(40 41 42 43 44 45 46 47 48 49)

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
TASK="${TASKS[$((INDEX / ${#SEEDS[@]}))]}"
SEED="${SEEDS[$((INDEX % ${#SEEDS[@]}))]}"

echo "task ${TASK}, seed ${SEED}"

# --no-sync: the environment is built once on the frontend. Compute nodes
# must not resolve or install anything.
uv run --no-sync python -m scripts.data.augmentation.ccil.replication.train \
    --ccil-directory-path="/scratch/$USER/CCIL" \
    --wandb-project=ccil-replication \
    --tasks="${TASK}" \
    --seeds="${SEED}"
