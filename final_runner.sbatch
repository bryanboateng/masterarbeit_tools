#!/bin/bash
#SBATCH --job-name=final-run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x.%A_%a.out
# One array task = one final run.
# Submit per (dataset, method):
#   sbatch --array=1-9 \
#     --export=ALL,DATASET=dataset-PushT-20-of-206:latest,METHOD=tacil-dp,SWEEP_ID=pligzyst \
#     /scratch/boateng/masterarbeit_tools/final_runner.sbatch
# The array size is top_k * number_of_seeds (default 3 * 3 = 9).
# Stop the sweep in wandb before submitting, so the top-3 ranking is frozen
# and identical across every array task.
set -euo pipefail

cd /scratch/boateng/imitation_learning_and_data_control_augmentation

# Land every final run in the team entity (where the sweeps live), regardless
# of the login's default entity.
export WANDB_ENTITY="bryanboateng-team"

# Tag the method only. The dataset is read back off the run's own artifact
# name, so a forgotten tag cannot drop a run from the summary.
export WANDB_TAGS="${METHOD}"

COMMAND=$(uv run --no-sync --project /scratch/boateng/masterarbeit_tools \
    python /scratch/boateng/masterarbeit_tools/launch_final_runs.py \
    --dataset-artifact "$DATASET" --method "$METHOD" --sweep-id "$SWEEP_ID" \
    | grep '^python -m' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$COMMAND" ]; then
    echo "No command for array index ${SLURM_ARRAY_TASK_ID}." >&2
    exit 1
fi

echo "RUNNING: $COMMAND"
uv run --no-sync $COMMAND
