#!/bin/bash
#SBATCH --job-name=final-run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x.%A_%a.out
# One array task = one final run.
# Submit per (dataset percentage, method):
#   sbatch --array=1-9 \
#     --export=ALL,PCT=10,METHOD=tacil-dp-bounded,SWEEP_ID=pligzyst \
#     /scratch/boateng/masterarbeit_tools/final_runner.sbatch
# The array size is top_k * number_of_seeds (default 3 * 3 = 9).
# Stop the sweep in wandb before submitting, so the top-3 ranking is frozen
# and identical across every array task.
set -euo pipefail

cd /scratch/boateng/imitation_learning_and_data_control_augmentation

# Tag every run with its dataset size and method, so the single final project
# stays groupable and filterable in wandb.
export WANDB_TAGS="${PCT}pct,${METHOD}"

COMMAND=$(uv run --no-sync python -m scripts.sweep.launch_final_runs \
    --dataset-percentage "$PCT" --method "$METHOD" --sweep-id "$SWEEP_ID" \
    | grep '^python -m' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$COMMAND" ]; then
    echo "No command for array index ${SLURM_ARRAY_TASK_ID}." >&2
    exit 1
fi

echo "RUNNING: $COMMAND"
uv run --no-sync $COMMAND
