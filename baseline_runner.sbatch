#!/bin/bash
#SBATCH --job-name=baseline-run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/%u/logs/%x.%A_%a.out
# No-augmentation baseline: train the policy on the raw expert data only.
# One array task = one seed. Submit per (policy, dataset percentage):
#   sbatch --array=1-3 --export=ALL,POL=dp,PCT=10 \
#     /scratch/boateng/masterarbeit_tools/baseline_runner.sbatch
# POL is dp or bc; array 1-3 maps to seeds 0,1,2.
set -euo pipefail

cd /scratch/boateng/imitation_learning_and_data_control_augmentation

export WANDB_ENTITY="bryanboateng-team"
export WANDB_TAGS="${PCT}pct,${POL}-noaug"

SEED=$((SLURM_ARRAY_TASK_ID - 1))

uv run --no-sync python -m "scripts.policies.${POL}.run" \
    --wandb-project=2026-08-24-final \
    --expert-dataset-artifact="datasets/dataset-${PCT}:latest" \
    --seed="${SEED}" \
    --evaluate-all-conditions \
    augmentation:none
