#!/bin/bash
# Rewrite the prompts with Think-Then-Generate, measure the DiT and render example images.
#   sbatch experiments/think_then_generate/submit_rewrite.sh <output dir>
#SBATCH --job-name=t2g_rewrite
#SBATCH --output=logs/t2g-%j.out
#SBATCH --error=logs/t2g-%j.err
#SBATCH --partition=gpu2
#SBATCH --account=dldevel
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

source "${SLURM_SUBMIT_DIR:-.}/experiments/common/slurm_env.sh"
python -m experiments.think_then_generate.rewrite_prompts --out "$1"
