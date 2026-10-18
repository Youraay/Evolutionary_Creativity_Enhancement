#!/bin/bash
# Variant B: rewritten (or original) prompt + seed optimisation with SDXL, one run per job.
#   sbatch experiments/think_then_generate/submit_sdxl.sh "<original prompt>" <seed> [generations=100] [t2g|original] [population=60]
#SBATCH --job-name=t2g_sdxl
#SBATCH --output=logs/t2g-sdxl-%j.out
#SBATCH --error=logs/t2g-sdxl-%j.err
#SBATCH --partition=gpu2
#SBATCH --account=dldevel
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00

source "${SLURM_SUBMIT_DIR:-.}/experiments/common/slurm_env.sh"
python -m experiments.think_then_generate.worker --prompt "$1" --seed "$2" --generations "${3:-100}" \
    --condition "${4:-t2g}" --population "${5:-60}"
