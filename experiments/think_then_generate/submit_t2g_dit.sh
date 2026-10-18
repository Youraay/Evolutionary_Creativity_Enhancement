#!/bin/bash
# Variant A: T2G DiT as generator + seed optimisation, one run per job.
#   sbatch experiments/think_then_generate/submit_t2g_dit.sh "<original prompt>" <seed> <t2g|original> [generations=30] [steps=30] [population=60]
#SBATCH --job-name=t2g_dit
#SBATCH --output=logs/t2g-dit-%j.out
#SBATCH --error=logs/t2g-dit-%j.err
#SBATCH --partition=gpu2
#SBATCH --account=dldevel
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-12:00:00

source "${SLURM_SUBMIT_DIR:-.}/experiments/common/slurm_env.sh"
python -m experiments.think_then_generate.worker --generator t2g --prompt "$1" --seed "$2" --condition "$3" \
    --generations "${4:-30}" --steps "${5:-30}" --population "${6:-60}"
