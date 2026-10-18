#!/bin/bash
# One run of the experiment operator_ablation; submitted by experiments.operator_ablation.dispatch.
#   sbatch experiments/operator_ablation/submit_job.sh <job_id>
#SBATCH --job-name=operator_ablation
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu2
#SBATCH --account=dldevel
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00

source "${SLURM_SUBMIT_DIR:-.}/experiments/common/slurm_env.sh"
python -m experiments.operator_ablation.worker "$1"
