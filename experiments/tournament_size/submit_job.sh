#!/bin/bash
# One run of the experiment tournament_size; submitted by experiments.tournament_size.dispatch.
#   sbatch experiments/tournament_size/submit_job.sh <job_id>
#SBATCH --job-name=tournament_size
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu2
#SBATCH --account=dldevel
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00

source "${SLURM_SUBMIT_DIR:-.}/experiments/common/slurm_env.sh"
python -m experiments.tournament_size.worker "$1"
