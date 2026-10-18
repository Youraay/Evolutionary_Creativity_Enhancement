#!/bin/bash
# PCA of initial noise and BLIP-2 embeddings of the PartiPrompts experiment (CPU only).
#   sbatch analysis/submit_pca.sh [--limit N]
# ANALYSIS_PYTHON selects the Python of the analysis environment (default: python on the PATH).
#SBATCH --job-name=pca_parti
#SBATCH --output=logs/pca-parti-%j.out
#SBATCH --error=logs/pca-parti-%j.err
#SBATCH --partition=test
#SBATCH --account=dldevel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=180G
#SBATCH --time=02:00:00

export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
[ -f .env ] && set -a && source .env && set +a
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
echo "[INFO] Host: $(hostname)"
"${ANALYSIS_PYTHON:-python}" -m analysis.pca_parti_prompts compute --workers 12 "$@"
