#!/bin/bash
# Several model-comparison runs in ONE Slurm job: one list of job ids per GPU, processed one after
# another. Saves QOS slots (gpu2: at most 4 running jobs with up to 2 GPUs each). The jobs are run
# directly by id, independent of their status; give them the status HOLD so the dispatcher skips them.
#   sbatch --gres=gpu:2 [--nodelist=..] [--time=..] experiments/model_comparison/submit_gpu_queue.sh "<ids GPU 0>" "<ids GPU 1>"
# Ids within a list are separated by spaces or commas; every run writes logs/mcq-<slurm id>-job<id>.log.
#SBATCH --job-name=mc_queue
#SBATCH --output=logs/mcq-%j.out
#SBATCH --error=logs/mcq-%j.err
#SBATCH --partition=gpu2
#SBATCH --account=dldevel
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

source "${SLURM_SUBMIT_DIR:-.}/experiments/common/slurm_env.sh"
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES} | lists: $*"

run_queue() {
    local gpu=$1 ids=$2
    for id in ${ids//,/ }; do
        echo "[INFO] $(date '+%F %T') GPU $gpu: job $id starts"
        CUDA_VISIBLE_DEVICES=$gpu python -m experiments.model_comparison.worker "$id" \
            > "logs/mcq-${SLURM_JOB_ID}-job${id}.log" 2>&1
        echo "[INFO] $(date '+%F %T') GPU $gpu: job $id finished (rc=$?)"
    done
}

i=0
for ids in "$@"; do
    run_queue "${GPUS[$i]}" "$ids" &
    i=$((i + 1))
done
wait
echo "[INFO] $(date '+%F %T') all lists finished"
