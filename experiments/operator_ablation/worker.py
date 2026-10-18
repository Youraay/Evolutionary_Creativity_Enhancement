"""Run one job of the experiment operator_ablation.

    python -m experiments.operator_ablation.worker <job_id>
"""
from experiments.common.variants import worker_cli
from experiments.operator_ablation.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
