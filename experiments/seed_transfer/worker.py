"""Run one job of the experiment seed_transfer.

    python -m experiments.seed_transfer.worker <job_id>
"""
from experiments.common.variants import worker_cli
from experiments.seed_transfer.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
