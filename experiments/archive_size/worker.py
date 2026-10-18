"""Run one job of the experiment archive_size.

    python -m experiments.archive_size.worker <job_id>
"""
from experiments.archive_size.config import EXPERIMENT
from experiments.common.variants import worker_cli

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
