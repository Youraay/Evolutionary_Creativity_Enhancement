"""Run one job of the experiment long_runs.

    python -m experiments.long_runs.worker <job_id>
"""
from experiments.common.embedding_spaces import worker_cli
from experiments.long_runs.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
