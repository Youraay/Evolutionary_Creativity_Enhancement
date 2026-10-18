"""Run one job of the experiment tournament_size.

    python -m experiments.tournament_size.worker <job_id>
"""
from experiments.common.embedding_spaces import worker_cli
from experiments.tournament_size.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
