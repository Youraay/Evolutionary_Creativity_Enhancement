"""Run one job of the experiment fitness_embedding (called by ``submit_job.sh``).

    python -m experiments.fitness_embedding.worker <job_id>
"""
from experiments.common.embedding_spaces import worker_cli
from experiments.fitness_embedding.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
