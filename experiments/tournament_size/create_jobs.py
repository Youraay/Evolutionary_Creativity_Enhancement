"""Create a job database of the experiment tournament_size: variant x prompt x seed.

    python -m experiments.tournament_size.create_jobs --db new_jobs.db
    python -m experiments.tournament_size.create_jobs --db new_jobs.db --prompts "a cat" --seeds 1943746817
"""
from experiments.common.embedding_spaces import create_jobs_cli
from experiments.tournament_size.config import EXPERIMENT

if __name__ == "__main__":
    create_jobs_cli(EXPERIMENT, __doc__)
