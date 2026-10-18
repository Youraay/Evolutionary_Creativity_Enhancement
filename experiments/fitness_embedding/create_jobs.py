"""Create a job database of the experiment fitness_embedding: variant x prompt x seed.

    python -m experiments.fitness_embedding.create_jobs --db new_jobs.db
    python -m experiments.fitness_embedding.create_jobs --db new_jobs.db --prompts "a cat" --seeds 1943746817
"""
from experiments.common.embedding_spaces import create_jobs_cli
from experiments.fitness_embedding.config import EXPERIMENT

if __name__ == "__main__":
    create_jobs_cli(EXPERIMENT, __doc__)
