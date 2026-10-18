"""Run one job of the experiment fitness_function.

    python -m experiments.fitness_function.worker <job_id>
"""
from experiments.common.variants import worker_cli
from experiments.fitness_function.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
