"""Create a job database of the experiment crossover_function.

    python -m experiments.crossover_function.create_jobs --db new_jobs.db                  # open jobs for all variants
    python -m experiments.crossover_function.create_jobs --db new_jobs.db --variants Uniform --seeds 1
    python -m experiments.crossover_function.create_jobs --register                        # finished runs as DONE
"""
from experiments.common.variants import create_jobs_cli
from experiments.crossover_function.config import EXPERIMENT

if __name__ == "__main__":
    create_jobs_cli(EXPERIMENT, __doc__)
