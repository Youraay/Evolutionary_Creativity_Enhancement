"""Create a job database of the experiment mutation_only.

    python -m experiments.mutation_only.create_jobs --db new_jobs.db                  # open jobs for all variants
    python -m experiments.mutation_only.create_jobs --db new_jobs.db --variants <variant> --seeds 1
    python -m experiments.mutation_only.create_jobs --register                        # finished runs as DONE
"""
from experiments.common.variants import create_jobs_cli
from experiments.mutation_only.config import EXPERIMENT

if __name__ == "__main__":
    create_jobs_cli(EXPERIMENT, __doc__)
