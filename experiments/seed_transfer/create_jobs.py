"""Create a job database of the experiment seed_transfer.

    python -m experiments.seed_transfer.create_jobs --db new_jobs.db                  # open jobs for all variants
    python -m experiments.seed_transfer.create_jobs --db new_jobs.db --variants <variant> --seeds 1
    python -m experiments.seed_transfer.create_jobs --register                        # finished runs as DONE
"""
from experiments.common.variants import create_jobs_cli
from experiments.seed_transfer.config import EXPERIMENT

if __name__ == "__main__":
    create_jobs_cli(EXPERIMENT, __doc__)
