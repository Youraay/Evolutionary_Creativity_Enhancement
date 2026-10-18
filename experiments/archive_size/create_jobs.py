"""Create a job database of the experiment archive_size.

    python -m experiments.archive_size.create_jobs --db new_jobs.db                  # open jobs for all variants
    python -m experiments.archive_size.create_jobs --db new_jobs.db --variants <variant> --seeds 1
    python -m experiments.archive_size.create_jobs --register                        # finished runs as DONE
"""
from experiments.archive_size.config import EXPERIMENT
from experiments.common.variants import create_jobs_cli

if __name__ == "__main__":
    create_jobs_cli(EXPERIMENT, __doc__)
