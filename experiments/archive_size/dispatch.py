"""Submit all open jobs of the experiment archive_size to Slurm (run on a login node, e.g. in tmux).

    ARCHIVE_SIZE_DB_FILE=new_jobs.db python -m experiments.archive_size.dispatch
"""
from experiments.archive_size.config import EXPERIMENT
from experiments.common.variants import dispatch_cli

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
