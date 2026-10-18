"""Submit all open jobs of the experiment seed_transfer to Slurm (run on a login node, e.g. in tmux).

    SEED_TRANSFER_DB_FILE=new_jobs.db python -m experiments.seed_transfer.dispatch
"""
from experiments.common.variants import dispatch_cli
from experiments.seed_transfer.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
