"""Submit all open jobs of the experiment long_runs to Slurm (run on a login node, e.g. in tmux).

    LONG_RUNS_DB_FILE=new_jobs.db python -m experiments.long_runs.dispatch
"""
from experiments.common.embedding_spaces import dispatch_cli
from experiments.long_runs.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
