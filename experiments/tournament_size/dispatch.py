"""Submit all open jobs of the experiment tournament_size to Slurm (run on a login node, e.g. in tmux).

    TOURNAMENT_SIZE_DB_FILE=new_jobs.db python -m experiments.tournament_size.dispatch
"""
from experiments.common.embedding_spaces import dispatch_cli
from experiments.tournament_size.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
