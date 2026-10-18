"""Submit all open jobs of the experiment mutation_only to Slurm (run on a login node, e.g. in tmux).

    MUTATION_ONLY_DB_FILE=new_jobs.db python -m experiments.mutation_only.dispatch
"""
from experiments.common.variants import dispatch_cli
from experiments.mutation_only.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
