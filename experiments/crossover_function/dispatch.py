"""Submit all open jobs of the experiment crossover_function to Slurm (run on a login node, e.g. in tmux).

    CROSSOVER_FUNCTION_DB_FILE=new_jobs.db python -m experiments.crossover_function.dispatch
"""
from experiments.common.variants import dispatch_cli
from experiments.crossover_function.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
