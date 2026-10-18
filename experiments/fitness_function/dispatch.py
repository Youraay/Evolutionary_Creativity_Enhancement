"""Submit all open jobs of the experiment fitness_function to Slurm (run on a login node, e.g. in tmux).

    FITNESS_FUNCTION_DB_FILE=new_jobs.db python -m experiments.fitness_function.dispatch
"""
from experiments.common.variants import dispatch_cli
from experiments.fitness_function.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
