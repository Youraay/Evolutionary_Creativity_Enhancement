"""Submit all open jobs of the experiment fitness_embedding to Slurm (run on a login node, e.g. in tmux).

    FITNESS_EMBEDDING_DB_FILE=new_jobs.db python -m experiments.fitness_embedding.dispatch
"""
from experiments.common.embedding_spaces import dispatch_cli
from experiments.fitness_embedding.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
