"""Submit all open jobs of the experiment example_prompts to Slurm (run on a login node, e.g. in tmux).

    EXAMPLE_PROMPTS_DB_FILE=new_jobs.db python -m experiments.example_prompts.dispatch
"""
from experiments.common.variants import dispatch_cli
from experiments.example_prompts.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
