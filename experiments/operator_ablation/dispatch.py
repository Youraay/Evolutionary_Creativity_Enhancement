"""Submit all open jobs of the experiment operator_ablation to Slurm (run on a login node, e.g. in tmux).

    OPERATOR_ABLATION_DB_FILE=new_jobs.db python -m experiments.operator_ablation.dispatch
"""
from experiments.common.variants import dispatch_cli
from experiments.operator_ablation.config import EXPERIMENT

if __name__ == "__main__":
    dispatch_cli(EXPERIMENT)
