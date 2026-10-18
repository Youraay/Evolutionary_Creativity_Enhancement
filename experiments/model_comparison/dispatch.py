"""Submit all open jobs of the model comparison to Slurm (run on a login node, e.g. in tmux).

    python -m experiments.model_comparison.dispatch
"""
from experiments.common.dispatcher import run_dispatcher
from experiments.model_comparison.config import SUBMIT_SCRIPT, TIME_LIMITS, job_queue

if __name__ == "__main__":
    run_dispatcher(job_queue(), SUBMIT_SCRIPT, time_limit=lambda job: TIME_LIMITS.get(job["model"]),
                   describe=lambda job: f"({job['model']})")
