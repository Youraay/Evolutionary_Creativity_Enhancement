"""Submit all open jobs of the PartiPrompts experiment to Slurm (run on a login node, e.g. in tmux).

    python -m experiments.parti_prompts.dispatch
"""
from experiments.common.dispatcher import run_dispatcher
from experiments.parti_prompts.config import SUBMIT_SCRIPT, job_queue

if __name__ == "__main__":
    run_dispatcher(job_queue(), SUBMIT_SCRIPT, describe=lambda job: f"({job['prompt']!r}, seed {job['seed']})")
