"""Run one job of the experiment example_prompts.

    python -m experiments.example_prompts.worker <job_id>
"""
from experiments.common.variants import worker_cli
from experiments.example_prompts.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
