"""Run one job of the experiment mutation_only (called by ``submit_job.sh``).

    python -m experiments.mutation_only.worker <job_id>
"""
from experiments.common.variants import worker_cli
from experiments.mutation_only.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
