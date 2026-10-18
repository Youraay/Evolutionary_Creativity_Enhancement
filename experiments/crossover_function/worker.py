"""Run one job of the experiment crossover_function (called by ``submit_job.sh``).

    python -m experiments.crossover_function.worker <job_id>
"""
from experiments.common.variants import worker_cli
from experiments.crossover_function.config import EXPERIMENT

if __name__ == "__main__":
    worker_cli(EXPERIMENT)
