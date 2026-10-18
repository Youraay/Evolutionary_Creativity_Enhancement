"""Submits the jobs of a job database to Slurm, one batch job per run.

The dispatcher runs on a login node (e.g. in tmux) until no unfinished job is left. In every loop it

1. reopens jobs that were claimed but never submitted, and running jobs whose Slurm job disappeared,
2. waits while the user already has ``MAX_RUNNING_JOBS`` pending or running Slurm jobs,
3. claims the next open job and submits it with ``sbatch <submit_script> <job id>``.

Several dispatchers (one per experiment) can run at the same time; the job limit counts all Slurm jobs
of the user.
"""
import logging
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from experiments.common import settings
from experiments.common.job_queue import JobQueue

logger = logging.getLogger(__name__)

LIMIT_KEYWORDS = ("qos", "quota", "association", "job violates accounting/qos policy", "maximum number", "limit",
                  "too many jobs")
"""Substrings of ``sbatch`` errors that indicate a scheduler limit rather than a broken job."""


def _squeue(*options: str) -> list[str]:
    """Return the non-empty output lines of ``squeue`` for the pending and running jobs of the user.

    Args:
        options: Additional ``squeue`` options.

    Returns:
        Output lines; an empty list if ``squeue`` fails.
    """
    user = os.environ.get("USER")
    if not user:
        return []
    try:
        result = subprocess.run(["squeue", "-h", "-u", user, "-t", "PENDING,RUNNING", *options],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("squeue failed: %s", error)
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def count_active_slurm_jobs() -> int:
    """Number of pending or running Slurm jobs of the current user."""
    return len(_squeue())


def active_slurm_job_ids() -> set[str]:
    """Ids of the pending or running Slurm jobs of the current user."""
    return set(_squeue("-o", "%A"))


def is_limit_error(message: str) -> bool:
    """Whether an ``sbatch`` error message points to a scheduler limit.

    Args:
        message: Error output of ``sbatch``.

    Returns:
        ``True`` if the submission should simply be retried later.
    """
    text = message.lower()
    return any(keyword in text for keyword in LIMIT_KEYWORDS)


def _parse_utc(timestamp: str | None) -> datetime | None:
    """Parse an ISO timestamp from the job database as UTC."""
    if not timestamp:
        return None
    parsed = datetime.fromisoformat(timestamp)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class Dispatcher:
    """Submits the open jobs of one job database to Slurm.

    Attributes:
        queue: Job database of the experiment.
        submit_script: Slurm batch script called as ``sbatch <script> <job id>``.
        time_limit: Optional function returning an ``sbatch --time`` value for a job (``None`` keeps the
            limit of the batch script).
        describe: Optional function returning a short label of a job for the log.
    """

    def __init__(self, queue: JobQueue, submit_script: str,
                 time_limit: Callable[[sqlite3.Row], str | None] | None = None,
                 describe: Callable[[sqlite3.Row], str] | None = None) -> None:
        """
        Args:
            queue: Job database of the experiment.
            submit_script: Slurm batch script, relative to the repository root or absolute.
            time_limit: Optional per-job time limit.
            describe: Optional per-job label for the log.
        """
        self.queue = queue
        self.submit_script = submit_script
        self.time_limit = time_limit or (lambda job: None)
        self.describe = describe or (lambda job: "")
        self.worker_name = os.environ.get("USER", "unknown")

    def submit(self, job: sqlite3.Row) -> str:
        """Submit one job with ``sbatch``.

        Args:
            job: Claimed job.

        Returns:
            The Slurm job id.

        Raises:
            RuntimeError: If ``sbatch`` fails or its output contains no job id.
        """
        command = ["sbatch", "--parsable", *shlex.split(settings.SBATCH_EXTRA_ARGS)]
        limit = self.time_limit(job)
        if limit:
            command.append(f"--time={limit}")
        command += [self.submit_script, str(job["id"])]
        result = subprocess.run(command, capture_output=True, text=True, cwd=settings.REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        match = re.search(r"^(\d+)", result.stdout.strip())
        if not match:
            raise RuntimeError(f"Could not parse the Slurm job id from: {result.stdout.strip()}")
        return match.group(1)

    def recover_stale_running_jobs(self) -> int:
        """Reopen running jobs whose Slurm job is gone (after ``STALE_RUNNING_MINUTES``).

        Returns:
            Number of reopened jobs.
        """
        active = active_slurm_job_ids()
        now = datetime.now(timezone.utc)
        recovered = 0
        for job in self.queue.get_running_jobs():
            slurm_id = str(job["job_id"]) if job["job_id"] is not None else None
            started = _parse_utc(job["started_at"])
            if not slurm_id:
                self.queue.requeue_stale_running_job(job["id"], "RUNNING job without Slurm job_id")
                recovered += 1
            elif slurm_id not in active and not (
                    started and now - started < timedelta(minutes=settings.STALE_RUNNING_MINUTES)):
                self.queue.requeue_stale_running_job(job["id"], f"Slurm job {slurm_id} no longer active")
                recovered += 1
        return recovered

    def run(self) -> None:
        """Submit jobs until no unfinished job is left."""
        logger.info("Dispatcher started as %s, job database %s", self.worker_name, self.queue.db_file)
        while True:
            reset = self.queue.reset_stale_claimed_jobs(settings.STALE_CLAIM_MINUTES)
            if reset:
                logger.info("%d stale CLAIMED jobs reopened", reset)
            recovered = self.recover_stale_running_jobs()
            if recovered:
                logger.info("%d stale RUNNING jobs reopened", recovered)

            active = count_active_slurm_jobs()
            if active >= settings.MAX_RUNNING_JOBS:
                logger.info("Job limit reached (%d/%d), waiting %ds", active, settings.MAX_RUNNING_JOBS,
                            settings.POLL_INTERVAL)
                time.sleep(settings.POLL_INTERVAL)
                continue

            job = self.queue.claim_next_job(self.worker_name)
            if job is None:
                if self.queue.has_unfinished_jobs():
                    logger.info("No job can be claimed, waiting")
                    time.sleep(settings.POLL_INTERVAL)
                    continue
                logger.info("No open jobs left. Status: %s", self.queue.count_jobs_by_status())
                return

            logger.info("Job %s %s claimed, submitting", job["id"], self.describe(job))
            try:
                slurm_id = self.submit(job)
            except (RuntimeError, OSError) as error:
                logger.warning("Submission of job %s failed: %s", job["id"], error)
                self.queue.requeue_job(job["id"], str(error))
                wait = settings.BACKOFF_ON_LIMIT if is_limit_error(str(error)) else settings.POLL_INTERVAL
                time.sleep(wait)
                continue
            self.queue.mark_running(job["id"], slurm_id)
            logger.info("Job %s runs as Slurm job %s", job["id"], slurm_id)


def run_dispatcher(queue: JobQueue, submit_script: str, time_limit: Callable[[sqlite3.Row], str | None] | None = None,
                   describe: Callable[[sqlite3.Row], str] | None = None) -> None:
    """Command line entry point: log to stdout and dispatch until all jobs are submitted.

    Args:
        queue: Job database of the experiment.
        submit_script: Slurm batch script.
        time_limit: Optional per-job time limit.
        describe: Optional per-job label for the log.
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    Dispatcher(queue, submit_script, time_limit, describe).run()
