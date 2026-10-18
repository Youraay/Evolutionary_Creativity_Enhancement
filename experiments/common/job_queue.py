"""SQLite job queue shared by dispatcher and workers.

Every row describes one run. Its life cycle is::

    OPEN -> CLAIMED (dispatcher) -> RUNNING (submitted to Slurm) -> DONE | FAILED
                \\-> RETRY (submission failed, claim or Slurm job lost) -> CLAIMED ...

Jobs with any other status (e.g. ``HOLD``) are ignored by the dispatcher and can be run manually.
"""
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

STATUS_COLUMNS = """
    status TEXT NOT NULL DEFAULT 'OPEN',
    claimed_by TEXT,
    claimed_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    job_id TEXT,
    last_error TEXT,
    output_path TEXT"""
"""Bookkeeping columns present in every job table (``job_id`` is the Slurm job id)."""

UNFINISHED = ("OPEN", "RETRY", "CLAIMED", "RUNNING")


def utc_now() -> str:
    """Current UTC time as ISO string with second precision (naive, as stored in the database)."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


class JobQueue:
    """Job table of one experiment.

    Attributes:
        db_file: Path of the SQLite database.
        columns: Experiment-specific columns as ``(name, SQL type)`` pairs, e.g. ``("prompt", "TEXT NOT NULL")``.
    """

    def __init__(self, db_file: str, columns: list[tuple[str, str]]) -> None:
        """
        Args:
            db_file: Path of the SQLite database.
            columns: Experiment-specific columns as ``(name, SQL type)`` pairs.
        """
        self.db_file = str(db_file)
        self.columns = columns

    def _connect(self) -> sqlite3.Connection:
        """Open a connection whose rows can be indexed by column name."""
        connection = sqlite3.connect(self.db_file, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _execute(self, sql: str, params: tuple = ()) -> int:
        """Run one write statement and commit it.

        Args:
            sql: SQL statement.
            params: Statement parameters.

        Returns:
            Number of affected rows.
        """
        connection = self._connect()
        try:
            cursor = connection.execute(sql, params)
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def _fetch(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Run one query.

        Args:
            sql: SQL query.
            params: Query parameters.

        Returns:
            All result rows.
        """
        connection = self._connect()
        try:
            return connection.execute(sql, params).fetchall()
        finally:
            connection.close()

    def create_table(self) -> None:
        """Create the ``jobs`` table if it does not exist yet."""
        experiment_columns = ",\n    ".join(f"{name} {sql_type}" for name, sql_type in self.columns)
        self._execute(f"CREATE TABLE IF NOT EXISTS jobs (\n    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                      f"    {experiment_columns},{STATUS_COLUMNS}\n)")

    def add_jobs(self, jobs: Iterable[dict]) -> int:
        """Insert new jobs with status ``OPEN``.

        Args:
            jobs: One dict per job with a value for every experiment-specific column.

        Returns:
            Number of inserted jobs.
        """
        names = [name for name, _ in self.columns]
        connection = self._connect()
        try:
            cursor = connection.executemany(
                f"INSERT INTO jobs ({', '.join(names)}) VALUES ({', '.join(':' + n for n in names)})", list(jobs))
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def last_id(self) -> int:
        """Highest row id (0 for an empty table)."""
        return self._fetch("SELECT COALESCE(MAX(id), 0) AS n FROM jobs")[0]["n"]

    def get_job(self, job_id: int) -> sqlite3.Row | None:
        """Return the row of a job, or ``None`` if the id does not exist."""
        rows = self._fetch("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return rows[0] if rows else None

    def get_running_jobs(self) -> list[sqlite3.Row]:
        """Return all jobs with status ``RUNNING``."""
        return self._fetch("SELECT * FROM jobs WHERE status = 'RUNNING'")

    def claim_next_job(self, worker_name: str) -> sqlite3.Row | None:
        """Atomically claim the open or retried job with the lowest id.

        Args:
            worker_name: Name stored in ``claimed_by``.

        Returns:
            The claimed job, or ``None`` if no job is open or another dispatcher was faster.
        """
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM jobs WHERE status IN ('OPEN', 'RETRY') ORDER BY id ASC LIMIT 1").fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute("UPDATE jobs SET status = 'CLAIMED', claimed_by = ?, claimed_at = ? "
                               "WHERE id = ? AND status IN ('OPEN', 'RETRY')", (worker_name, utc_now(), row["id"]))
            updated = connection.execute("SELECT changes()").fetchone()[0]
            connection.commit()
        finally:
            connection.close()
        return self.get_job(row["id"]) if updated == 1 else None

    def mark_running(self, job_id: int, slurm_job_id: str) -> None:
        """Record the Slurm job id after a successful submission."""
        self._execute("UPDATE jobs SET status = 'RUNNING', job_id = ?, started_at = ? WHERE id = ?",
                      (str(slurm_job_id), utc_now(), job_id))

    def mark_done(self, job_id: int, output_path: str | None = None) -> None:
        """Mark a job as finished and store its result directory."""
        self._execute("UPDATE jobs SET status = 'DONE', output_path = ?, finished_at = ?, last_error = NULL "
                      "WHERE id = ?", (output_path, utc_now(), job_id))

    def register_finished(self, job_id: int, output_path: str, slurm_job_id: str | None, finished_at: str) -> None:
        """Mark a job as done that ran outside the dispatcher (e.g. a run started with its own script).

        Args:
            job_id: Row id.
            output_path: Result directory of the run.
            slurm_job_id: Slurm job id of the run, if known.
            finished_at: ISO timestamp (UTC) of the end of the run.
        """
        self._execute("UPDATE jobs SET status = 'DONE', output_path = ?, job_id = ?, finished_at = ?, last_error = NULL "
                      "WHERE id = ?", (output_path, slurm_job_id, finished_at, job_id))

    def mark_failed(self, job_id: int, error_message: str) -> None:
        """Mark a job as failed; the message is truncated to 2000 characters."""
        self._execute("UPDATE jobs SET status = 'FAILED', last_error = ?, finished_at = ? WHERE id = ?",
                      (error_message[:2000], utc_now(), job_id))

    def requeue_job(self, job_id: int, error_message: str) -> None:
        """Reopen a claimed job whose submission failed."""
        self._execute("UPDATE jobs SET status = 'RETRY', last_error = ?, claimed_by = NULL, claimed_at = NULL, "
                      "job_id = NULL, retry_count = retry_count + 1 WHERE id = ?", (error_message[:2000], job_id))

    def reset_stale_claimed_jobs(self, max_age_minutes: int = 60) -> int:
        """Reopen jobs that were claimed longer than ``max_age_minutes`` ago but never submitted.

        Returns:
            Number of reopened jobs.
        """
        cutoff = (datetime.now(timezone.utc).replace(tzinfo=None)
                  - timedelta(minutes=max_age_minutes)).isoformat(timespec="seconds")
        return self._execute("UPDATE jobs SET status = 'RETRY', last_error = 'Claim timed out', claimed_by = NULL, "
                             "claimed_at = NULL, retry_count = retry_count + 1 "
                             "WHERE status = 'CLAIMED' AND claimed_at IS NOT NULL AND claimed_at < ?", (cutoff,))

    def requeue_stale_running_job(self, job_id: int, error_message: str) -> None:
        """Reopen a running job whose Slurm job no longer exists."""
        self._execute("UPDATE jobs SET status = 'RETRY', last_error = ?, job_id = NULL, retry_count = retry_count + 1 "
                      "WHERE id = ? AND status = 'RUNNING'", (error_message[:2000], job_id))

    def has_unfinished_jobs(self) -> bool:
        """Whether any job is open, retried, claimed or running."""
        placeholders = ", ".join("?" for _ in UNFINISHED)
        return bool(self._fetch(f"SELECT 1 FROM jobs WHERE status IN ({placeholders}) LIMIT 1", UNFINISHED))

    def count_jobs_by_status(self) -> dict[str, int]:
        """Return the number of jobs per status."""
        return {row["status"]: row["n"] for row in self._fetch("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")}
