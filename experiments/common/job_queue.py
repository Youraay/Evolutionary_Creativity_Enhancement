"""SQLite job database of an experiment: one row per run.

``create_jobs`` adds rows with status ``OPEN``; the worker of the experiment runs one row by its id and marks it
``RUNNING`` and then ``DONE`` or ``FAILED``. Runs started otherwise are registered as ``DONE``
(:meth:`JobQueue.register_finished`).
"""
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone

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
"""Bookkeeping columns present in every job table (``job_id``: id of the batch job that ran the run, if known;
``claimed_by``, ``claimed_at`` and ``retry_count`` are kept for the job databases of the analysed runs)."""


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

    def mark_running(self, job_id: int) -> None:
        """Mark a job as started."""
        self._execute("UPDATE jobs SET status = 'RUNNING', started_at = ? WHERE id = ?", (utc_now(), job_id))

    def mark_done(self, job_id: int, output_path: str | None = None) -> None:
        """Mark a job as finished and store its result directory."""
        self._execute("UPDATE jobs SET status = 'DONE', output_path = ?, finished_at = ?, last_error = NULL "
                      "WHERE id = ?", (output_path, utc_now(), job_id))

    def register_finished(self, job_id: int, output_path: str, batch_job_id: str | None, finished_at: str) -> None:
        """Mark a job as done that ran outside the worker (e.g. a run started with its own script).

        Args:
            job_id: Row id.
            output_path: Result directory of the run.
            batch_job_id: Id of the batch job of the run, if known.
            finished_at: ISO timestamp (UTC) of the end of the run.
        """
        self._execute("UPDATE jobs SET status = 'DONE', output_path = ?, job_id = ?, finished_at = ?, last_error = NULL "
                      "WHERE id = ?", (output_path, batch_job_id, finished_at, job_id))

    def mark_failed(self, job_id: int, error_message: str) -> None:
        """Mark a job as failed; the message is truncated to 2000 characters."""
        self._execute("UPDATE jobs SET status = 'FAILED', last_error = ?, finished_at = ? WHERE id = ?",
                      (error_message[:2000], utc_now(), job_id))
