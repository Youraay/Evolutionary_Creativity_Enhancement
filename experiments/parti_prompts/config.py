"""Paths and job table of the PartiPrompts experiment."""
from pathlib import Path

from experiments.common.job_queue import JobQueue
from experiments.common.settings import DATA_DIR, db_file, result_dir

DB_FILE = db_file("parti_prompts")
RESULT_DIR = result_dir("parti_prompts")
PROMPT_FILE = DATA_DIR / "prompts" / "parti_prompts_selection.tsv"

COLUMNS = [
    ("prompt", "TEXT NOT NULL"),
    ("seed", "TEXT NOT NULL"),
    ("category", "TEXT"),
    ("challenge", "TEXT"),
    ("note", "TEXT"),
]


def job_queue(db_file: Path = DB_FILE) -> JobQueue:
    """Return the job database of this experiment.

    Args:
        db_file: Path of the SQLite database.
    """
    return JobQueue(str(db_file), COLUMNS)
