"""Paths, model list and job table of the model comparison.

This module must not import torch or diffusers, so that ``create_jobs`` and ``dispatch`` start quickly.
"""
from pathlib import Path

from experiments.common.job_queue import JobQueue
from experiments.common.settings import DATA_DIR, db_file, result_dir, submit_script

DB_FILE = db_file("model_comparison")
RESULT_DIR = result_dir("model_comparison")
PROMPT_FILE = DATA_DIR / "prompts" / "parti_prompts_selection.tsv"
SUBMIT_SCRIPT = submit_script("model_comparison")

MODEL_NAMES = [
    "sdxl", "sdxl_turbo", "sd15", "sd_turbo", "playground_v25", "ssd_1b", "segmind_vega", "realvisxl_v5",
    "juggernaut_xl_v9", "z_image", "flux2_klein_4b", "flux2_klein_9b_kv", "krea_2_turbo", "flux2_dev",
]
"""Model names as stored in the job database; must match ``models.TEXT_TO_IMAGE_MODELS``."""

TIME_LIMITS = {
    "z_image": "2-00:00:00",
    "flux2_klein_4b": "2-00:00:00",
    "flux2_klein_9b_kv": "2-00:00:00",
    "krea_2_turbo": "2-00:00:00",
    "flux2_dev": "5-00:00:00",
}
"""Slurm time limits that replace the 15 h of the batch script for slower models (gpu2 allows 5 days)."""

COLUMNS = [
    ("prompt", "TEXT NOT NULL"),
    ("seed", "TEXT NOT NULL"),
    ("model", "TEXT NOT NULL"),
    ("category", "TEXT"),
    ("challenge", "TEXT"),
    ("note", "TEXT"),
]


def job_queue(db_file: Path = DB_FILE) -> JobQueue:
    """Return the job database of the model comparison.

    Args:
        db_file: Path of the SQLite database.
    """
    return JobQueue(str(db_file), COLUMNS)
