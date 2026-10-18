"""Settings read from environment variables (and the optional ``.env`` file in the repository root).

Variables that are already set in the environment take precedence over ``.env``. See ``.env.example``
for a documented template.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def env_str(name: str, default: str) -> str:
    """Read a string variable; unset or empty values give ``default``.

    Args:
        name: Variable name.
        default: Fallback value.

    Returns:
        The value of the variable or ``default``.
    """
    return os.getenv(name) or default


def env_int(name: str, default: int) -> int:
    """Read an integer variable; unset or empty values give ``default``.

    Args:
        name: Variable name.
        default: Fallback value.

    Returns:
        The parsed value or ``default``.
    """
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


SEEDS = ["1943746817", "1394004965", "3030801653"]
"""Seeds of the three runs per prompt (seed A, B, C); identical across all experiments."""

RESULTS_ROOT = Path(env_str("RESULTS_ROOT", "/work/dldevel/public/Evolutionary_Novelty_Enhancement_Results"))
"""Root directory of the result directories (candidate tables and archives with images, embeddings and noise)."""

DATA_DIR = REPO_ROOT / "data"
"""Data in the repository: prompt tables, job databases and the candidate tables of all analysed runs."""


def db_file(experiment: str) -> Path:
    """Job database of an experiment: ``<EXPERIMENT>_DB_FILE``, by default ``data/jobs/<experiment>.db``.

    The databases in ``data/jobs`` list the analysed runs (status ``DONE``); new jobs are written to a new
    database (``create_jobs --db``) that ``<EXPERIMENT>_DB_FILE`` points to.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        Path of the SQLite database.
    """
    return Path(env_str(f"{experiment.upper()}_DB_FILE", str(DATA_DIR / "jobs" / f"{experiment}.db")))


def result_dir(experiment: str) -> Path:
    """Result directory of an experiment: ``<EXPERIMENT>_RESULT_DIR``, by default ``<RESULTS_ROOT>/<experiment>``.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        Directory that holds (and receives) the run directories.
    """
    return Path(env_str(f"{experiment.upper()}_RESULT_DIR", str(RESULTS_ROOT / experiment)))


def table_dir(experiment: str) -> Path:
    """Copy of the candidate tables and run configurations of an experiment in the repository.

    The layout below this directory is that of :func:`result_dir`; the archives (``<run>.zip``) exist only
    in the result directory.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        ``data/results/<experiment>``.
    """
    return DATA_DIR / "results" / experiment


HF_CACHE = env_str("HF_CACHE", "")
"""Hugging Face cache directory passed as ``cache_dir`` to every ``from_pretrained`` call."""

DEVICE = env_str("DEVICE", "cuda")

POPULATION_SIZE = env_int("POPULATION_SIZE", 60)
NUM_GENERATIONS = env_int("NUM_GENERATIONS", 30)
