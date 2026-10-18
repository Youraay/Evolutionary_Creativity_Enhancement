"""Input and output locations of the analysis.

Candidate tables and run configurations are read from the repository (``data/results/<experiment>``); images,
embeddings and noise come from the run archives in the result directories (``<EXPERIMENT>_RESULT_DIR``, see
``experiments.common.settings``), which have the same layout.
"""
from pathlib import Path

from experiments.common.settings import HF_CACHE as _HF_CACHE
from experiments.common.settings import DATA_DIR, REPO_ROOT, RESULTS_ROOT, env_str, result_dir, table_dir

PLOTS_DIR = Path(env_str("PLOTS_DIR", str(REPO_ROOT / "plots")))
"""Root of all generated figures and tables (one directory per experiment)."""

PROMPT_TABLE = DATA_DIR / "prompts" / "parti_prompts_selection.tsv"
"""PartiPrompts selection with the columns Prompt, Category, Challenge and Note."""

PCA_DATA = Path(env_str("PCA_DATA_DIR", str(RESULTS_ROOT / "pca_parti")))
"""Projections of the PartiPrompts PCA (too large for the repository)."""

HF_CACHE = Path(_HF_CACHE or "/work/dldevel/public/Evolutionary_Diffusion_Enhancement/huggingface")
"""Hugging Face cache with the CLIP model used for the random baselines."""


def tables(experiment: str) -> Path:
    """Directory of the candidate tables of an experiment in the repository.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        ``data/results/<experiment>``.
    """
    return table_dir(experiment)


def archive(run_dir: Path, experiment: str) -> Path:
    """Archive (images, embeddings, noise) of a run whose tables are in ``run_dir``.

    Args:
        run_dir: Run directory below :func:`tables`.
        experiment: Package name below ``experiments``.

    Returns:
        ``<result dir>/<relative run path>/<run>.zip``.
    """
    return result_dir(experiment) / run_dir.relative_to(table_dir(experiment)) / f"{run_dir.name}.zip"


__all__ = ["PLOTS_DIR", "PROMPT_TABLE", "PCA_DATA", "HF_CACHE", "REPO_ROOT", "RESULTS_ROOT", "DATA_DIR", "tables",
           "archive"]
