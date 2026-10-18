"""Selection of the runs of the experiments with population 100 (``experiments.common.variants``).

A run configuration is a dict with ``path`` (run directory with ``<name>.csv`` below
:func:`analysis.common.paths.tables`), ``name`` (variant or prompt), ``seed`` (label of the run within its
name) and ``color``. All directory listings are sorted, so the selection does not depend on the file system.
"""
from pathlib import Path

import pandas as pd

from analysis.common.paths import tables
from analysis.common.style import COLOR_MAP
from experiments.common.variants import RUN_PATTERN

TRANSFER_SEEDS = ["Seed A", "Seed B", "Seed C"]


def find(directory: Path, pattern: str) -> list[Path]:
    """Sorted, case-insensitive glob.

    Args:
        directory: Directory to search.
        pattern: Glob pattern.

    Returns:
        Matching paths in sorted order.
    """
    return sorted(Path(directory).glob(pattern, case_sensitive=False))


def runs_by_name(directory: Path, names: list[str], colors: list[str]) -> list[dict]:
    """One configuration per run directory ``<name>_*`` (spaces as underscores); runs numbered per name.

    Args:
        directory: Result directory of the experiment.
        names: Prompt or variant names.
        colors: Colour per name.

    Returns:
        Run configurations in the order of ``names``.
    """
    configs = []
    for name, color in zip(names, colors):
        for k, path in enumerate(find(directory, f"{name.replace(' ', '_')}_*")):
            configs.append({"path": path, "name": name, "seed": f"Seed {k + 1}", "color": color})
    return configs


def runs_by_substring(directory: Path, names: list[str], seeds: list[str], color: str = "#000") -> list[dict]:
    """Configurations for the supplementary comparisons: every directory ``<name>_*`` belongs to the first
    name it contains, and its seed label is found in the directory name (``"Seed 4"`` -> ``Seed_4``).

    Note that ``<name>_*`` also matches variants with a longer name, e.g. ``1Point_*`` includes the
    ``1Point_Pixel_*`` runs; these are pooled with the plain runs, as in the published figures.

    Args:
        directory: Result directory of the comparison.
        names: Variant names.
        seeds: Seed labels as they appear in the directory names (with a space).
        color: Colour of all variants.

    Returns:
        Run configurations; directories without a known name or seed are skipped.
    """
    configs = []
    for name in names:
        for path in find(directory, f"{name.replace(' ', '_')}_*"):
            match = next((n for n in names if n in path.name), None)
            seed = next((s for s in seeds if s.replace(" ", "_") in path.name), None)
            if match is not None and seed is not None:
                configs.append({"path": path, "name": match, "seed": seed, "color": color})
    return configs


def variant_runs(experiment: str, variant: str, color: str = "#000000") -> list[dict]:
    """Runs of exactly one variant (``<variant>_Seed_<label>[_<job>_<line>]``), sorted by seed label.

    Unlike :func:`runs_by_name`, variants whose name starts with ``variant`` (e.g. ``1Point_Pixel`` for
    ``1Point``) are not included.

    Args:
        experiment: Package name below ``experiments``.
        variant: Variant name (case-insensitive).
        color: Colour of the runs.

    Returns:
        Run configurations with ``seed`` ``"Seed <label>"``.
    """
    configs = []
    for path in find(tables(experiment), f"{variant}_Seed_*"):
        match = RUN_PATTERN.match(path.name)
        if match and match["name"].lower() == variant.lower():
            configs.append({"path": path, "name": variant, "seed": f"Seed {match['label']}", "color": color})
    return sorted(configs, key=lambda c: int(c["seed"].split()[-1]))


def transfer_runs(sources: list[str], targets: list[str]) -> list[dict]:
    """Pair every transfer run (source prompt -> target prompt) with an original run of the target prompt.

    Transfer runs ``<source>_*_<target>`` (``seed_transfer``) and runs optimised for the target prompt
    ``<target>_*`` (``example_prompts``) are paired in sorted order, at most three pairs per transfer.

    Args:
        sources: Source prompts.
        targets: Target prompts.

    Returns:
        Configurations with ``path`` (transfer run), ``original_path``, ``seed``, ``source``, ``target``,
        ``source color``, ``target color`` and ``name``.
    """
    transfer_dir, original_dir = tables("seed_transfer"), tables("example_prompts")
    configs = []
    for source in sources:
        for target in targets:
            transfer_paths = find(transfer_dir, f"{source.replace(' ', '_')}_*_{target.replace(' ', '_')}")
            original_paths = find(original_dir, f"{target.replace(' ', '_')}_*")
            for seed, (path, original) in zip(TRANSFER_SEEDS, zip(transfer_paths, original_paths)):
                configs.append({"path": path, "original_path": original, "seed": seed, "source": source,
                                "target": target, "source color": COLOR_MAP.get(source, "blue"),
                                "target color": COLOR_MAP.get(target, "orange"), "name": f"{source} -> {target}"})
    return configs


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give candidate tables written by ``experiments.common.variants.run_job`` the column names of the
    registered runs.

    The registered runs store ``global_score`` (distance to the mean of generation 0), ``diversity_score``
    (mean pairwise distance of the population, the same value for all candidates) and ``prompt_fidelty``.
    The worker records ``model_LocalMaxMeanDivergence``, the per-candidate ``model_DiversityWithinPopulation``
    (its mean over a generation is the population diversity) and ``model_CLIPScore``, which becomes
    ``clip_score``. Tables of the registered runs are returned unchanged.

    Args:
        df: Candidate table of one run.

    Returns:
        The table with the columns of the original runs added where possible.
    """
    if "global_score" in df.columns or "model_LocalMaxMeanDivergence" not in df.columns:
        return df
    return df.assign(global_score=df["model_LocalMaxMeanDivergence"],
                     diversity_score=df.groupby("generation")["model_DiversityWithinPopulation"].transform("mean"),
                     clip_score=df["model_CLIPScore"])


def read_run(config: dict, key: str = "path") -> pd.DataFrame:
    """Read the candidate table ``<run>/<run>.csv`` of a configuration (see :func:`normalize_columns`).

    Args:
        config: Run configuration.
        key: Key of the run directory (``"path"`` or ``"original_path"``).

    Returns:
        The candidate table.
    """
    path = config[key]
    return normalize_columns(pd.read_csv(path / f"{path.name}.csv"))
