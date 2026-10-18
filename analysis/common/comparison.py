"""Figures shared by the comparisons of GA variants on "a cat" (archive size, crossover, fitness function, ablation).

For an experiment the output in ``plots/<experiment>/`` is

    runs        one row per variant: distance to the mean of generation 0 of every run (violins of generations
                0/10/20/30) and the diversity within the generation (mean +- SEM over the runs)
    overview    one row per variant: violins of the first run, diversity and CLIP score with the random
                baseline of "a cat" (only for experiments with CLIP-score tables)
    tests.csv   Mann-Whitney U tests of generation 0 vs. 30 per variant (runs pooled), Holm-corrected
"""
import pandas as pd

from analysis.common.overview import render_comparison_overview
from analysis.common.paths import PLOTS_DIR, tables
from analysis.common.runs import read_run, runs_by_substring, variant_runs
from analysis.common.statistics import group_tests
from analysis.common.style import reset_style
from analysis.common.violins import render_run_violins

CLIP_BASELINE_A_CAT = tables("example_prompts") / "clip_scores" / "baseline_scores_a_cat.csv"
"""Random CLIP-score baseline of "a cat" (100 images of the other example prompts)."""
OVERVIEW_LIMITS = dict(yscale_viol=[0.0, 0.3], ylims_viol=(0.0, 0.3), yscale_div=[0.1, 0.25], ylims_div=(0.1, 0.25),
                       yscale_clip=[0.4, 0.8], ylims_clip=(0.4, 0.8))


def render_comparison(experiment: str, variants: dict[str, str], overview: dict[str, str] | None = None,
                      overview_seeds: list[str] | None = None, ylims=(0.0, 0.3)) -> pd.DataFrame:
    """Run violins, CLIP-score overview and tests of one comparison.

    Args:
        experiment: Package name below ``experiments``.
        variants: Variant -> row title of the run violins, in the order of the rows.
        overview: Variant -> row title of the overview; no overview if ``None``.
        overview_seeds: Seed labels of the overview runs (``"Seed 1"`` ...).
        ylims: y limits (and ticks) of the run violins.

    Returns:
        The tests (also written to ``tests.csv``).
    """
    out = PLOTS_DIR / experiment
    reset_style()
    rows = [{"title": title, "configs": variant_runs(experiment, name)} for name, title in variants.items()]
    render_run_violins(out / "runs", rows, ylims=ylims, yticks=ylims)
    if overview:
        reset_style()
        names = list(overview)
        render_comparison_overview(out / "overview", runs_by_substring(tables(experiment), names, overview_seeds),
                                   names, tables(experiment) / "clip_scores", CLIP_BASELINE_A_CAT,
                                   display_names=overview, **OVERVIEW_LIMITS)
    groups = {row["title"]: pd.concat([read_run(c) for c in row["configs"]], ignore_index=True) for row in rows}
    tests = group_tests(groups, ["global_score", "diversity_score"])
    tests.to_csv(out / "tests.csv", index=False)
    return tests
