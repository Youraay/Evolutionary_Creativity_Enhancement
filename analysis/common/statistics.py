"""Statistical tests shared by the figures: generation 0 vs. a later generation, Holm correction, star labels."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values.

    Args:
        p: Raw p-values.

    Returns:
        Adjusted p-values (capped at 1), in the order of ``p``.
    """
    p = np.asarray(p, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def stars(p: float) -> str:
    """Significance label: ``***`` < 0.001, ``**`` < 0.01, ``*`` < 0.05, else ``n.s.``."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def generation_test(df: pd.DataFrame, metric: str, first: int = 0, last: int | None = None) -> dict:
    """Two-sided Mann-Whitney U test of all candidates of generation ``first`` vs. ``last``, runs pooled.

    The statistic is that of the later generation (``U = n_first * n_last`` if every later value is higher), as
    reported in the figure legends of the manuscript.

    Args:
        df: Candidate table(s) with ``generation`` and ``metric``.
        metric: Column to compare.
        first: Earlier generation.
        last: Later generation; by default the last generation of ``df``.

    Returns:
        Dict with the generations, sample sizes, medians, means, ``U``, rank-biserial ``effect_r`` (> 0: later
        generation higher) and ``p``.
    """
    last = int(df["generation"].max()) if last is None else last
    a = df.loc[df["generation"] == first, metric].dropna()
    b = df.loc[df["generation"] == last, metric].dropna()
    u, p = mannwhitneyu(b, a, alternative="two-sided")
    return {"metric": metric, "gen_first": first, "gen_last": last, "n_first": len(a), "n_last": len(b),
            "median_first": a.median(), "median_last": b.median(), "mean_first": a.mean(), "mean_last": b.mean(),
            "U": u, "effect_r": 2 * u / (len(a) * len(b)) - 1, "p": p}


def group_tests(groups: dict[str, pd.DataFrame], metrics: list[str], first: int = 0, last: int = 30) -> pd.DataFrame:
    """:func:`generation_test` for every group (e.g. variant) and metric, with Holm correction over the table.

    Args:
        groups: Name -> candidates of all runs of the group.
        metrics: Columns to test.
        first: Earlier generation.
        last: Later generation.

    Returns:
        One row per group and metric with ``p_holm`` and ``sig``.
    """
    table = pd.DataFrame([{"group": name, **generation_test(df, metric, first, last)}
                          for name, df in groups.items() for metric in metrics])
    table["p_holm"] = holm(table["p"])
    table["sig"] = table["p_holm"].map(stars)
    return table
