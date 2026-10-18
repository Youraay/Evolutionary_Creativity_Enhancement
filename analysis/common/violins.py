"""Violins of the distance to the initial generation over generations 0, 10, 20 and 30.

``render_pooled_violins``: one panel per prompt or variant, all runs pooled, with the Mann-Whitney U test of the
first vs. the last generation as bracket (Fig. 2C, Supplementary Fig. 9B).
``render_run_violins``: one row per variant, one panel per run plus the diversity within the generation
(mean +- SEM over the runs) (Supplementary Figs. 4, 6, 7).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.common.runs import read_run
from analysis.common.statistics import generation_test, stars
from analysis.common.style import (BOTTOM_MARGIN, ERRORBAR, FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, RIGHT_MARGIN,
                                   TICK_SIZE, TITLE_SIZE, TOP_MARGIN, save_figure, seed_means)

GENERATIONS = [0, 10, 20, 30]
DISTANCE_LABEL = "distance to mean of\ninitial generation [a.u.]"
DIVERSITY_LABEL = "diversity within\ngeneration [a.u.]"


def violins(ax: plt.Axes, df: pd.DataFrame, color: str, metric: str = "global_score", gens=GENERATIONS,
            dot_size: float = 1.5) -> None:
    """Violins (transparent body, coloured outline) and points of ``metric`` for the given generations.

    Args:
        ax: Target axes.
        df: Candidates of one or more runs.
        color: Colour of bodies, outlines and points.
        metric: Column to show.
        gens: Generations, one violin each.
        dot_size: Marker size of the candidates.
    """
    data = df[df["generation"].isin(gens)]
    body = sns.violinplot(data=data, x="generation", y=metric, order=gens, ax=ax, inner=None, color=color,
                          alpha=0.3, cut=0, density_norm="width")
    sns.stripplot(data=data, x="generation", y=metric, order=gens, ax=ax, size=dot_size, color=color, alpha=0.8,
                  jitter=True)
    for collection in body.collections:
        collection.set_edgecolor(color)
        collection.set_linewidth(1)


def significance_bracket(ax: plt.Axes, x0: float, x1: float, y: float, label: str, color: str = "black") -> None:
    """Bracket from ``x0`` to ``x1`` at height ``y`` (data coordinates) with a label above.

    Args:
        ax: Target axes.
        x0: Left end.
        x1: Right end.
        y: Height of the bracket.
        label: Text above the bracket, e.g. ``"***"``.
        color: Colour of bracket and text.
    """
    tick = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=color, linewidth=0.8, clip_on=False)
    ax.text((x0 + x1) / 2, y, label, ha="center", va="bottom", fontsize=TICK_SIZE, color=color)


def render_pooled_violins(path: Path, panels: list[dict], metric: str = "global_score", ylims=(0.0, 0.4),
                          yticks=(0.0, 0.4), ylabel: str = DISTANCE_LABEL, nrows: int = 1) -> pd.DataFrame:
    """One panel per entry of ``panels``: violins of all runs pooled, bracket = test of generation 0 vs. 30.

    Args:
        path: Output path without extension; the test results are written to ``<path>_tests.csv``.
        panels: Dicts with ``title``, ``configs`` (runs, see :mod:`analysis.common.runs`) and ``color``.
        metric: Column to show.
        ylims: y limits.
        yticks: y ticks.
        ylabel: Label of the y axes.
        nrows: Number of rows; the panels fill the rows first.

    Returns:
        Table of the tests (one row per panel, see :func:`analysis.common.statistics.generation_test`).
    """
    ncols = -(-len(panels) // nrows)
    total_w = ncols * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN
    total_h = nrows * FIXED_COL_WIDTH * 0.7 + TOP_MARGIN + BOTTOM_MARGIN
    fig, axes = plt.subplots(nrows, ncols, figsize=(total_w, total_h), squeeze=False, sharey=True,
                             layout="constrained")
    tests = []
    for i, panel in enumerate(panels):
        ax = axes[i // ncols, i % ncols]
        data = pd.concat([read_run(config) for config in panel["configs"]], ignore_index=True)
        violins(ax, data, panel["color"], metric)
        ax.set_ylim(ylims[0], ylims[1] * 1.12)
        ax.set_yticks(yticks)
        result = generation_test(data, metric, GENERATIONS[0], GENERATIONS[-1])
        tests.append({"panel": panel["title"], "runs": len(panel["configs"]), **result})
        significance_bracket(ax, 0, len(GENERATIONS) - 1, ylims[1] * 1.04, stars(result["p"]))
        ax.set_title(panel["title"], fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.set_xlabel("Generation" if i // ncols == nrows - 1 else "", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel if i % ncols == 0 and nrows == 1 else "", fontsize=LABEL_SIZE)
    for i in range(len(panels), nrows * ncols):
        fig.delaxes(axes[i // ncols, i % ncols])
    if nrows > 1:
        fig.supylabel(ylabel, fontsize=LABEL_SIZE)
    save_figure(fig, path)
    table = pd.DataFrame(tests)
    table.to_csv(Path(path).with_name(f"{Path(path).name}_tests.csv"), index=False)
    return table


def render_run_violins(path: Path, rows: list[dict], color: str = "#000000", ylims=(0.0, 0.3), yticks=(0.0, 0.3),
                       diversity_ylims=(0.1, 0.25), diversity_yticks=(0.1, 0.25)) -> None:
    """One row per variant: violins of every run and the diversity within the generation (mean +- SEM).

    Args:
        path: Output path without extension.
        rows: Dicts with ``title`` and ``configs`` (the runs of one variant, sorted by seed).
        color: Colour of violins, points and curves.
        ylims: y limits of the violins.
        yticks: y ticks of the violins.
        diversity_ylims: y limits of the diversity.
        diversity_yticks: y ticks of the diversity.
    """
    nruns = max(len(row["configs"]) for row in rows)
    ncols = nruns + 1
    total_w = ncols * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN
    total_h = len(rows) * (FIXED_COL_WIDTH * 0.8 + 0.25) + TOP_MARGIN + BOTTOM_MARGIN
    fig = plt.figure(figsize=(total_w, total_h), layout="constrained")
    for r, (subfig, row) in enumerate(zip(np.atleast_1d(fig.subfigures(len(rows), 1)), rows)):
        subfig.suptitle(row["title"], x=0.1, ha="left", fontsize=TITLE_SIZE + 1, fontweight="bold")
        axes = subfig.subplots(1, ncols)
        runs = []
        for c, config in enumerate(row["configs"]):
            ax = axes[c]
            df = read_run(config)
            violins(ax, df, color, dot_size=1)
            ax.set_ylim(ylims)
            ax.set_yticks(yticks)
            ax.set_ylabel(DISTANCE_LABEL if c == 0 else "", fontsize=LABEL_SIZE - 1)
            if c > 0:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(f"Run {c + 1}", fontsize=TITLE_SIZE, fontweight="bold", pad=6)
            runs.append(df.assign(Seed=config["seed"]))
        for c in range(len(row["configs"]), nruns):
            axes[c].set_axis_off()
        ax = axes[-1]
        data = pd.concat(runs, ignore_index=True)
        sns.lineplot(data=seed_means(data, "diversity_score"), x="generation", y="diversity_score", ax=ax,
                     errorbar=ERRORBAR, color=color, linewidth=1.5)
        ax.set_xlim(0, 30)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_ylim(diversity_ylims)
        ax.set_yticks(diversity_yticks)
        ax.set_ylabel(DIVERSITY_LABEL, fontsize=LABEL_SIZE - 1)
        for ax in axes:
            ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
            ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    save_figure(fig, path)
