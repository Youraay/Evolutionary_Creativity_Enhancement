"""Plot building blocks shared by the analyses: mean +- SEM curves, violins and image grids."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.common.style import (COLOR_MAP, IMG_BOTTOM_MARGIN, IMG_LEFT_MARGIN, IMG_RIGHT_MARGIN, IMG_TOP_MARGIN,
                                   LABEL_SIZE, PAGE_WIDTH_IN, TICK_SIZE, TITLE_SIZE, save_figure)

PROMPTS = ["a cat", "a chair", "boredom"]
"""Prompts of the ablations, the model comparison and the T2G experiment."""
SEED_ORDER = ["1943746817", "1394004965", "3030801653"]
SEED_NAMES = dict(zip(SEED_ORDER, ["Seed A", "Seed B", "Seed C"]))
VIOLIN_GENS = [0, 10, 20, 30]

GEN_RAMP = {0: "#969696", 10: "#737373", 20: "#525252", 30: "#252525"}
"""Generations from light to dark; pooled prompts are drawn in grey."""
FILL_ALPHA = 0.3
BAND_ALPHA = 0.2
INK_GREY = "#737373"
"""Legend entries "data" / "random baseline" as in Supp. Fig. 4."""

METRIC_LABELS = {
    "fitness": "fitness [a.u.]",
    "divergence": "distance to mean of\ninitial generation [a.u.]",
    "diversity": "diversity within\ngeneration [a.u.]",
}


def run_sem(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean and SEM per generation over runs (every run is first averaged over its candidates).

    Args:
        df: Candidates with the columns prompt, seed, generation and ``col``.
        col: Metric.

    Returns:
        Table indexed by generation with mean, std, count and sem.
    """
    per_run = df.groupby(["prompt", "seed", "generation"])[col].mean()
    stats = per_run.groupby("generation").agg(["mean", "std", "count"])
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    return stats


def curve(ax: plt.Axes, df: pd.DataFrame, col: str, color: str, lw: float = 2, label: str | None = None,
          band: bool = True, band_alpha: float = BAND_ALPHA) -> pd.DataFrame:
    """Draw mean +- SEM over runs (prompt x seed), the independent units.

    Args:
        ax: Target axes.
        df: Candidates of the runs to pool.
        col: Metric.
        color: Line and band colour.
        lw: Line width.
        label: Legend label.
        band: Whether to draw the SEM band.
        band_alpha: Opacity of the band.

    Returns:
        The statistics of :func:`run_sem`.
    """
    stats = run_sem(df, col)
    ax.plot(stats.index, stats["mean"], color=color, linewidth=lw, label=label, zorder=3)
    if band:
        ax.fill_between(stats.index, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"], color=color,
                        alpha=band_alpha, linewidth=0)
    return stats


def prompt_curves(ax: plt.Axes, df: pd.DataFrame, col: str, lw: float = 2) -> None:
    """Draw one mean +- SEM curve per prompt of :data:`PROMPTS` in its paper colour.

    Args:
        ax: Target axes.
        df: Candidates.
        col: Metric.
        lw: Line width.
    """
    for prompt in PROMPTS:
        curve(ax, df[df["prompt"] == prompt], col, COLOR_MAP[prompt], lw)


def gen_violins(ax: plt.Axes, df: pd.DataFrame, col: str, gens=VIOLIN_GENS, dot_size: float = 0.6) -> None:
    """Violins as in Fig. 3 (transparent body, opaque outline and dots), one per generation.

    Args:
        ax: Target axes.
        df: Candidates (all prompts pooled).
        col: Metric.
        gens: Generations to show.
        dot_size: Marker size of the candidates.
    """
    rng = np.random.default_rng(0)
    for i, generation in enumerate(gens):
        values = df.loc[df["generation"] == generation, col].dropna().to_numpy()
        color = GEN_RAMP[generation]
        body = ax.violinplot([values], positions=[i], widths=0.8, showextrema=False)["bodies"][0]
        body.set_facecolor(mpl.colors.to_rgba(color, FILL_ALPHA))
        body.set_edgecolor(color)
        body.set_linewidth(1)
        body.set_alpha(None)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=dot_size, color=color, alpha=0.8,
                   linewidths=0, zorder=3)
    ax.set_xticks(range(len(gens)), [str(g) for g in gens])
    ax.set_xlim(-0.6, len(gens) - 0.4)
    ax.grid(axis="x", visible=False)


def prompt_legend_handles(lw: float = 1.5) -> list[Line2D]:
    """Legend handles for the prompts of :data:`PROMPTS`."""
    return [Line2D([], [], color=COLOR_MAP[p], lw=lw, label=p) for p in PROMPTS]


def set_ticks(ax: plt.Axes, xticks=None) -> None:
    """Set the tick label size and, optionally, the x ticks and limits.

    Args:
        ax: Target axes.
        xticks: x ticks; the limits become the first and last tick.
    """
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    if xticks is not None:
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_xticks(xticks)


def image_grid(rows: list[list[np.ndarray]], row_names: list[str], col_names: list[str], title: str,
               path: Path) -> None:
    """Save a grid of images over the full page width.

    Args:
        rows: Images per row.
        row_names: Label left of every row.
        col_names: Title above every column.
        title: Figure title.
        path: Output path without extension.
    """
    nrows, ncols = len(rows), len(col_names)
    width = (PAGE_WIDTH_IN - IMG_LEFT_MARGIN - IMG_RIGHT_MARGIN) / ncols
    total_w, total_h = PAGE_WIDTH_IN, nrows * width + IMG_TOP_MARGIN + IMG_BOTTOM_MARGIN
    fig, axes = plt.subplots(nrows, ncols, figsize=(total_w, total_h), squeeze=False)
    plt.subplots_adjust(left=IMG_LEFT_MARGIN / total_w, right=1 - IMG_RIGHT_MARGIN / total_w,
                        top=1 - IMG_TOP_MARGIN / total_h, bottom=IMG_BOTTOM_MARGIN / total_h,
                        wspace=0.05, hspace=0.05)
    for r, images in enumerate(rows):
        for c, image in enumerate(images):
            ax = axes[r, c]
            ax.imshow(image, interpolation="none")
            ax.axis("off")
            if c == 0:
                ax.text(-0.12, 0.5, row_names[r], va="center", ha="center", fontsize=TICK_SIZE,
                        transform=ax.transAxes, rotation=90)
            if r == 0:
                ax.set_title(col_names[c], fontsize=LABEL_SIZE, pad=6)
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", va="top")
    save_figure(fig, path)
