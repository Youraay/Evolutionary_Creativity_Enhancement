"""Mutation grid search (Methods): mutated images per fraction ``p_m`` and strength ``sigma_m``.

Output (``plots/mutation_grid_search/``):

    <run>_c<k>          image grid of candidate k: rows = p_m, columns = sigma_m, unmutated image on top
    <run>_distance      mean BLIP-2 distance of the mutated to the unmutated image over the candidates

The tables are read from ``data/results/mutation_grid_search``, the images from the result directory.

    python -m analysis.mutation_grid_search
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analysis.common.paths import PLOTS_DIR, tables
from analysis.common.style import LABEL_SIZE, PAGE_WIDTH_IN, TICK_SIZE, TITLE_SIZE, save_figure, set_preview_dir
from experiments.common.settings import result_dir

OUT_DIR = PLOTS_DIR / "mutation_grid_search"


def render_candidate(run: str, table: pd.DataFrame, candidate: int) -> None:
    """Image grid of one candidate: rows = mutated fraction, columns = strength.

    Args:
        run: Run directory name.
        table: Grid table of the run.
        candidate: Candidate number.
    """
    directory = result_dir("mutation_grid_search") / run
    fractions, strengths = sorted(table["fraction"].unique()), sorted(table["strength"].unique())
    size = PAGE_WIDTH_IN / (len(strengths) + 0.6)
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, size * (len(fractions) + 1.4)))
    grid = fig.add_gridspec(len(fractions) + 1, len(strengths), hspace=0.08, wspace=0.04)
    ax = fig.add_subplot(grid[0, len(strengths) // 2])
    ax.imshow(Image.open(directory / f"original_c{candidate}.JPEG"))
    ax.set_title("unmutated", fontsize=TICK_SIZE)
    ax.axis("off")
    for r, fraction in enumerate(fractions):
        for c, strength in enumerate(strengths):
            ax = fig.add_subplot(grid[r + 1, c])
            ax.imshow(Image.open(directory / f"mutated_c{candidate}_p{fraction:g}_s{strength:g}.JPEG"))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"σ_m = {strength:g}", fontsize=TICK_SIZE)
            if c == 0:
                ax.set_ylabel(f"p_m = {fraction:g}", fontsize=TICK_SIZE)
    fig.suptitle(f"Candidate {candidate}", fontsize=TITLE_SIZE, fontweight="bold")
    save_figure(fig, OUT_DIR / f"{run}_c{candidate}")


def render_distance(run: str, table: pd.DataFrame) -> None:
    """Heat map of the mean BLIP-2 distance between mutated and unmutated image.

    Args:
        run: Run directory name.
        table: Grid table of the run.
    """
    mean = table.groupby(["fraction", "strength"])["blip2_distance"].mean().unstack()
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN / 2, PAGE_WIDTH_IN / 2.4), layout="constrained")
    image = ax.imshow(mean.to_numpy(), cmap="viridis", aspect="auto")
    for (i, j), value in np.ndenumerate(mean.to_numpy()):
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=TICK_SIZE - 1, color="white")
    ax.set_xticks(range(mean.shape[1]), [f"{s:g}" for s in mean.columns], fontsize=TICK_SIZE)
    ax.set_yticks(range(mean.shape[0]), [f"{p:g}" for p in mean.index], fontsize=TICK_SIZE)
    ax.set_xlabel("σ_m", fontsize=LABEL_SIZE)
    ax.set_ylabel("p_m", fontsize=LABEL_SIZE)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("BLIP-2 distance to\nunmutated image", fontsize=LABEL_SIZE - 1)
    save_figure(fig, OUT_DIR / f"{run}_distance")


def render() -> None:
    """All figures of every run of the grid search."""
    runs = sorted(p for p in tables("mutation_grid_search").glob("*/*.csv") if p.stem == p.parent.name)
    if not runs:
        print(f"[INFO] no results in {tables('mutation_grid_search')}; run experiments.mutation_grid_search first")
        return
    for path in runs:
        table = pd.read_csv(path)
        render_distance(path.stem, table)
        for candidate in sorted(table["candidate"].unique()):
            render_candidate(path.stem, table, int(candidate))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render()


if __name__ == "__main__":
    main()
