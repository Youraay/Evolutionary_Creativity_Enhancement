"""Tournament size: selection tournaments of 2, 3, 5 or 10 candidates (Supplementary Fig. 8).

Output (``plots/tournament_size/``):

    tournament_<metric>_curves     one panel per tournament size, mean +- SEM per prompt
    tournament_<metric>_violins    one panel per tournament size, violins of generations 0/10/20/30

with ``<metric>`` fitness, diversity or divergence (distance to the mean of generation 0), all in BLIP-2 space.

    python -m analysis.tournament_size
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.common.paths import PLOTS_DIR
from analysis.common.plotting import METRIC_LABELS, gen_violins, prompt_curves, prompt_legend_handles, set_ticks
from analysis.common.results import load_embedding_experiment
from analysis.common.style import (BOTTOM_MARGIN, FIXED_COL_WIDTH, LABEL_SIZE, PAGE_WIDTH_IN, TICK_SIZE, TITLE_SIZE,
                                   TOP_MARGIN, save_figure, set_preview_dir)

OUT_DIR = PLOTS_DIR / "tournament_size"


def render_tournament() -> None:
    """Curves and violins of fitness, diversity and divergence per tournament size."""
    data, _ = load_embedding_experiment("tournament_size")
    variants = sorted(data["variant"].unique(), key=lambda v: int(v.split("_")[-1]))
    size = (PAGE_WIDTH_IN, FIXED_COL_WIDTH * 0.7 + TOP_MARGIN + BOTTOM_MARGIN)
    for key in ["fitness", "diversity", "divergence"]:
        for kind in ("curves", "violins"):
            fig, axes = plt.subplots(1, len(variants), figsize=size, sharey=True, layout="constrained")
            for ax, variant in zip(axes, variants):
                subset = data[data["variant"] == variant]
                if kind == "curves":
                    prompt_curves(ax, subset, key, lw=1.5)
                    set_ticks(ax, [0, 15, 30])
                else:
                    gen_violins(ax, subset, key)
                    set_ticks(ax)
                ax.set_title(f"k = {variant.split('_')[-1]}", fontsize=TITLE_SIZE, fontweight="bold", pad=10)
                ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
            axes[0].set_ylabel(METRIC_LABELS[key], fontsize=LABEL_SIZE)
            if kind == "curves":
                fig.legend(handles=prompt_legend_handles(), loc="outside upper center", ncols=3,
                           fontsize=TICK_SIZE, frameon=False, handlelength=1.8)
            save_figure(fig, OUT_DIR / f"tournament_{key}_{kind}")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render_tournament()


if __name__ == "__main__":
    main()
