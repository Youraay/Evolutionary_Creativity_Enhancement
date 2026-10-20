"""Fitness embedding: novelty measured with BLIP-2, CLIP, DINOv2 or SigLIP 2 (Supplementary Fig. 3C-D).

Output (``plots/fitness_embedding/``):

    embedding_heatmap              change of the median from generation 0 to 30 in %, fitness embedding x
                                   metric (novelty and diversity in all four spaces, CLIP and SigLIP score)
    embedding_<metric>_curves      course per fitness embedding (BLIP-2 metrics) in one axes;
                                   ``_panels``: one panel per fitness embedding

    python -m analysis.fitness_embedding
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from analysis.common.paths import PLOTS_DIR
from analysis.common.plotting import curve, set_ticks
from analysis.common.results import load_embedding_experiment
from analysis.common.style import (BOTTOM_MARGIN, FIXED_COL_WIDTH, LABEL_SIZE, PAGE_WIDTH_IN, TICK_SIZE, TITLE_SIZE,
                                   TOP_MARGIN, save_figure, set_preview_dir)

OUT_DIR = PLOTS_DIR / "fitness_embedding"

EMBEDDINGS = {"blip2": "BLIP-2", "clip": "CLIP", "dino": "DINOv2", "siglip": "SigLIP 2"}
EMBEDDING_COLORS = {"blip2": "#1B9E77", "clip": "#E7298A", "dino": "#E6AB02", "siglip": "#A6761D"}
"""Categorical (Dark2 without orange and purple), so that nothing is confused with the prompt colours."""
EMBEDDING_YLABELS = {
    "divergence": "distance to mean of initial\ngeneration (BLIP-2) [a.u.]",
    "diversity": "diversity within\ngeneration (BLIP-2) [a.u.]",
}

def cross_embedding_summary(data: pd.DataFrame, gens=(0, 30)) -> pd.DataFrame:
    """Relative change of the median from the first to the last generation in %, prompts and seeds pooled.

    Args:
        data: Candidates of the embedding ablation.
        gens: ``(first, last)`` generation.

    Returns:
        Fitness embedding (rows) x metric (columns: novelty and diversity in all four spaces, CLIP and SigLIP score).
    """
    metrics = ([f"model_MMD_{e}" for e in EMBEDDINGS] + [f"model_Diversity_{e}" for e in EMBEDDINGS]
               + ["model_CLIPScore", "model_SigLIPScore"])
    data = data.rename(columns={"divergence": "model_MMD_blip2", "diversity": "model_Diversity_blip2"})
    medians = data[data["generation"].isin(gens)].groupby(["variant", "generation"])[metrics].median()
    first, last = medians.xs(gens[0], level="generation"), medians.xs(gens[-1], level="generation")
    relative = (last - first) / first.abs() * 100
    return relative.reindex([f"fitness_{e}" for e in EMBEDDINGS])


def render_embedding_heatmap(data: pd.DataFrame) -> None:
    """Heat map of :func:`cross_embedding_summary` (red = increase, blue = decrease); own embedding framed.

    Args:
        data: Candidates of the embedding ablation.
    """
    relative = cross_embedding_summary(data)
    relative.to_csv(OUT_DIR / "embedding_heatmap.csv")
    values = relative.to_numpy(dtype=float)
    limit = np.ceil(np.nanmax(np.abs(values)) / 50) * 50
    cmap, norm = plt.get_cmap("RdBu_r"), TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 2.3), layout="constrained")
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        r, g, b, _ = cmap(norm(v))
        dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45
        ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=TICK_SIZE - 1,
                color="white" if dark else "black")
    for i in range(len(EMBEDDINGS)):        # own fitness embedding in the divergence block
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.2, zorder=4))
    for x in (3.5, 7.5):
        ax.axvline(x, color="white", lw=3, zorder=3)
    names = list(EMBEDDINGS.values())
    ax.set_xticks(range(values.shape[1]), names + names + ["CLIP score", "SigLIP score"],
                  rotation=35, ha="right", rotation_mode="anchor", fontsize=TICK_SIZE)
    ax.set_yticks(range(values.shape[0]), names, fontsize=TICK_SIZE)
    ax.set_ylabel("fitness embedding", fontsize=LABEL_SIZE)
    for x, label in ((1.5, "distance to mean of\ninitial generation"), (5.5, "diversity within\ngeneration"),
                     (8.5, "fidelity")):
        ax.text(x, -0.65, label, ha="center", va="bottom", fontsize=LABEL_SIZE, fontweight="bold")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Δ median gen. 0 → 30 [%]", fontsize=LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_SIZE, length=2, width=0.5)
    colorbar.outline.set_visible(False)
    save_figure(fig, OUT_DIR / "embedding_heatmap")


def render_embedding() -> None:
    """Heat map and BLIP-2 curves per fitness embedding (3 prompts x 3 seeds)."""
    data, _ = load_embedding_experiment("fitness_embedding")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_embedding_heatmap(data)
    for key in ["divergence", "diversity"]:
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN / 2, 2.2), layout="constrained")
        summary = []
        for embedding, label in EMBEDDINGS.items():
            stats = curve(ax, data[data["variant"] == f"fitness_{embedding}"], key, EMBEDDING_COLORS[embedding],
                          lw=1.5, label=label, error="sem")
            summary.append(stats.reset_index().assign(fitness_embedding=label))
        pd.concat(summary).to_csv(OUT_DIR / f"embedding_{key}_curves_sem.csv", index=False)
        set_ticks(ax, [0, 15, 30])
        ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
        ax.set_ylabel(EMBEDDING_YLABELS[key], fontsize=LABEL_SIZE)
        ax.legend(title="fitness embedding", title_fontsize=TICK_SIZE - 1, fontsize=TICK_SIZE - 1,
                  loc="upper left", ncols=2, frameon=False, handlelength=1.4, columnspacing=1.0)
        save_figure(fig, OUT_DIR / f"embedding_{key}_curves")

        fig, axes = plt.subplots(1, len(EMBEDDINGS), figsize=(PAGE_WIDTH_IN, FIXED_COL_WIDTH * 0.7 + TOP_MARGIN
                                                               + BOTTOM_MARGIN), sharey=True, layout="constrained")
        for ax, (embedding, label) in zip(axes, EMBEDDINGS.items()):
            curve(ax, data[data["variant"] == f"fitness_{embedding}"], key, EMBEDDING_COLORS[embedding], lw=1.5,
                  error="sem")
            set_ticks(ax, [0, 15, 30])
            ax.set_title(f"fitness: {label}", fontsize=TITLE_SIZE, fontweight="bold", pad=10)
            ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
        axes[0].set_ylabel(EMBEDDING_YLABELS[key], fontsize=LABEL_SIZE)
        save_figure(fig, OUT_DIR / f"embedding_{key}_curves_panels")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render_embedding()


if __name__ == "__main__":
    main()
