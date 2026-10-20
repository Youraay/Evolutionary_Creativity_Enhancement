"""Long runs: the standard GA over 300 generations (Supplementary Fig. 1D).

Output (``plots/long_runs/``):

    long_run_metrics               Supplementary Fig. 1D (bottom): fitness, divergence and diversity over 300
                                   generations, all prompts in one panel
    long_run_<metric>              fitness / divergence / diversity / fidelity, one panel per prompt
    long_run_examples_<prompt>     example images of generations 0 / 50 / 100 / 200 / 300 (random candidates of
                                   the generation with a fixed random seed, not a selection)
    data/long_run_mean_sd.csv      the curves as a table

Divergence = distance to the mean of generation 0 in BLIP-2 space, diversity = diversity within the generation
(BLIP-2), fitness = novelty in BLIP-2 space, fidelity = CLIP score with the random baseline of 100 images of
the other prompts.

    python -m analysis.long_runs
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.common.paths import PLOTS_DIR
from analysis.common.plotting import (INK_GREY, METRIC_LABELS, PROMPTS, SEED_NAMES, SEED_ORDER, curve, image_grid,
                                      prompt_curves, prompt_legend_handles, set_ticks)
from analysis.common.results import load_embedding_experiment, load_images, random_candidates, random_clip_baseline
from analysis.common.style import (BOTTOM_MARGIN, COLOR_MAP, FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, RIGHT_MARGIN,
                                   TICK_SIZE, TITLE_SIZE, TOP_MARGIN, save_figure, set_preview_dir)

OUT_DIR = PLOTS_DIR / "long_runs"

LONG_RUN_METRICS = {
    "fitness": ("fitness", METRIC_LABELS["fitness"]),
    "divergence": ("divergence", METRIC_LABELS["divergence"]),
    "diversity": ("diversity", METRIC_LABELS["diversity"]),
    "fidelity": ("model_CLIPScore", "Fidelity score [a.u.]"),
}


def long_run_baseline(data: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Random CLIP-score baseline of the long-run prompts (30 pool images per run), cached as CSV.

    Args:
        data: Candidates of the long runs.
        runs: Runs of the long runs.

    Returns:
        One row per prompt and baseline image.
    """
    cache = OUT_DIR / "data" / "baseline_clip_scores.csv"
    if cache.exists():
        return pd.read_csv(cache)
    baseline = random_clip_baseline(runs, data, pool_per_run=30)
    cache.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(cache, index=False)
    return baseline


def render_metrics(data: pd.DataFrame) -> None:
    """Fitness, divergence and diversity side by side, one curve per prompt (Supplementary Fig. 1D).

    Args:
        data: Candidates of the long runs.
    """
    n_gen = int(data["generation"].max())
    size = (3 * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN, FIXED_COL_WIDTH * 0.8 + TOP_MARGIN + BOTTOM_MARGIN)
    fig, axes = plt.subplots(1, 3, figsize=size, layout="constrained")
    for ax, key in zip(axes, ("fitness", "divergence", "diversity")):
        prompt_curves(ax, data, key, lw=1.2)
        set_ticks(ax, [0, n_gen // 2, n_gen])
        ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
        ax.set_ylabel(METRIC_LABELS[key], fontsize=LABEL_SIZE - 1)
    axes[0].legend(handles=prompt_legend_handles(), loc="lower right", fontsize=TICK_SIZE - 1, frameon=False,
                   handlelength=1.5)
    save_figure(fig, OUT_DIR / "long_run_metrics")


def render_long_run() -> None:
    """Curves of all metrics over 300 generations, their table and example images."""
    data, runs = load_embedding_experiment("long_runs")
    render_metrics(data)
    baseline = long_run_baseline(data, runs)
    n_gen = int(data["generation"].max())
    size = (3 * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN, FIXED_COL_WIDTH * 0.7 + TOP_MARGIN + BOTTOM_MARGIN)

    for key, (col, ylabel) in LONG_RUN_METRICS.items():
        fig, axes = plt.subplots(1, 3, figsize=size, sharey=True, layout="constrained")
        for ax, prompt in zip(axes, PROMPTS):
            curve(ax, data[data["prompt"] == prompt], col, COLOR_MAP[prompt], lw=1.5)
            if key == "fidelity":
                b = baseline.loc[baseline["prompt"] == prompt, "baseline_clip_score"]
                ax.axhline(b.mean(), color=COLOR_MAP[prompt], linestyle="--", linewidth=2)
                ax.fill_between([0, n_gen], b.mean() - b.std(), b.mean() + b.std(), color=COLOR_MAP[prompt], alpha=0.15,
                                linewidth=0)
            ax.set_title(prompt, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
            set_ticks(ax, [0, n_gen // 2, n_gen])
            ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
        axes[0].set_ylabel(ylabel, fontsize=LABEL_SIZE)
        if key == "fidelity":
            axes[-1].legend(handles=[Line2D([], [], color=INK_GREY, lw=1.5, label="data"),
                                     Line2D([], [], color=INK_GREY, lw=1.5, ls="--", label="random\nbaseline")],
                            loc="center right", fontsize=TICK_SIZE - 1, frameon=False, handlelength=1.8)
        save_figure(fig, OUT_DIR / f"long_run_{key}")

    summary = data.groupby(["prompt", "generation"])[[c for c, _ in LONG_RUN_METRICS.values()]].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.reset_index().to_csv(OUT_DIR / "data" / "long_run_mean_sd.csv", index=False)

    gens = [0, 50, 100, 200, 300]
    rng = np.random.default_rng(0)
    for prompt in PROMPTS:
        rows, names = [], []
        for seed in SEED_ORDER[:2]:
            run = runs[(runs["prompt"] == prompt) & (runs["seed"] == seed)].iloc[0]
            d = data[(data["prompt"] == prompt) & (data["seed"] == seed)]
            rows.append(load_images(run["zip"], [random_candidates(d, g, 1, rng)[0] for g in gens], size=400))
            names.append(SEED_NAMES[seed])
        image_grid(rows, names, [f"Gen. {g}" for g in gens], f'"{prompt}"',
                   OUT_DIR / f"long_run_examples_{prompt.replace(' ', '_')}")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render_long_run()


if __name__ == "__main__":
    main()
