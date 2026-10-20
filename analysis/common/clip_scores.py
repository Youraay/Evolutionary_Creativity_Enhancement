"""CLIP scores of the six original prompts with random baseline (Fig. 3 / supplement)."""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.common.style import (BOTTOM_MARGIN, ERRORBAR, FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, RIGHT_MARGIN,
                                   TICK_SIZE, TITLE_SIZE, TOP_MARGIN, save_figure)


def extract_generation(image_path: str) -> float:
    """Generation from an image path ``.../g<gen>_idn...`` (NaN if absent).

    Args:
        image_path: Path stored in the CLIP-score CSV files.

    Returns:
        The generation, or NaN.
    """
    match = re.search(r"/g(\d+)_idn", str(image_path))
    return int(match.group(1)) if match else np.nan


def render_clip_scores(path: Path, clip_scores_dir: Path, colors: list[str], names: list[str], ncols: int = 3,
                       nrows: int = 2, yscale=(0.4, 0.5, 0.6, 0.7, 0.8), ylims=(0.4, 0.8),
                       positions=((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2))) -> None:
    """One panel per prompt: CLIP score (mean +- SD over all images of the 3 seeds) and random baseline (mean +- SD).

    Reads ``scores_<prompt>_seed_<1-3>.csv`` and ``baseline_scores_<prompt>.csv`` from ``clip_scores_dir``.

    Args:
        path: Output path without extension.
        clip_scores_dir: Directory of the CLIP-score tables.
        colors: Colour per prompt.
        names: Prompts, one panel each.
        ncols: Grid columns.
        nrows: Grid rows.
        yscale: y ticks.
        ylims: y limits.
        positions: Grid position per prompt; unused positions are removed.
    """
    total_w = ncols * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN
    total_h = nrows * FIXED_COL_WIDTH * 0.7 + TOP_MARGIN + BOTTOM_MARGIN
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(total_w, total_h), sharey=True, squeeze=False,
                             layout="constrained")
    base_dir = Path(clip_scores_dir)
    for i, prompt in enumerate(names):
        ax = axes[positions[i]]
        suffix = prompt.replace(" ", "_")
        frames = []
        for seed in (1, 2, 3):
            csv_path = base_dir / f"scores_{suffix}_seed_{seed}.csv"
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
                frame["generation"] = frame["image_path"].apply(extract_generation)
                frame["Type"] = prompt
                frame["Seed"] = seed
                frames.append(frame)
        if frames:
            sns.lineplot(data=pd.concat(frames, ignore_index=True), x="generation",
                         y="clip_score", ax=ax, errorbar=ERRORBAR, color=colors[i], linewidth=2)

        baseline_path = base_dir / f"baseline_scores_{suffix}.csv"
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)["baseline_clip_score"]
            mean, error = baseline.mean(), baseline.std()
            ax.axhline(mean, color=colors[i], linestyle="--", linewidth=2)
            ax.fill_between([0, 30], mean - error, mean + error, color=colors[i], alpha=0.15)

        ax.set_title(prompt, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.set_ylim(ylims)
        ax.set_yticks(yscale)
        ax.set_xlim(0, 30)
        ax.set_xticks([0, 15, 30])
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        if positions[i][0] == nrows - 1:
            ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
    axes[0, 0].set_ylabel("CLIP Score", fontsize=LABEL_SIZE)
    if nrows > 1:
        axes[1, 0].set_ylabel("CLIP Score", fontsize=LABEL_SIZE)
    for position in positions[len(names):]:
        fig.delaxes(axes[position])
    save_figure(fig, path, redraw=True)
