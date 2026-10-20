"""Overview of GA variants: violins of the first run, diversity and CLIP score (Supplementary Figs. 4, 6, 7)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.common.clip_scores import extract_generation
from analysis.common.runs import read_run
from analysis.common.style import (BOTTOM_MARGIN, ERRORBAR, FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, RIGHT_MARGIN,
                                   TICK_SIZE, TITLE_SIZE, TOP_MARGIN, save_figure)


def _violins(ax: plt.Axes, df: pd.DataFrame, color: str) -> None:
    """Violins and points of ``global_score`` for every 10th (and the last) generation of one run."""
    all_gens = sorted(df["generation"].unique())
    target_gens = [g for g in all_gens if g % 10 == 0 or g == all_gens[-1]]
    data = df[df["generation"].isin(target_gens)].copy()
    violins = sns.violinplot(data=data, x="generation", y="global_score", ax=ax, inner=None, color=color, alpha=0.3,
                             cut=0, density_norm="width")
    sns.stripplot(data=data, x="generation", y="global_score", ax=ax, size=1, color=color, alpha=0.8, jitter=True)
    for collection in violins.collections:
        collection.set_edgecolor(color)
        collection.set_linewidth(1)


def _clip_scores(clip_dir: Path, name: str, configs: list[dict]) -> pd.DataFrame | None:
    """CLIP scores of all runs of a variant from ``<clip_dir>/*<name>*<seed>*.csv``.

    Args:
        clip_dir: Directory of the CLIP-score tables of the comparison.
        name: Variant name.
        configs: Runs of the variant.

    Returns:
        Scores with generation and seed, or ``None`` if no table is found.
    """
    frames = []
    for config in configs:
        seed = config["seed"].replace(" ", "_").lower()
        files = list(clip_dir.glob(f"*{name}*{seed}*.csv", case_sensitive=False))
        if not files:
            number = "".join(filter(str.isdigit, config["seed"]))
            files = list(clip_dir.glob(f"*{name}*seed_{number}*.csv"))
        for file in sorted(files):
            frame = pd.read_csv(file)
            if "image_path" in frame.columns and "clip_score" in frame.columns:
                frame["generation"] = frame["image_path"].apply(extract_generation)
                frame["Seed"] = config["seed"]
                frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).dropna(subset=["generation"])


def render_comparison_overview(path: Path, configs: list[dict], names: list[str], clip_scores_dir: Path,
                               baseline_clip_path: Path, display_names: dict[str, str] | None = None,
                               yscale_viol=(0.0, 0.4), ylims_viol=(0.0, 0.4), yscale_div=(0.1, 0.3),
                               ylims_div=(0.1, 0.25), yscale_clip=(0.2, 0.3), ylims_clip=(0.2, 0.3)) -> None:
    """One row per variant: distance to the initial generation (violins of the first seed), diversity and
    CLIP score (mean +- SD over all candidates, the seeds pooled, with the random baseline of "a cat").

    Args:
        path: Output path without extension.
        configs: Run configurations of all variants.
        names: Variants, one row each.
        clip_scores_dir: Directory of the CLIP-score tables of the comparison.
        baseline_clip_path: Random baseline of the prompt.
        display_names: Row titles per variant.
        yscale_viol: y ticks of the violins.
        ylims_viol: y limits of the violins.
        yscale_div: y ticks of the diversity.
        ylims_div: y limits of the diversity.
        yscale_clip: y ticks of the CLIP score.
        ylims_clip: y limits of the CLIP score.
    """
    nrows, ncols = len(names), 3
    total_w = ncols * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN
    total_h = nrows * FIXED_COL_WIDTH * 0.8 + TOP_MARGIN + BOTTOM_MARGIN
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(total_w, total_h), squeeze=False,
                             layout="constrained")
    for row, name in enumerate(names):
        variant = [c for c in configs if c["name"] == name]
        if not variant:
            continue
        color = variant[0]["color"]

        ax_viol = axes[row, 0]
        _violins(ax_viol, read_run(variant[0]), color)
        ax_viol.set_yticks(yscale_viol)
        ax_viol.set_ylim(ylims_viol)

        runs = []
        for config in variant:
            df = read_run(config)
            df["Seed"] = config["seed"]
            runs.append(df)
        data = pd.concat(runs, ignore_index=True)
        ax_div = axes[row, 1]
        if "diversity_score" in data.columns:
            sns.lineplot(data=data, x="generation", y="diversity_score", ax=ax_div,
                         errorbar=ERRORBAR, color=color, linewidth=2)
            ax_div.set_yticks(yscale_div)
            ax_div.set_ylim(ylims_div)
            ax_div.set_xlim(0, 30)
            ax_div.set_xticks([0, 15, 30])
            ax_div.grid(True, linestyle=":", alpha=0.6)

        ax_clip = axes[row, 2]
        if baseline_clip_path and Path(baseline_clip_path).exists():
            baseline = pd.read_csv(baseline_clip_path)
            if "baseline_clip_score" in baseline.columns:
                mean, error = baseline["baseline_clip_score"].mean(), baseline["baseline_clip_score"].std()
                ax_clip.axhline(mean, color=color, linestyle="--", linewidth=1)
                ax_clip.fill_between([0, 30], mean - error, mean + error, color=color, alpha=0.15)
        clip = _clip_scores(Path(clip_scores_dir), name, variant)
        if clip is not None:
            sns.lineplot(data=clip, x="generation", y="clip_score", ax=ax_clip,
                         errorbar=ERRORBAR, color=color, linewidth=2)
        ax_clip.set_yticks(yscale_clip)
        ax_clip.set_ylim(ylims_clip)
        ax_clip.set_xlim(0, 30)
        ax_clip.set_xticks([0, 15, 30])
        ax_clip.grid(True, linestyle=":", alpha=0.6)

        title = display_names.get(name, name) if display_names else name
        ax_div.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=15)
        for col in range(ncols):
            ax = axes[row, col]
            ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
            if row == nrows - 1:
                ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
        ax_viol.set_ylabel("distance to mean of\ninitial generation [a.u.]", fontsize=LABEL_SIZE - 1)
        ax_div.set_ylabel("diversity within\ngeneration [a.u.]", fontsize=LABEL_SIZE - 1)
        ax_clip.set_ylabel("CLIP Score", fontsize=LABEL_SIZE)
    save_figure(fig, path, redraw=True)
