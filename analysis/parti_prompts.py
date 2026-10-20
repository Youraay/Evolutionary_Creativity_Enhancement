"""Figures of the PartiPrompts experiment (114 prompts x 3 seeds, SDXL, 30 generations).

Output in ``plots/parti_prompts/``:

    fidelity_pooled          fidelity (CLIP score) over the generations, all prompts pooled, with random
                             baseline (as Supp. Fig. 4, but pooled)
    divergence_by_challenge  box plots of the distance to the mean of the initial generation, generation 0 vs.
                             30, per PartiPrompts challenge
    diversity_by_challenge   the same for the diversity within the generation
    *_by_category            the same per PartiPrompts category
    *_split                  variant: generation 0 on the left, generation 30 on the right
    <metric>_trajectories_by_challenge / _by_category
                             supplement: one panel per class, one curve per prompt (mean +- SD over the
                             seeds); the legend is a separate figure ``trajectories_by_<group>_legend``

Curves: mean +- SD; pooled fidelity over all candidates, class trajectories over the seeds (every run first
averaged over its 60 candidates). One point of a box plot
is one prompt (mean over 3 seeds and all candidates of the generation). Test: Wilcoxon signed-rank,
generation 0 vs. 30, paired over the prompts of a class.

Random baseline: CLIP score of a prompt against 100 random images of the other prompts; the CLIP image
embeddings are taken from the result archives.

    python -m analysis.parti_prompts                 # data (cached) + all figures
    python -m analysis.parti_prompts --recompute     # rebuild the cache (new runs)
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

from analysis.common.paths import PLOTS_DIR
from analysis.common.results import attach_prompt_meta, discover_parti_runs, random_clip_baseline
from analysis.common.style import (FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, PAGE_WIDTH_IN, RIGHT_MARGIN, BOTTOM_MARGIN,
                                   TICK_SIZE, TITLE_SIZE, TOP_MARGIN, save_figure, set_preview_dir)

OUT_DIR = PLOTS_DIR / "parti_prompts"
DATA_DIR = OUT_DIR / "data"

EXPECTED_ROWS = 1860
"""31 generations x 60 candidates."""
GENS = (0, 30)
POOL_PER_RUN = 10
"""Random images per run in the pool of the baseline."""

CHALLENGE_ORDER = [
    "Basic", "Simple Detail", "Fine-grained Detail", "Complex",
    "Quantity", "Properties & Positioning", "Perspective",
    "Style & Format", "Writing & Symbols", "Linguistic Structures", "Imagination",
]
"""Roughly from easy to hard: the four levels of detail first, then the other challenges."""

CATEGORY_ORDER = [
    "Animals", "People", "Produce & Plants",
    "Food & Beverage", "Artifacts", "Vehicles",
    "Indoor Scenes", "Outdoor Scenes",
    "Arts", "Illustrations", "World Knowledge", "Abstract",
]
"""From animate via objects and scenes to depictions and abstract concepts (as animate / inanimate /
abstract in Fig. 3)."""

GROUPINGS = {"challenge": ("Challenge", CHALLENGE_ORDER), "category": ("Category", CATEGORY_ORDER)}

METRICS = {
    "divergence": ("model_LocalMaxMeanDivergence", "distance to mean of\ninitial generation [a.u.]"),
    "diversity": ("model_DiversityWithinPopulation", "diversity within\ngeneration [a.u.]"),
}

POOLED_COLOR = "#525252"
"""All prompts pooled are drawn in neutral grey; the prompt colours stay reserved for single prompts."""
GEN_COLOR = {0: "#969696", 30: "#252525"}
FILL_ALPHA = 0.3

PROMPT_PALETTE = ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
                  "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#D9B700", "#B15928",
                  "#35978F", "#636363"]
"""ColorBrewer Paired (yellow darkened to stay visible on white) plus two colours for classes with more
than 12 prompts (Artifacts: 14), so that no colour repeats within a panel."""

TRAJECTORY_METRICS = {
    "divergence": ("model_LocalMaxMeanDivergence", "distance to mean of initial generation [a.u.]"),
    "diversity": ("model_DiversityWithinPopulation", "diversity within generation [a.u.]"),
    "fidelity": ("model_CLIPScore", "Fidelity score [a.u.]"),
}


# ------------------------------------------------------------------ data
def load_candidates(runs: pd.DataFrame) -> pd.DataFrame:
    """Read the metrics of all runs.

    Args:
        runs: Output of :func:`analysis.common.results.discover_parti_runs`.

    Returns:
        One row per candidate and generation with CLIP score, divergence, diversity, prompt and seed.
    """
    columns = ["generation", "file_name", "model_CLIPScore"] + [m for m, _ in METRICS.values()]
    frames = []
    for run in runs.itertuples():
        df = pd.read_csv(run.csv, usecols=columns)
        if len(df) < EXPECTED_ROWS:
            print(f"[WARN] incomplete ({len(df)} rows): {run.csv.name}")
        frames.append(df.assign(prompt=run.prompt, seed=run.seed))
    return pd.concat(frames, ignore_index=True)


def build_data(recompute: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load candidates and random baseline, from the cache in ``DATA_DIR`` if present.

    Args:
        recompute: Rebuild the cache from the result directories.

    Returns:
        ``(candidates with Category and Challenge, baseline)``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates_path = DATA_DIR / "candidates.parquet"
    baseline_path = DATA_DIR / "baseline_clip_scores.csv"
    if recompute or not candidates_path.exists() or not baseline_path.exists():
        runs = discover_parti_runs()
        print(f"[INFO] {len(runs)} runs, {runs['prompt'].nunique()} prompts")
        candidates = load_candidates(runs)
        candidates.to_parquet(candidates_path, index=False)
        random_clip_baseline(runs, candidates, POOL_PER_RUN).to_csv(baseline_path, index=False)
    return attach_prompt_meta(pd.read_parquet(candidates_path)), pd.read_csv(baseline_path)


def per_prompt(candidates: pd.DataFrame, metric: str, col: str, gens=GENS) -> pd.DataFrame:
    """One value per prompt and generation: mean over all candidates, then over the seeds.

    Args:
        candidates: Candidates with the class column ``col``.
        metric: Metric column.
        col: Class column (``"Challenge"`` or ``"Category"``).
        gens: Generations to keep.

    Returns:
        Table with class, prompt, generation and the metric.
    """
    subset = candidates[candidates["generation"].isin(gens)]
    per_run = subset.groupby([col, "prompt", "seed", "generation"])[metric].mean()
    return per_run.groupby([col, "prompt", "generation"]).mean().reset_index()


def group_stats(pp: pd.DataFrame, metric: str, col: str, order: list[str], gens=GENS) -> pd.DataFrame:
    """Wilcoxon signed-rank test of the last vs. the first generation per class and over all prompts.

    Args:
        pp: Output of :func:`per_prompt`.
        metric: Metric column.
        col: Class column.
        order: Classes in display order.
        gens: ``(first, last)`` generation.

    Returns:
        One row per class (plus ``"all"``) with the number of prompts, both medians, W and p.
    """
    wide = pp.pivot_table(index=[col, "prompt"], columns="generation", values=metric)

    def row(name, values):
        """Test of one class: ``values`` has one column per generation and one row per prompt."""
        result = wilcoxon(values[gens[1]], values[gens[0]])
        return {col: name, "n_prompts": len(values),
                f"median_gen{gens[0]}": values[gens[0]].median(), f"median_gen{gens[1]}": values[gens[1]].median(),
                "W": result.statistic, "p": result.pvalue}

    return pd.DataFrame([row(g, wide.loc[g]) for g in order] + [row("all", wide.droplevel(0))])


def stars(p: float) -> str:
    """Significance label: ``***`` < 0.001, ``**`` < 0.01, ``*`` < 0.05, else ``n.s.``."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def prompt_trajectories(candidates: pd.DataFrame, metric: str, col: str) -> pd.DataFrame:
    """Mean and SD over the seeds per prompt and generation (every run first averaged over its candidates).

    Args:
        candidates: Candidates with the class column ``col``.
        metric: Metric column.
        col: Class column.

    Returns:
        Table with class, prompt, generation, mean, std and count.
    """
    per_run = candidates.groupby([col, "prompt", "seed", "generation"])[metric].mean()
    return per_run.groupby([col, "prompt", "generation"]).agg(["mean", "std", "count"]).reset_index()


def prompt_colors(df: pd.DataFrame, col: str, order: list[str]) -> dict[str, str]:
    """Fixed colour per prompt within its class (alphabetical), identical in plots and legends.

    Args:
        df: Table with the columns prompt and ``col``.
        col: Class column.
        order: Classes.

    Returns:
        Prompt -> colour.

    Raises:
        ValueError: If a class has more prompts than the palette has colours.
    """
    colors = {}
    for group in order:
        prompts = sorted(df.loc[df[col] == group, "prompt"].unique(), key=str.lower)
        if len(prompts) > len(PROMPT_PALETTE):
            raise ValueError(f"{group}: {len(prompts)} prompts, but only {len(PROMPT_PALETTE)} colours")
        colors.update({p: PROMPT_PALETTE[i] for i, p in enumerate(prompts)})
    return colors


# ------------------------------------------------------------------ figures
def render_fidelity_pooled(candidates: pd.DataFrame, baseline: pd.DataFrame,
                           yscale=(0.2, 0.4, 0.6, 0.8, 1.0), ylims=(0.2, 1.0)) -> None:
    """Fidelity of all prompts pooled (as Supp. Fig. 4).

    Data: mean +- SD over all candidates. Baseline: mean +- SD over all random images.

    Args:
        candidates: All candidates.
        baseline: Random baseline.
        yscale: y ticks.
        ylims: y limits.
    """
    total_w = FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN
    total_h = FIXED_COL_WIDTH * 0.7 + TOP_MARGIN + BOTTOM_MARGIN
    fig, ax = plt.subplots(figsize=(total_w, total_h), layout="constrained")

    stats = candidates.groupby("generation")["model_CLIPScore"].agg(["mean", "std"])
    ax.plot(stats.index, stats["mean"], color=POOLED_COLOR, linewidth=2)
    ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=POOLED_COLOR,
                    alpha=0.2, linewidth=0)

    mean_base, std_base = baseline["baseline_clip_score"].mean(), baseline["baseline_clip_score"].std()
    ax.axhline(mean_base, color=POOLED_COLOR, linestyle="--", linewidth=2)
    ax.fill_between([0, 30], mean_base - std_base, mean_base + std_base, color=POOLED_COLOR, alpha=0.15, linewidth=0)

    ax.set_title(f"all prompts (n = {candidates['prompt'].nunique()})", fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_ylim(ylims)
    ax.set_yticks(yscale)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 15, 30])
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_ylabel("Fidelity score [a.u.]", fontsize=LABEL_SIZE)
    ax.legend(handles=[Line2D([], [], color=POOLED_COLOR, lw=1.5, label="data"),
                       Line2D([], [], color=POOLED_COLOR, lw=1.5, ls="--", label="random baseline")],
              loc="center right", fontsize=TICK_SIZE - 1, frameon=False, handlelength=1.8,
              borderaxespad=0.3, labelspacing=0.2)
    save_figure(fig, OUT_DIR / "fidelity_pooled")


def _box(ax: plt.Axes, values: np.ndarray, pos: float, gen: int, width: float, horizontal: bool = False) -> None:
    """Draw a box like the violins of Fig. 3 (transparent body, opaque outline and dots in the group colour).

    Args:
        ax: Target axes.
        values: One value per prompt.
        pos: Position of the box.
        gen: Generation (selects the colour and the jitter seed).
        width: Box width.
        horizontal: Draw a horizontal box.
    """
    color = GEN_COLOR[gen]
    line = dict(color=color, linewidth=1)
    boxes = ax.boxplot(values, positions=[pos], widths=width, patch_artist=True, showfliers=False,
                       orientation="horizontal" if horizontal else "vertical",
                       medianprops=dict(color=color, linewidth=1.5), whiskerprops=line, capprops=line,
                       boxprops=dict(edgecolor=color, linewidth=1))
    boxes["boxes"][0].set_facecolor(mpl.colors.to_rgba(color, FILL_ALPHA))
    rng = np.random.default_rng(int(round((pos + 1) * 100)) + 10000 * gen)
    jitter = pos + rng.uniform(-width * 0.3, width * 0.3, len(values))
    x, y = (values, jitter) if horizontal else (jitter, values)
    ax.scatter(x, y, s=3, color=color, alpha=0.8, linewidths=0, zorder=3)


def _bracket(ax: plt.Axes, x0: float, x1: float, y: float, text: str, h: float) -> None:
    """Draw a significance bracket from ``x0`` to ``x1`` at height ``y`` with a label."""
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color="black", linewidth=0.6)
    ax.text((x0 + x1) / 2, y + h, text, ha="center", va="bottom",
            fontsize=TICK_SIZE - 1 if text == "n.s." else TICK_SIZE)


def _gen_legend(target, gens=GENS, **kwargs) -> None:
    """Add a legend with one patch per generation to an axes or figure."""
    handles = [Patch(facecolor=mpl.colors.to_rgba(GEN_COLOR[g], FILL_ALPHA), edgecolor=GEN_COLOR[g],
                     linewidth=1, label=f"Gen. {g}") for g in gens]
    target.legend(handles=handles, fontsize=TICK_SIZE, frameon=False, handlelength=1.2, **kwargs)


def _style_category_axis(ax: plt.Axes, labels: list[str]) -> None:
    """Rotated class labels on the x axis, no vertical grid."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=TICK_SIZE)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", which="major", labelsize=TICK_SIZE)


def render_by_group(candidates: pd.DataFrame, key: str, group: str, yticks, ylims) -> None:
    """Box plots generation 0 vs. 30 per class, in two layouts, plus the test table.

    Args:
        candidates: All candidates.
        key: ``"divergence"`` or ``"diversity"``.
        group: ``"challenge"`` or ``"category"``.
        yticks: Ticks of the metric axis.
        ylims: Limits of the metric axis.
    """
    metric, ylabel = METRICS[key]
    col, order = GROUPINGS[group]
    name = f"{key}_by_{group}"
    pp = per_prompt(candidates, metric, col)
    stats = group_stats(pp, metric, col, order)
    stats.to_csv(OUT_DIR / f"{name}_stats.csv", index=False)
    pp.to_csv(DATA_DIR / f"{name}_per_prompt.csv", index=False)
    print(stats.to_string(index=False))

    labels = [f"{c} ({int(n)})" for c, n in zip(stats[col][:-1], stats["n_prompts"][:-1])]
    width, offset = 0.34, 0.19
    h = (ylims[1] - ylims[0]) * 0.02

    def values(cls, gen):
        """Per-prompt values of one class and generation."""
        return pp[(pp[col] == cls) & (pp["generation"] == gen)][metric].to_numpy()

    # Layout 1: generation 0 and 30 side by side per class
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 2.6), layout="constrained")
    for i, cls in enumerate(order):
        for gen, dx in zip(GENS, (-offset, offset)):
            _box(ax, values(cls, gen), i + dx, gen, width)
        top = max(values(cls, g).max() for g in GENS)
        _bracket(ax, i - offset, i + offset, top + h, stars(stats.loc[i, "p"]), h)
    _style_category_axis(ax, labels)
    ax.set_ylim(ylims)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    _gen_legend(ax, loc="lower right", ncols=2, bbox_to_anchor=(1, 1.0), borderaxespad=0.2)
    save_figure(fig, OUT_DIR / name)

    # Layout 2: generation 0 left, generation 30 right (horizontal boxes, classes labelled once)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 2.6), sharex=True, sharey=True, layout="constrained")
    for ax, gen in zip(axes, GENS):
        for i, cls in enumerate(order):
            _box(ax, values(cls, gen), i, gen, 0.6, horizontal=True)
        ax.set_title(f"Generation {gen}", fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylim(len(labels) - 0.4, -0.6)
    axes[0].set_xlim(ylims)
    axes[0].set_xticks(yticks)
    fig.supxlabel(ylabel.replace("\n", " "), fontsize=LABEL_SIZE, x=0.62)
    save_figure(fig, OUT_DIR / f"{name}_split")


def class_grid(n: int, height: float, constrained: bool = True):
    """Grid of 3 columns for ``n`` classes; unused panels are hidden.

    Args:
        n: Number of classes.
        height: Figure height in inches.
        constrained: Use the constrained layout.

    Returns:
        ``(figure, flat array of axes)``.
    """
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_WIDTH_IN, height), sharex=True, sharey=True,
                             layout="constrained" if constrained else None, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
        # empty panel at the bottom: show the x tick labels on the panel above
        above = axes.flat[list(axes.flat).index(ax) - ncols]
        above.tick_params(axis="x", labelbottom=True)
    return fig, axes.flat


def render_trajectories(candidates: pd.DataFrame, baseline: pd.DataFrame, group: str, key: str) -> None:
    """Supplement: one panel per class, one curve per prompt (mean +- SD over the seeds).

    Args:
        candidates: All candidates.
        baseline: Random baseline (fidelity panels show the mean baseline of the class).
        group: ``"challenge"`` or ``"category"``.
        key: Metric key of :data:`TRAJECTORY_METRICS`.
    """
    metric, ylabel = TRAJECTORY_METRICS[key]
    col, order = GROUPINGS[group]
    trajectories = prompt_trajectories(candidates, metric, col)
    trajectories.to_csv(DATA_DIR / f"{key}_trajectories_by_{group}.csv", index=False)
    colors = prompt_colors(candidates, col, order)
    base = baseline.groupby("prompt")["baseline_clip_score"].mean()

    fig, axes = class_grid(len(order), 6.0)
    for ax, cls in zip(axes, order):
        subset = trajectories[trajectories[col] == cls]
        prompts = sorted(subset["prompt"].unique(), key=str.lower)
        for prompt in prompts:
            t = subset[subset["prompt"] == prompt]
            ax.plot(t["generation"], t["mean"], color=colors[prompt], linewidth=1)
            ax.fill_between(t["generation"], t["mean"] - t["std"], t["mean"] + t["std"], color=colors[prompt],
                            alpha=0.12, linewidth=0)
        if key == "fidelity":
            ax.axhline(base.reindex(prompts).mean(), color=POOLED_COLOR, linestyle="--", linewidth=1)
        ax.set_title(f"{cls} ({len(prompts)})", fontsize=TICK_SIZE, fontweight="bold", pad=4)
        ax.set_xlim(0, 30)
        ax.set_xticks([0, 15, 30])
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE - 1)
    if key == "fidelity":
        axes[0].figure.legend(handles=[Line2D([], [], color=POOLED_COLOR, lw=1, ls="--", label="random baseline")],
                              loc="outside upper right", fontsize=TICK_SIZE - 1, frameon=False)
    fig.supxlabel("Generation", fontsize=LABEL_SIZE)
    fig.supylabel(ylabel, fontsize=LABEL_SIZE)
    save_figure(fig, OUT_DIR / f"{key}_trajectories_by_{group}")


def render_trajectory_legend(candidates: pd.DataFrame, group: str, max_chars: int = 46) -> None:
    """Legend of :func:`render_trajectories`: the prompts of every class with their line colour (same grid).

    Args:
        candidates: All candidates.
        group: ``"challenge"`` or ``"category"``.
        max_chars: Longer prompts are truncated with an ellipsis.
    """
    col, order = GROUPINGS[group]
    colors = prompt_colors(candidates, col, order)
    rows = candidates.groupby(col)["prompt"].nunique().max()        # row height from the largest class
    fig, axes = class_grid(len(order), 4 * (0.25 + rows * 0.105), constrained=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.06, hspace=0.18)
    for ax, cls in zip(axes, order):
        prompts = sorted(candidates.loc[candidates[col] == cls, "prompt"].unique(), key=str.lower)
        ax.axis("off")
        ax.set_title(cls, fontsize=TICK_SIZE, fontweight="bold", loc="left", pad=2)
        for i, prompt in enumerate(prompts):
            y = 1 - (i + 0.5) / rows
            ax.add_line(Line2D([0, 0.07], [y, y], color=colors[prompt], linewidth=1.5, transform=ax.transAxes))
            text = prompt if len(prompt) <= max_chars else prompt[:max_chars - 1].rstrip() + "…"
            ax.text(0.09, y, text, fontsize=5, va="center", ha="left", transform=ax.transAxes)
    save_figure(fig, OUT_DIR / f"trajectories_by_{group}_legend")
    pd.DataFrame([{col: cls, "prompt": p, "color": colors[p]} for cls in order
                  for p in sorted(candidates.loc[candidates[col] == cls, "prompt"].unique(), key=str.lower)]
                 ).to_csv(OUT_DIR / f"trajectories_by_{group}_legend.csv", index=False)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="rebuild the data cache")
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    candidates, baseline = build_data(args.recompute)
    print(f"[INFO] {candidates.groupby(['prompt', 'seed']).ngroups} runs, {candidates['prompt'].nunique()} prompts, "
          f"baseline: {len(baseline)} scores")
    render_fidelity_pooled(candidates, baseline)
    for group in GROUPINGS:
        render_by_group(candidates, "divergence", group, yticks=[0, 0.1, 0.2, 0.3], ylims=(0, 0.3))
        render_by_group(candidates, "diversity", group, yticks=[0, 0.1, 0.2, 0.3], ylims=(0, 0.35))
        for key in TRAJECTORY_METRICS:
            render_trajectories(candidates, baseline, group, key)
        render_trajectory_legend(candidates, group)


if __name__ == "__main__":
    main()
