"""Mean +- SD curves of the experiments with population 100 (Fig. 1E, Fig. 2F)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.common.runs import read_run
from analysis.common.statistics import generation_test, stars
from analysis.common.style import (BOTTOM_MARGIN, ERRORBAR, FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, MAX_FIG_COLUMNS,
                                   RIGHT_MARGIN, TICK_SIZE, TITLE_SIZE, TOP_MARGIN, save_figure)
from analysis.common.violins import significance_bracket

METRIC_LABELS = {"diversity_score": "diversity within\ngeneration [a.u.]", "clip_score": "CLIP Score"}
"""Axis labels of the manuscript."""


def _collect(configs: list[dict], metric: str) -> pd.DataFrame:
    """Stack ``generation`` and ``metric`` of all runs that have the metric, with ``Type`` and ``Seed``."""
    frames = []
    for config in configs:
        df = read_run(config)
        if metric in df.columns:
            frame = df[["generation", metric]].copy()
            frame["Type"] = config["name"]
            frame["Seed"] = config["seed"]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def render_metric_panels(path: Path, configs: list[dict], names: list[str], colors: list[str], metric: str,
                         ncols: int = 3, nrows: int = 2, yscale=(0.1, 0.3), ylims=(0.1, 0.25),
                         positions=((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)),
                         test: bool = False) -> pd.DataFrame | None:
    """One panel per name with mean +- SD over all candidates, the seeds pooled (Fig. 3, supplement).

    Args:
        path: Output path without extension.
        configs: Run configurations.
        names: Names to show, one panel each.
        colors: Colour per name.
        metric: ``"diversity_score"`` or ``"clip_score"``.
        ncols: Grid columns.
        nrows: Grid rows.
        yscale: y ticks.
        ylims: y limits.
        positions: Grid position per name; unused positions are removed.
        test: Mark the Mann-Whitney U test of generation 0 vs. the last generation (all candidates of the runs
            pooled) with a bracket and write the results to ``<path>_tests.csv``.

    Returns:
        The tests (if ``test``), else ``None``.
    """
    data = _collect(configs, metric)
    tests = []
    total_w = ncols * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN
    total_h = nrows * FIXED_COL_WIDTH * 0.7 + TOP_MARGIN + BOTTOM_MARGIN
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(total_w, total_h), sharey=True, squeeze=False,
                             layout="constrained")
    x_max = int(data["generation"].max())          # 30 generations -> 0/15/30, long runs -> 0/50/100
    for i, name in enumerate(names):
        ax = axes[positions[i]]
        sns.lineplot(data=data[data["Type"] == name], x="generation", y=metric, ax=ax,
                     errorbar=ERRORBAR, color=colors[i], linewidth=2)
        ax.set_title(name, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.set_ylim(ylims)
        ax.set_yticks(yscale)
        if test:
            result = generation_test(data[data["Type"] == name], metric, 0, x_max)
            tests.append({"name": name, **result})
            ax.set_ylim(ylims[0], ylims[1] + 0.12 * (ylims[1] - ylims[0]))
            significance_bracket(ax, 0.5, x_max - 0.5, ylims[1] + 0.04 * (ylims[1] - ylims[0]), stars(result["p"]))
        ax.set_xlim(0, x_max)
        ax.set_xticks([0, x_max // 2, x_max])
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        if positions[i][0] == nrows - 1:
            ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
    axes[0, 0].set_ylabel(METRIC_LABELS[metric], fontsize=LABEL_SIZE)
    if nrows > 1:
        axes[1, 0].set_ylabel(METRIC_LABELS[metric], fontsize=LABEL_SIZE)
    for position in positions[len(names):]:
        fig.delaxes(axes[position])
    save_figure(fig, path, redraw=True)
    if not test:
        return None
    table = pd.DataFrame(tests)
    table.to_csv(Path(path).with_name(f"{Path(path).name}_tests.csv"), index=False)
    return table


def render_line_plot(path: Path, configs: list[dict], metric: str = "global_score",
                     ylabel: str = "distanceToInitialSeed", target_names: list[str] | None = None,
                     yscale=(0.0, 0.4), ylims=(0.0, 0.4), height_rows: int = 1) -> None:
    """One curve per name (mean +- SD over all candidates, the seeds pooled) over 30 generations (Fig. 1E).

    Args:
        path: Output path without extension.
        configs: Run configurations.
        metric: Column to plot.
        ylabel: y-axis label.
        target_names: Names to include (all if ``None``).
        yscale: y ticks.
        ylims: y limits.
        height_rows: Figure height in panel rows.
    """
    frames, palette = [], {}
    for config in configs:
        if target_names is not None and config["name"] not in target_names:
            continue
        df = read_run(config)
        df["config_name"] = config["name"]
        df["Seed"] = config["seed"]
        palette.setdefault(config["name"], config["color"])
        frames.append(df)
    if not frames:
        print("[WARN] no data to plot; check target_names")
        return
    data = pd.concat(frames, ignore_index=True)

    total_w = MAX_FIG_COLUMNS * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN
    total_h = height_rows * FIXED_COL_WIDTH * 0.7 + TOP_MARGIN + BOTTOM_MARGIN
    fig, ax = plt.subplots(figsize=(total_w, total_h), layout="constrained")
    sns.lineplot(data=data, x="generation", y=metric,
                 hue="config_name", palette=palette, errorbar=ERRORBAR, ax=ax, linewidth=2, legend=False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_yticks(yscale)
    ax.set_ylim(ylims)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlim(0, 30)
    save_figure(fig, path, redraw=True)
