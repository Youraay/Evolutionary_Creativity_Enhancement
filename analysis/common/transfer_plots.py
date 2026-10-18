"""Transfer figures: noise optimised for one prompt rendered with another (Fig. 3, Supplementary Fig. 11)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysis.common.paths import archive, tables
from analysis.common.plotting import image_grid
from analysis.common.results import load_images
from analysis.common.runs import read_run
from analysis.common.statistics import generation_test, stars
from analysis.common.style import (BOTTOM_MARGIN, ERRORBAR, FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, RIGHT_MARGIN,
                                   TICK_SIZE, TITLE_SIZE, TOP_MARGIN, save_figure, seed_means)

GENERATIONS = [0, 10, 20, 30]
"""Generations of the source runs that were rendered with the target prompt."""


def _quoted(prompt: str) -> str:
    """Prompt in typographic quotes."""
    return f"“{prompt}”"


def transfer_tables(configs: list[dict], metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidates of the transfer runs and of the runs optimised directly for the target prompt.

    41 of the 87 transfer tables contain every row twice (the evaluation was appended a second time); exact
    duplicate rows are dropped, which leaves the run means unchanged but matters for the tests.

    Args:
        configs: Transfer configurations (:func:`analysis.common.runs.transfer_runs`) of one source and target.
        metric: Column to keep.

    Returns:
        ``(transfer, direct)`` with generation, metric and ``Seed``, restricted to :data:`GENERATIONS`.
    """
    transfer, direct = [], []
    for config in configs:
        for key, frames in (("path", transfer), ("original_path", direct)):
            df = read_run(config, key).drop_duplicates()
            frames.append(df.loc[df["generation"].isin(GENERATIONS), ["generation", metric]]
                          .assign(Seed=config["seed"]))
    return pd.concat(transfer, ignore_index=True), pd.concat(direct, ignore_index=True)


def render_transfer_curves(path: Path, configs: list[dict], metric: str, ylabel: str, title: str, yticks=(0.0, 0.2),
                           ylims=(0.0, 0.2)) -> pd.DataFrame:
    """Transfer run vs. run optimised for the target prompt: mean +- SEM over the runs at generations 0-30.

    The stars give the Mann-Whitney U test of generation 0 vs. 30 (all candidates of the runs pooled) for the
    transfer (colour of the source prompt) and the direct optimisation (colour of the target prompt).

    Args:
        path: Output path without extension; the tests are written to ``<path>_tests.csv``.
        configs: Transfer configurations of one source and target prompt.
        metric: ``global_score`` (distance to the mean of generation 0) or ``diversity_score``.
        ylabel: Label of the y axis.
        title: Title of the panel.
        yticks: y ticks.
        ylims: y limits.

    Returns:
        Table of the two tests.
    """
    source, target = configs[0]["source"], configs[0]["target"]
    source_color, target_color = configs[0]["source color"], configs[0]["target color"]
    transfer, direct = transfer_tables(configs, metric)
    fig, ax = plt.subplots(figsize=(FIXED_COL_WIDTH * 1.4 + LEFT_MARGIN + RIGHT_MARGIN,
                                    FIXED_COL_WIDTH * 0.9 + TOP_MARGIN + BOTTOM_MARGIN), layout="constrained")
    for df, color in ((direct, target_color), (transfer, source_color)):
        sns.lineplot(data=seed_means(df, metric), x="generation", y=metric, ax=ax, errorbar=ERRORBAR, color=color,
                     linewidth=2)
    tests = []
    for kind, df in (("transfer", transfer), ("direct", direct)):
        tests.append({"source": source, "target": target, "curve": kind, "runs": df["Seed"].nunique(),
                      **generation_test(df, metric, GENERATIONS[0], GENERATIONS[-1])})
    ax.set_xlim(0, 30)
    ax.set_xticks(GENERATIONS)
    ax.set_ylim(ylims[0], ylims[1] * 1.15)
    ax.set_yticks(yticks)
    top = ylims[1] * 1.05
    tick = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([0.3, 0.3, 29.7, 29.7], [top - tick, top, top, top - tick], color="black", linewidth=0.8)
    ax.text(1.5, top, stars(tests[0]["p"]), color=source_color, ha="left", va="bottom", fontsize=TICK_SIZE + 1)
    ax.text(5.0, top, "/ " + stars(tests[1]["p"]), color=target_color, ha="left", va="bottom",
            fontsize=TICK_SIZE + 1)
    ax.legend(handles=[Line2D([], [], color=source_color, lw=2,
                              label=f"{_quoted(source)} → {_quoted(target)} (transfer)"),
                       Line2D([], [], color=target_color, lw=2, label=f"{_quoted(target)} (direct optimization)")],
              loc="lower right", fontsize=TICK_SIZE - 1, frameon=False, handlelength=1.5)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_xlabel("generation", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    save_figure(fig, path)
    table = pd.DataFrame(tests)
    table.to_csv(Path(path).with_name(f"{Path(path).name}_tests.csv"), index=False)
    return table


def render_transfer_examples(path: Path, source: str, target: str, selection: dict) -> None:
    """Images of a source run (generations 0-30) above the same candidates rendered with the target prompt.

    Args:
        path: Output path without extension.
        source: Source prompt.
        target: Target prompt.
        selection: ``source_run``, ``transfer_run``, ``source_images`` and ``transfer_images`` (file names in
            the archives, see ``data/figure_images.json``).
    """
    source_zip = archive(tables("example_prompts") / selection["source_run"], "example_prompts")
    transfer_zip = archive(tables("seed_transfer") / selection["transfer_run"], "seed_transfer")
    rows = [load_images(source_zip, [Path(n).stem for n in selection["source_images"]], size=400),
            load_images(transfer_zip, [Path(n).stem for n in selection["transfer_images"]], size=400)]
    image_grid(rows, [_quoted(source), _quoted(target)], [f"Gen. {g}" for g in GENERATIONS],
               f"{_quoted(source)} → {_quoted(target)}", path)
