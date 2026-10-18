"""Distribution of the initial noise values over the generations (Supplementary Fig. 12A and the same figure for
the long runs and the PartiPrompts).

Same figure as the noise histograms of the paper (one row per prompt or class, one column per
generation). Per run and generation all noise values (every candidate x 4 x 128 x 128) are counted into a
fine histogram (bin 0.01 on [-6, 6]) and cached. The plotted density is this histogram smoothed with the
Gaussian kernel ``seaborn.kdeplot`` used on the 500k sample of the original figure (Scott: SD *
500000^(-1/5)); pooling simply adds the counts. In addition, every run gets mean, SD, skewness, excess
kurtosis and a KS test against N(0, 1) on 5000 random values (as in the original normality table).

Output in ``plots/noise_distribution/``:
    example_prompts_noise_hist           Supplementary Fig. 12A: "a cat", "a chair", "boredom" (population 100),
                                         generations 0/10/20/30, 3 seeds pooled
    long_run_noise_hist                  rows = prompts, columns = generations 0/100/200/300 (3 seeds pooled)
    parti_noise_hist_all                 all 114 prompts pooled, generations 0/10/20/30
    parti_noise_hist_challenge/_category one row per class
    by_challenge/, by_category/          one figure per class with one row per prompt
    <experiment>_normality.csv           statistics and KS p per run and generation (+ COMBINED per prompt)

    python -m analysis.noise_distribution [--recompute] [--workers 16] [--what example_prompts long_run parti]
"""
import argparse
import io
import textwrap
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.common.paths import PLOTS_DIR, archive, tables
from analysis.common.plotting import PROMPTS
from analysis.common.results import attach_prompt_meta, discover_parti_runs, discover_runs
from analysis.common.runs import runs_by_name
from analysis.common.statistics import holm
from analysis.common.style import (BOTTOM_MARGIN, COLOR_MAP, FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, RIGHT_MARGIN,
                                   TICK_SIZE, TITLE_SIZE, TOP_MARGIN, save_figure, set_preview_dir)
from analysis.parti_prompts import GROUPINGS, prompt_colors
from experiments.common.naming import slugify_prompt

OUT_DIR = PLOTS_DIR / "noise_distribution"
DATA_DIR = OUT_DIR / "data"
EDGES = np.round(np.linspace(-6, 6, 1201), 6)
BIN = 0.01
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
KS_SAMPLES = 5000
KDE_POINTS = 500_000
"""Sample size of the original kernel density estimate; it determines the kernel bandwidth."""
EXPERIMENTS = {"example_prompts": [0, 10, 20, 30], "long_run": [0, 100, 200, 300], "parti": [0, 10, 20, 30]}

GEN_EDGE = sns.color_palette("husl", 4)
"""Outline colour per generation column (as in the original figure); the fill is the prompt colour."""
POOLED_FILL = "#BDBDBD"
YMAX = 0.42
STYLE = {"axes.grid": False, "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.facecolor": "white",
         "xtick.bottom": True, "ytick.left": True, "xtick.major.size": 3, "ytick.major.size": 3,
         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.color": "black", "ytick.color": "black"}
"""Plain matplotlib look of the original figure (black frame, ticks, no grid)."""

LABEL_STYLES = {
    # short prompts as in the paper figure: rotated, bold, in the row colour
    "rotated": dict(rotation=90, fontsize=LABEL_SIZE - 1, fontweight="bold", multialignment="center"),
    # class names and prompts: horizontal and wrapped, otherwise the rows get uneven heights
    "class": dict(fontsize=TICK_SIZE - 1, fontweight="bold", multialignment="right", linespacing=1.1),
    "prompt": dict(fontsize=TICK_SIZE - 1.5, multialignment="right", linespacing=1.1),
}


# ------------------------------------------------------------------ data
def _histogram_run(item: tuple[dict, str, list[int]]) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """Histogram, statistics and KS sample of all noise values of one run, per generation.

    Args:
        item: ``(run key with prompt and seed, zip path, generations)``.

    Returns:
        ``(statistics rows, histogram counts, KS samples)``, one entry per generation.
    """
    import torch
    from scipy import stats

    key, zip_path, gens = item
    torch.set_num_threads(1)        # otherwise every worker starts as many threads as the node has cores
    rows, counts, samples = [], [], []
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.startswith("/noise/")]
        for generation in gens:
            selected = sorted(n for n in names if n.rsplit("/", 1)[-1].startswith(f"g{generation}_"))
            x = torch.cat([torch.load(io.BytesIO(archive.read(n)), map_location="cpu").float().reshape(-1)
                           for n in selected]).numpy().astype(np.float64)
            index = np.clip(np.floor((x - EDGES[0]) / BIN).astype(np.int64), 0, len(CENTERS) - 1)
            counts.append(np.bincount(index, minlength=len(CENTERS)))
            sample = np.random.default_rng([int(key["seed"]), generation]).choice(x, KS_SAMPLES, replace=False)
            samples.append(sample.astype(np.float32))
            rows.append(dict(**key, generation=generation, n_candidates=len(selected), n_values=len(x), mean=x.mean(),
                             std=x.std(ddof=1), skew=stats.skew(x), kurtosis=stats.kurtosis(x),
                             ks_p=stats.kstest(sample, "norm").pvalue))
    return rows, counts, samples


def _runs(experiment: str) -> pd.DataFrame:
    """Runs of the example prompts ("a cat", "a chair", "boredom"), the long runs or the PartiPrompts experiment."""
    if experiment == "example_prompts":
        configs = runs_by_name(tables("example_prompts"), PROMPTS, [COLOR_MAP[p] for p in PROMPTS])
        return pd.DataFrame([{"prompt": c["name"], "seed": c["seed"].split()[-1],
                              "zip": archive(c["path"], "example_prompts")} for c in configs])
    if experiment == "long_run":
        return discover_runs("long_runs", "variant")
    return discover_parti_runs()


def build(experiment: str, recompute: bool = False, workers: int = 16) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load (or compute and cache) histograms, statistics and KS samples of all runs.

    Args:
        experiment: ``"example_prompts"``, ``"long_run"`` or ``"parti"``.
        recompute: Recompute the cache from the archives.
        workers: Number of processes.

    Returns:
        ``(statistics, counts [rows, bins], KS samples [rows, 5000])``, rows aligned; for the PartiPrompts
        with Category and Challenge.
    """
    cache = DATA_DIR / f"{experiment}_noise_hist.npz"
    stats_csv = DATA_DIR / f"{experiment}_noise_stats.csv"
    if recompute or not cache.exists():
        runs = _runs(experiment)
        items = [({"prompt": r.prompt, "seed": str(r.seed)}, str(r.zip), EXPERIMENTS[experiment])
                 for r in runs.itertuples()]
        print(f"[INFO] {experiment}: {len(items)} runs", flush=True)
        rows, counts, samples = [], [], []
        with ProcessPoolExecutor(workers) as executor:
            for i, (r, c, s) in enumerate(executor.map(_histogram_run, items)):
                rows += r
                counts += c
                samples += s
                if (i + 1) % 25 == 0 or i + 1 == len(items):
                    print(f"[INFO] {i + 1}/{len(items)}", flush=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, counts=np.stack(counts).astype(np.int32), samples=np.stack(samples))
        pd.DataFrame(rows).to_csv(stats_csv, index=False)
    arrays = np.load(cache)
    stats = pd.read_csv(stats_csv, dtype={"seed": str})
    if experiment == "parti":
        stats = attach_prompt_meta(stats)
    return stats, arrays["counts"], arrays["samples"]


def normality_table(stats: pd.DataFrame, samples: np.ndarray, experiment: str) -> pd.DataFrame:
    """Write the statistics per run plus a COMBINED row per prompt (KS on 5000 values from all seeds).

    ``ks_p_holm`` is the Holm-corrected KS p-value, corrected over all runs and generations of the experiment
    (and separately over the COMBINED rows).

    Args:
        stats: Statistics per run and generation.
        samples: KS samples aligned with ``stats``.
        experiment: Key of :data:`EXPERIMENTS` (file name).

    Returns:
        The written table.
    """
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(0)
    rows = []
    for (prompt, generation), index in stats.groupby(["prompt", "generation"]).groups.items():
        subset = stats.loc[index]
        pooled = rng.choice(samples[index].ravel(), KS_SAMPLES, replace=False)
        mean = subset["mean"].mean()          # equal number of values per run -> simple mean
        rows.append({**{c: subset[c].iat[0] for c in ("Challenge", "Category") if c in subset}, "prompt": prompt,
                     "seed": "COMBINED", "generation": generation, "n_values": subset["n_values"].sum(), "mean": mean,
                     "std": np.sqrt((subset["std"] ** 2 + (subset["mean"] - mean) ** 2).mean()),
                     "ks_p": scipy_stats.kstest(pooled, "norm").pvalue})
    out = pd.concat([stats, pd.DataFrame(rows)], ignore_index=True).sort_values(["prompt", "seed", "generation"])
    out["ks_p_holm"] = out.groupby(out["seed"] == "COMBINED")["ks_p"].transform(holm)
    out.to_csv(OUT_DIR / f"{experiment}_normality.csv", index=False)
    smallest = out.loc[out["seed"] != "COMBINED", "ks_p_holm"].min()
    print(f"[INFO] {experiment}: smallest Holm-corrected KS p per run {smallest:.3f}")
    per_generation = stats.groupby("generation").agg(
        runs=("ks_p", "size"), ks_p_below_005=("ks_p", lambda v: (v < 0.05).sum()), mean=("mean", "mean"),
        std=("std", "mean"), skew=("skew", "mean"), kurtosis=("kurtosis", "mean"))
    print(f"[INFO] {experiment}: normality per generation (mean over runs)\n{per_generation.round(4).to_string()}")
    return out


# ------------------------------------------------------------------ figures
def density(counts: np.ndarray) -> np.ndarray:
    """Density from the fine histogram, smoothed like ``seaborn.kdeplot`` on the 500k sample of the original.

    Args:
        counts: Histogram counts on :data:`EDGES`.

    Returns:
        Density at :data:`CENTERS`.
    """
    n = counts.sum()
    mean = (counts * CENTERS).sum() / n
    sd = np.sqrt((counts * (CENTERS - mean) ** 2).sum() / n)
    bandwidth = sd * min(n, KDE_POINTS) ** (-1 / 5)
    k = np.arange(-np.ceil(4 * bandwidth / BIN), np.ceil(4 * bandwidth / BIN) + 1) * BIN
    kernel = np.exp(-0.5 * (k / bandwidth) ** 2)
    return np.convolve(counts / (n * BIN), kernel / kernel.sum(), mode="same")


def _row_label(ax: plt.Axes, text: str, color: str, style: str) -> None:
    """Label left of a row (outside the ``Density`` axis label)."""
    ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction", xytext=(-44, 0), textcoords="offset points",
                ha="right", va="center", color=color, **LABEL_STYLES[style])


def render_grid(rows: list[dict], gens: list[int], name: str, row_h: float, label_style: str = "rotated",
                label_w: float = 0.0) -> None:
    """Draw one row of densities per entry of ``rows`` and one column per generation.

    Args:
        rows: Dicts with ``label``, ``fill``, optional ``label_color`` and ``counts`` (generation -> counts).
        gens: Generations (columns).
        name: Output name relative to ``plots/noise_distribution``.
        row_h: Row height in inches.
        label_style: Key of :data:`LABEL_STYLES`.
        label_w: Extra figure width for the row labels in inches.
    """
    width = len(gens) * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN + label_w
    height = len(rows) * row_h + TOP_MARGIN + BOTTOM_MARGIN
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(rows), len(gens), figsize=(width, height), sharex=True, sharey=True,
                                 squeeze=False, layout="constrained")
        for r, row in enumerate(rows):
            for c, generation in enumerate(gens):
                ax = axes[r, c]
                d = density(row["counts"][generation])
                ax.fill_between(CENTERS, d, color=row["fill"], linewidth=0)
                ax.plot(CENTERS, d, color=GEN_EDGE[c], linewidth=1.5)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                if r == 0:
                    ax.set_title(f"Gen {generation}", fontsize=TITLE_SIZE, fontweight="bold")
                if c == 0:
                    ax.set_ylabel("Density", fontsize=LABEL_SIZE)
                    _row_label(ax, row["label"], row.get("label_color", "black"), label_style)
                if r == len(rows) - 1:
                    ax.set_xlabel("Noise Values", fontsize=LABEL_SIZE)
        ax.set_xlim(-5, 5)
        ax.set_xticks([-5, 0, 5])
        ax.set_ylim(0, YMAX)
        ax.set_yticks([0, YMAX], ["0", f"{YMAX:g}"])
        for a in axes.flat:
            a.tick_params(labelsize=TICK_SIZE)
        save_figure(fig, OUT_DIR / name)


def _counts(stats: pd.DataFrame, counts: np.ndarray, mask: pd.Series, gens: list[int]) -> dict[int, np.ndarray]:
    """Sum the histograms of all selected runs per generation."""
    return {g: counts[(mask & (stats["generation"] == g)).to_numpy()].sum(0) for g in gens}


def _quoted(prompt: str) -> str:
    """``a cat`` -> ``"A cat"`` (row label as in the paper figure)."""
    return f'"{prompt[:1].upper()}{prompt[1:]}"'


def _prompt_label(prompt: str, width: int = 28, max_lines: int = 3) -> str:
    """Wrap a prompt to at most ``max_lines`` lines of ``width`` characters."""
    lines = textwrap.wrap(prompt, width)
    return "\n".join(lines[:max_lines]) + (" ..." if len(lines) > max_lines else "")


def render_example_prompts(stats: pd.DataFrame, counts: np.ndarray) -> None:
    """Rows = "a cat", "a chair", "boredom", columns = generations 0/10/20/30 (Supplementary Fig. 12A)."""
    gens = EXPERIMENTS["example_prompts"]
    rows = [dict(label=_quoted(p), fill=COLOR_MAP[p], label_color=COLOR_MAP[p],
                 counts=_counts(stats, counts, stats["prompt"] == p, gens)) for p in PROMPTS]
    render_grid(rows, gens, "example_prompts_noise_hist", row_h=FIXED_COL_WIDTH * 0.7)


def render_long_run(stats: pd.DataFrame, counts: np.ndarray) -> None:
    """Rows = long-run prompts, columns = generations 0/100/200/300."""
    gens = EXPERIMENTS["long_run"]
    rows = [dict(label=_quoted(p), fill=COLOR_MAP[p], label_color=COLOR_MAP[p],
                 counts=_counts(stats, counts, stats["prompt"] == p, gens)) for p in PROMPTS]
    render_grid(rows, gens, "long_run_noise_hist", row_h=FIXED_COL_WIDTH * 0.7)


def render_parti(stats: pd.DataFrame, counts: np.ndarray) -> None:
    """All prompts pooled, one row per class, and one figure per class with one row per prompt."""
    gens = EXPERIMENTS["parti"]
    every = pd.Series(True, index=stats.index)
    render_grid([dict(label=f"All prompts\n(n = {stats['prompt'].nunique()})", fill=POOLED_FILL,
                      counts=_counts(stats, counts, every, gens))], gens, "parti_noise_hist_all",
                row_h=FIXED_COL_WIDTH * 0.7, label_style="class", label_w=0.5)
    for group, (col, order) in GROUPINGS.items():
        rows = [dict(label=textwrap.fill(g, 14), fill=POOLED_FILL, counts=_counts(stats, counts, stats[col] == g, gens))
                for g in order if (stats[col] == g).any()]
        render_grid(rows, gens, f"parti_noise_hist_{group}", row_h=0.7, label_style="class", label_w=0.6)
        colors = prompt_colors(stats, col, order)
        for g in order:
            prompts = sorted(stats.loc[stats[col] == g, "prompt"].unique(), key=str.lower)
            rows = [dict(label=_prompt_label(p), fill=colors[p],
                         counts=_counts(stats, counts, stats["prompt"] == p, gens)) for p in prompts]
            render_grid(rows, gens, f"by_{group}/{slugify_prompt(g)}", row_h=0.6, label_style="prompt", label_w=1.2)


RENDERERS = {"example_prompts": render_example_prompts, "long_run": render_long_run, "parti": render_parti}


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="recompute the histogram cache")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--what", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for experiment in args.what:
        stats, counts, samples = build(experiment, args.recompute, args.workers)
        normality_table(stats, samples, experiment)
        RENDERERS[experiment](stats, counts)


if __name__ == "__main__":
    main()
