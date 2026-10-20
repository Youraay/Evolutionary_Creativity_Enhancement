"""Think-Then-Generate (T2G) + seed optimisation with SDXL compared with the original prompt (Supplementary Fig. 10).

Conditions (3 seeds per prompt, same GA setup, same initial noise per seed):
    original prompt           long run with the original prompt, generation 0
    seed optimization         long run, generation 30 (or 100)
    reasoning                 T2G-rewritten prompt (experiments/think_then_generate), generation 0
    reasoning + seed opt.     T2G run, generation 30 (or 100)

Metrics:
    divergence  distance to the mean of generation 0 of the ORIGINAL prompt (same seed) in BLIP-2 space, a
                common reference for all conditions; recomputed for the T2G runs from the BLIP-2 embeddings of
                the archives (identical to ``model_MMD_blip2`` of the long run for the original prompt)
    diversity   diversity within the generation (BLIP-2)
    fidelity    CLIP score against the original prompt, random baseline as for the long runs

Output in ``plots/think_then_generate/``:
    t2g_<metric>_g30 / _g100       one panel per prompt, original vs. reasoning
    t2g_conditions_g30 / _g100     the four conditions side by side (rows = metrics, columns = prompts)
    t2g_conditions_g*_tests.csv    tests generation 0 vs. 30/100 without and with reasoning (stars)
    t2g_supplement_g30             Supplementary Fig. 10: four conditions and the course over 30 generations
    t2g_pairwise_g30 / _g100       pairwise tests of all four conditions (colour = effect size, text = stars);
                                   table in t2g_pairwise_g*.csv

    python -m analysis.think_then_generate [--recompute]
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.common.paths import PLOTS_DIR
from analysis.common.plotting import PROMPTS, curve, set_ticks
from analysis.common.results import discover_runs, load_embedding_experiment, load_runs
from analysis.common.statistics import holm, stars
from analysis.common.style import (COLOR_MAP, FIXED_COL_WIDTH, LABEL_SIZE, LEFT_MARGIN, PAGE_WIDTH_IN, RIGHT_MARGIN,
                                   TICK_SIZE, TITLE_SIZE, save_figure, set_preview_dir)
from analysis.long_runs import long_run_baseline

OUT_DIR = PLOTS_DIR / "think_then_generate"
DATA_DIR = OUT_DIR / "data"

ORIG_COLOR = COLOR_MAP
"""Original prompt: the prompt colour of all figures."""
T2G_COLOR = {"a cat": "#1F78B4", "a chair": "#E6550D", "boredom": "#54278F"}
"""Reasoning: the dark partner of the prompt colour in the same Paired palette."""
LEGEND_GREY = {"original prompt": "#BDBDBD", "reasoning (T2G)": "#525252"}

LABELS = {
    "divergence": "distance to mean of initial\ngeneration (original prompt) [a.u.]",
    "diversity": "diversity within\ngeneration [a.u.]",
    "fidelity": "Fidelity score [a.u.]",
}
CONDITIONS = ["original prompt", "seed optimization", "reasoning", "reasoning + seed opt."]
COMPARISONS = {"original prompt -> seed optimization": "lr", "reasoning -> reasoning + seed opt.": "t2g"}
"""Generation 0 -> generation ``gen`` without and with reasoning (the arrows of the condition plot)."""


def _blip2(archive: zipfile.ZipFile, names: list[str]):
    """Stack BLIP-2 embeddings of an archive into a float tensor ``[len(names), 1408]``."""
    import torch

    return torch.cat([torch.load(io.BytesIO(archive.read(n)), map_location="cpu").float().reshape(1, -1)
                      for n in names])


def common_divergence(lr_runs: pd.DataFrame, t2g_runs: pd.DataFrame, max_gen: int) -> pd.DataFrame:
    """Distance of the T2G candidates to the generation-0 mean of the long run with the original prompt (same seed).

    Args:
        lr_runs: Long runs (reference).
        t2g_runs: T2G runs with the original prompt in ``prompt``.
        max_gen: Last generation to include.

    Returns:
        One row per T2G candidate with prompt, seed, file_name and divergence.
    """
    import torch

    rows = []
    for run in t2g_runs.itertuples():
        reference_run = lr_runs[(lr_runs["prompt"] == run.prompt) & (lr_runs["seed"] == run.seed)].iloc[0]
        with zipfile.ZipFile(reference_run["zip"]) as archive:
            reference = _blip2(archive, sorted(n for n in archive.namelist() if n.startswith("/blip2/g0_")))
            reference = reference.mean(0, keepdim=True)
        with zipfile.ZipFile(run.zip) as archive:
            names = [n for n in archive.namelist() if n.startswith("/blip2/g")
                     and int(n.split("/")[-1].split("_")[0][1:]) <= max_gen]
            embeddings = _blip2(archive, names)
        divergence = ((1 - torch.nn.functional.cosine_similarity(embeddings, reference)) / 2).numpy()
        rows.append(pd.DataFrame({"prompt": run.prompt, "seed": run.seed,
                                  "file_name": [Path(n).stem for n in names], "divergence": divergence}))
        print(f"[INFO] {run.prompt} {run.seed}: {len(names)} embeddings", flush=True)
    return pd.concat(rows, ignore_index=True)


def load(recompute: bool = False, max_gen: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the long runs (original prompt), the T2G runs (SDXL, 100 generations) and the baseline.

    Args:
        recompute: Recompute the cached common divergence.
        max_gen: Last generation to include.

    Returns:
        ``(long runs, T2G runs, random baseline)`` with the columns divergence, diversity and fidelity.

    Raises:
        ValueError: If T2G candidates have no recomputed divergence.
    """
    lr_all, lr_runs = load_embedding_experiment("long_runs")
    baseline = long_run_baseline(lr_all, lr_runs)
    lr = lr_all[lr_all["generation"] <= max_gen].copy()
    t2g_runs = discover_runs("think_then_generate", "variant")
    t2g_runs = t2g_runs[t2g_runs["variant"] == "sdxl_t2g_g100"].copy()
    # <run>.json stores the generation prompt (rewritten); the original prompt is in <run>_t2g.json
    t2g_runs["prompt"] = [json.loads(c.with_name(f"{c.stem}_t2g.json").read_text())["original_prompt"]
                          for c in t2g_runs["csv"]]
    t2g = load_runs(t2g_runs, "variant", {"model_LocalMaxMeanDivergence": "divergence_own",
                                          "model_DiversityWithinPopulation": "diversity"})
    print(f"[INFO] T2G: {t2g.groupby(['prompt', 'seed']).ngroups} runs")

    cache = DATA_DIR / "t2g_divergence_common.parquet"
    if recompute or not cache.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        common_divergence(lr_runs, t2g_runs, max_gen).to_parquet(cache, index=False)
    t2g = t2g.merge(pd.read_parquet(cache), on=["prompt", "seed", "file_name"], how="left", validate="1:1")
    if t2g["divergence"].isna().any():
        raise ValueError("Candidates without recomputed divergence; rebuild the cache with --recompute")

    lr = lr.rename(columns={"model_CLIPScore": "fidelity"})
    t2g = t2g.rename(columns={"model_CLIPScore": "fidelity"})
    return lr, t2g, baseline


def render_trajectories(lr: pd.DataFrame, t2g: pd.DataFrame, baseline: pd.DataFrame, max_gen: int) -> None:
    """One panel per prompt: original prompt vs. reasoning, up to ``max_gen``.

    Args:
        lr: Long runs.
        t2g: T2G runs.
        baseline: Random baseline (dashed in the fidelity panels).
        max_gen: Last generation shown.
    """
    size = (3 * FIXED_COL_WIDTH + LEFT_MARGIN + RIGHT_MARGIN, 2.4)
    labels = {**LABELS, "divergence": "distance to mean of\ninitial generation\n(original prompt) [a.u.]"}
    for key in ("divergence", "diversity", "fidelity"):
        fig, axes = plt.subplots(1, 3, figsize=size, sharey=True, layout="constrained")
        for ax, prompt in zip(axes, PROMPTS):
            curve(ax, lr[(lr["prompt"] == prompt) & (lr["generation"] <= max_gen)], key, ORIG_COLOR[prompt], lw=1.5)
            curve(ax, t2g[(t2g["prompt"] == prompt) & (t2g["generation"] <= max_gen)], key, T2G_COLOR[prompt], lw=1.5)
            if key == "fidelity":
                b = baseline.loc[baseline["prompt"] == prompt, "baseline_clip_score"]
                ax.axhline(b.mean(), color=ORIG_COLOR[prompt], linestyle="--", linewidth=1.5)
            ax.set_title(prompt, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
            set_ticks(ax, [0, max_gen // 2, max_gen])
            ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
        axes[0].set_ylabel(labels[key], fontsize=LABEL_SIZE)
        handles = [Line2D([], [], color=c, lw=1.5, label=label) for label, c in LEGEND_GREY.items()]
        if key == "fidelity":
            handles.append(Line2D([], [], color=LEGEND_GREY["original prompt"], lw=1.5, ls="--",
                                  label="random baseline"))
        fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), fontsize=TICK_SIZE - 1,
                   frameon=False, handlelength=1.8)
        save_figure(fig, OUT_DIR / f"t2g_{key}_g{max_gen}")


def condition_summary(lr: pd.DataFrame, t2g: pd.DataFrame, gen: int) -> pd.DataFrame:
    """Mean, SD and count over all candidates (the 3 seeds pooled) for the four conditions.

    Args:
        lr: Long runs.
        t2g: T2G runs.
        gen: Generation of the optimised conditions.

    Returns:
        One row per prompt and condition.
    """
    parts = []
    for condition, df, g in ((CONDITIONS[0], lr, 0), (CONDITIONS[1], lr, gen), (CONDITIONS[2], t2g, 0),
                             (CONDITIONS[3], t2g, gen)):
        subset = df[df["generation"] == g]
        stats = subset.groupby("prompt")[["divergence", "diversity", "fidelity"]].agg(["mean", "std", "count"])
        stats.columns = [f"{m}_{a}" for m, a in stats.columns]
        parts.append(stats.reset_index().assign(condition=condition.replace("\n", " "), generation=g))
    return pd.concat(parts, ignore_index=True)


def condition_tests(lr: pd.DataFrame, t2g: pd.DataFrame, gen: int) -> pd.DataFrame:
    """Tests of generation 0 vs. ``gen`` per metric, without and with reasoning.

    per prompt   two-sided Mann-Whitney U on all candidates, as in the paper, the 3 seeds pooled (180
                 candidates each)
    all prompts  Wilcoxon signed-rank on the run means, paired per prompt x seed (n = 9); does not treat
                 the candidates of a run as independent
    effect_r     rank-biserial correlation (paired variant for Wilcoxon), > 0: generation ``gen`` higher

    Holm correction separately per test family (18 and 6 tests of a figure).

    Args:
        lr: Long runs.
        t2g: T2G runs.
        gen: Generation compared with generation 0.

    Returns:
        One row per metric, prompt (or ``"all prompts"``) and comparison with statistic, effect size, p,
        Holm-adjusted p and significance label.
    """
    from scipy.stats import mannwhitneyu, rankdata, wilcoxon

    data = {"lr": lr, "t2g": t2g}
    rows = []
    for key in ("divergence", "diversity", "fidelity"):
        for comparison, source in COMPARISONS.items():
            df = data[source]
            start, end = df[df["generation"] == 0], df[df["generation"] == gen]
            for prompt in PROMPTS:
                a, b = start.loc[start["prompt"] == prompt, key], end.loc[end["prompt"] == prompt, key]
                u, pv = mannwhitneyu(b, a, alternative="two-sided")
                rows.append(dict(metric=key, prompt=prompt, comparison=comparison, test="Mann-Whitney U (candidates)",
                                 n_start=len(a), n_end=len(b), mean_start=a.mean(), mean_end=b.mean(),
                                 statistic=u, effect_r=2 * u / (len(a) * len(b)) - 1, p=pv))
            a = start.groupby(["prompt", "seed"])[key].mean()
            b = end.groupby(["prompt", "seed"])[key].mean().reindex(a.index)
            w, pv = wilcoxon(b, a, alternative="two-sided")
            d = (b - a).to_numpy()
            ranks = rankdata(np.abs(d))
            rows.append(dict(metric=key, prompt="all prompts", comparison=comparison,
                             test="Wilcoxon signed-rank (run means)", n_start=len(a), n_end=len(b),
                             mean_start=a.mean(), mean_end=b.mean(), statistic=w,
                             effect_r=(ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum(), p=pv))
    result = pd.DataFrame(rows)
    result.insert(result.columns.get_loc("mean_end") + 1, "delta", result["mean_end"] - result["mean_start"])
    result["p_holm"] = result.groupby("test")["p"].transform(holm)
    result["sig"] = result["p_holm"].map(stars)
    return result


def condition_data(lr: pd.DataFrame, t2g: pd.DataFrame, gen: int) -> dict[str, pd.DataFrame]:
    """Candidates of the four conditions.

    Args:
        lr: Long runs.
        t2g: T2G runs.
        gen: Generation of the optimised conditions.

    Returns:
        Condition (see :data:`CONDITIONS`) -> candidates.
    """
    return {CONDITIONS[0]: lr[lr["generation"] == 0], CONDITIONS[1]: lr[lr["generation"] == gen],
            CONDITIONS[2]: t2g[t2g["generation"] == 0], CONDITIONS[3]: t2g[t2g["generation"] == gen]}


def pairwise_tests(lr: pd.DataFrame, t2g: pd.DataFrame, gen: int) -> pd.DataFrame:
    """Tests between every pair of the four conditions, per metric.

    per prompt   two-sided Mann-Whitney U on all candidates, the 3 seeds pooled (180 candidates per condition)
    all prompts  Wilcoxon signed-rank on the run means, paired per prompt x seed (n = 9); all conditions start
                 from the same initial noise per seed
    effect_r     rank-biserial correlation (paired variant for Wilcoxon), > 0: ``condition_b`` higher

    Holm correction over the six comparisons of a metric and prompt (and of a metric for all prompts). With
    n = 9 the smallest possible Wilcoxon p is 0.0039, i.e. 0.023 after the correction.

    Args:
        lr: Long runs.
        t2g: T2G runs.
        gen: Generation of the optimised conditions.

    Returns:
        One row per metric, prompt (or ``"all prompts"``) and pair of conditions.
    """
    from itertools import combinations

    from scipy.stats import mannwhitneyu, rankdata, wilcoxon

    data = condition_data(lr, t2g, gen)
    rows = []
    for key in ("divergence", "diversity", "fidelity"):
        for first, second in combinations(CONDITIONS, 2):
            a_all, b_all = data[first], data[second]
            for prompt in PROMPTS:
                a = a_all.loc[a_all["prompt"] == prompt, key]
                b = b_all.loc[b_all["prompt"] == prompt, key]
                u, pv = mannwhitneyu(b, a, alternative="two-sided")
                rows.append(dict(metric=key, prompt=prompt, condition_a=first, condition_b=second,
                                 test="Mann-Whitney U (candidates)", n_a=len(a), n_b=len(b), mean_a=a.mean(),
                                 mean_b=b.mean(), statistic=u, effect_r=2 * u / (len(a) * len(b)) - 1, p=pv))
            a = a_all.groupby(["prompt", "seed"])[key].mean()
            b = b_all.groupby(["prompt", "seed"])[key].mean().reindex(a.index)
            w, pv = wilcoxon(b, a, alternative="two-sided")
            d = (b - a).to_numpy()
            ranks = rankdata(np.abs(d))
            rows.append(dict(metric=key, prompt="all prompts", condition_a=first, condition_b=second,
                             test="Wilcoxon signed-rank (run means)", n_a=len(a), n_b=len(b), mean_a=a.mean(),
                             mean_b=b.mean(), statistic=w,
                             effect_r=(ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum(), p=pv))
    result = pd.DataFrame(rows)
    result.insert(result.columns.get_loc("mean_b") + 1, "delta", result["mean_b"] - result["mean_a"])
    result["p_holm"] = result.groupby(["metric", "prompt"])["p"].transform(holm)
    result["sig"] = result["p_holm"].map(stars)
    return result


def render_pairwise(lr: pd.DataFrame, t2g: pd.DataFrame, gen: int) -> pd.DataFrame:
    """Matrix of the pairwise tests per metric (rows) and prompt (columns, last: all prompts).

    Cell (row = condition b, column = condition a): colour = effect size (red: b higher, blue: b lower), text =
    Holm-adjusted significance.

    Args:
        lr: Long runs.
        t2g: T2G runs.
        gen: Generation of the optimised conditions.

    Returns:
        The tests (also written to ``t2g_pairwise_g<gen>.csv``).
    """
    tests = pairwise_tests(lr, t2g, gen)
    tests.to_csv(OUT_DIR / f"t2g_pairwise_g{gen}.csv", index=False)
    labels = [c.replace(" + ", " +\n") for c in CONDITIONS]
    columns = PROMPTS + ["all prompts"]
    keys = ["divergence", "diversity", "fidelity"]
    fig, axes = plt.subplots(len(keys), len(columns), figsize=(PAGE_WIDTH_IN * 1.15, 5.0), layout="constrained")
    cmap = plt.get_cmap("RdBu_r")
    for r, key in enumerate(keys):
        for c, prompt in enumerate(columns):
            ax = axes[r, c]
            matrix = np.full((len(CONDITIONS), len(CONDITIONS)), np.nan)
            subset = tests[(tests["metric"] == key) & (tests["prompt"] == prompt)]
            for row in subset.itertuples():
                i, j = CONDITIONS.index(row.condition_b), CONDITIONS.index(row.condition_a)
                matrix[i, j] = row.effect_r
                ax.text(j, i, row.sig, ha="center", va="center", fontsize=TICK_SIZE - 1,
                        color="white" if abs(row.effect_r) > 0.6 else "black")
            image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
            ax.set_xticks(range(len(CONDITIONS)), labels if r == len(keys) - 1 else [""] * len(CONDITIONS),
                          rotation=40, ha="right", rotation_mode="anchor", fontsize=TICK_SIZE - 2)
            ax.set_yticks(range(len(CONDITIONS)), labels if c == 0 else [""] * len(CONDITIONS), fontsize=TICK_SIZE - 2)
            ax.set_xlim(-0.5, len(CONDITIONS) - 1.5)
            ax.set_ylim(len(CONDITIONS) - 0.5, 0.5)
            ax.grid(False)
            ax.tick_params(length=0)
            if r == 0:
                ax.set_title(prompt, fontsize=TITLE_SIZE - 1, fontweight="bold")
            if c == len(columns) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(key, fontsize=LABEL_SIZE, rotation=270, labelpad=12)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, ticks=[-1, 0, 1])
    colorbar.set_label("effect size r (row vs. column)", fontsize=LABEL_SIZE - 1)
    colorbar.ax.tick_params(labelsize=TICK_SIZE - 1)
    fig.suptitle(f"Pairwise tests (seed optimization: generation {gen}); per prompt Mann-Whitney U on the "
                 "candidates,\nall prompts Wilcoxon signed-rank on the run means; Holm correction per panel",
                 fontsize=TITLE_SIZE - 1)
    save_figure(fig, OUT_DIR / f"t2g_pairwise_g{gen}")
    return tests


def render_conditions(lr: pd.DataFrame, t2g: pd.DataFrame, baseline: pd.DataFrame, gen: int) -> None:
    """Four conditions per prompt: point = mean with error bar.

    Arrows show the seed optimisation from generation 0 to ``gen``; stars give the Holm-adjusted
    Mann-Whitney U tests of :func:`condition_tests`.

    Args:
        lr: Long runs.
        t2g: T2G runs.
        baseline: Random baseline (dashed in the fidelity row).
        gen: Generation of the optimised conditions.
    """
    summary = condition_summary(lr, t2g, gen)
    summary.to_csv(OUT_DIR / f"t2g_conditions_g{gen}.csv", index=False)
    tests = condition_tests(lr, t2g, gen)
    tests.to_csv(OUT_DIR / f"t2g_conditions_g{gen}_tests.csv", index=False)
    significance = tests.set_index(["metric", "prompt", "comparison"])["sig"]
    keys = ["divergence", "diversity", "fidelity"]
    fig, axes = plt.subplots(len(keys), 3, figsize=(PAGE_WIDTH_IN, 6.0), sharey="row", layout="constrained")
    x = np.arange(len(CONDITIONS))
    for c, prompt in enumerate(PROMPTS):
        s = summary[summary["prompt"] == prompt].set_index("condition").reindex(
            [k.replace("\n", " ") for k in CONDITIONS])
        colors = [ORIG_COLOR[prompt], ORIG_COLOR[prompt], T2G_COLOR[prompt], T2G_COLOR[prompt]]
        for r, key in enumerate(keys):
            ax = axes[r, c]
            mean, sd = s[f"{key}_mean"].to_numpy(), s[f"{key}_std"].to_numpy()
            for i, comparison in zip((0, 2), COMPARISONS):
                ax.annotate("", xy=(x[i + 1], mean[i + 1]), xytext=(x[i], mean[i]),
                            arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=1, shrinkA=4, shrinkB=4,
                                            mutation_scale=8))
                ax.annotate(significance[(key, prompt, comparison)], xy=(x[i] + 0.5, max(mean[i:i + 2] + sd[i:i + 2])),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=TICK_SIZE - 1, color="#525252")
            for i in range(4):
                ax.errorbar(x[i], mean[i], yerr=sd[i], fmt="o", color=colors[i], markersize=5, capsize=2.5,
                            elinewidth=1, markerfacecolor=colors[i] if i % 2 else "white", markeredgewidth=1.2,
                            zorder=3)
            if key == "fidelity":
                ax.axhline(baseline.loc[baseline["prompt"] == prompt, "baseline_clip_score"].mean(),
                           color=ORIG_COLOR[prompt], linestyle="--", linewidth=1)
            ax.set_xlim(-0.5, 3.5)
            ax.set_xticks(x, CONDITIONS if r == len(keys) - 1 else [""] * 4, fontsize=TICK_SIZE - 1,
                          rotation=35, ha="right", rotation_mode="anchor")
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="y", labelsize=TICK_SIZE)
            if r == 0:
                ax.set_title(prompt, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
            if c == 0:
                ax.set_ylabel(LABELS[key], fontsize=LABEL_SIZE - 1)
    for row in axes:            # room for the stars above the highest error bar
        low, high = row[0].get_ylim()
        row[0].set_ylim(low, high + 0.12 * (high - low))
    handles = [Line2D([], [], marker="o", ls="", color="#525252", markerfacecolor="white", label="Gen. 0"),
               Line2D([], [], marker="o", ls="", color="#525252", label=f"Gen. {gen}"),
               Line2D([], [], color="#525252", lw=1, ls="--", label="random baseline")]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, fontsize=TICK_SIZE - 1, frameon=False)
    save_figure(fig, OUT_DIR / f"t2g_conditions_g{gen}")


SUPPLEMENT_LABELS = ["original prompt - initial", "original prompt - optimized", "reasoning - initial",
                     "reasoning - optimized"]
"""Condition labels of Supplementary Fig. 10A, in the order of :data:`CONDITIONS`."""


def render_supplement(lr: pd.DataFrame, t2g: pd.DataFrame, gen: int = 30) -> None:
    """Supplementary Fig. 10: (A) the four conditions, (B) the course over the generations.

    (A) point = mean with error bar; arrows show the seed optimisation from generation 0 to ``gen``. (B) original
    prompt (dashed) and reasoning (solid) over the generations. Rows: divergence (distance to the mean of
    generation 0 of the original prompt) and diversity within the generation.

    Args:
        lr: Long runs.
        t2g: T2G runs.
        gen: Generation of the optimised conditions and last generation of (B).
    """
    data = condition_data(lr, t2g, gen)
    keys = ["divergence", "diversity"]
    labels = {"divergence": "distance to mean of initial\ngeneration [a.u.]",
              "diversity": "diversity within\ngeneration [a.u.]"}
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_WIDTH_IN * 1.55), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1])
    axes_a = top.subplots(len(keys), len(PROMPTS), sharey=True)
    axes_b = bottom.subplots(len(keys), len(PROMPTS), sharey="row")
    top.suptitle("A", x=0.0, ha="left", fontweight="bold", fontsize=TITLE_SIZE + 2)
    bottom.suptitle("B", x=0.0, ha="left", fontweight="bold", fontsize=TITLE_SIZE + 2)
    x = np.arange(len(CONDITIONS))
    for c, prompt in enumerate(PROMPTS):
        colors = [ORIG_COLOR[prompt], ORIG_COLOR[prompt], T2G_COLOR[prompt], T2G_COLOR[prompt]]
        for r, key in enumerate(keys):
            ax = axes_a[r, c]
            values = [data[condition].loc[data[condition]["prompt"] == prompt, key] for condition in CONDITIONS]
            mean = np.array([v.mean() for v in values])
            sd = np.array([v.std(ddof=1) for v in values])
            for i in (0, 2):
                ax.annotate("", xy=(x[i + 1], mean[i + 1]), xytext=(x[i], mean[i]),
                            arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=1, shrinkA=4, shrinkB=4,
                                            mutation_scale=8))
            for i in range(len(CONDITIONS)):
                ax.errorbar(x[i], mean[i], yerr=sd[i], fmt="o", color=colors[i], markersize=5, capsize=2.5,
                            elinewidth=1, markerfacecolor=colors[i] if i % 2 else "white", markeredgewidth=1.2,
                            zorder=3)
            ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
            ax.set_xticks(x, SUPPLEMENT_LABELS if r == len(keys) - 1 else [""] * len(CONDITIONS),
                          fontsize=TICK_SIZE - 1, rotation=35, ha="right", rotation_mode="anchor")
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="y", labelsize=TICK_SIZE)
            if r == 0:
                ax.set_title(prompt, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(labels[key], fontsize=LABEL_SIZE - 1)
            ax = axes_b[r, c]
            for df, color, style in ((lr, ORIG_COLOR[prompt], "--"), (t2g, T2G_COLOR[prompt], "-")):
                curve(ax, df[(df["prompt"] == prompt) & (df["generation"] <= gen)], key, color, lw=1.5)
                ax.lines[-1].set_linestyle(style)
            set_ticks(ax, [0, gen // 2, gen])
            ax.set_xlabel("generation" if r == len(keys) - 1 else "", fontsize=LABEL_SIZE)
            if r == 0:
                ax.set_title(prompt, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(labels[key], fontsize=LABEL_SIZE - 1)
    axes_a[0, 0].legend(handles=[Line2D([], [], marker="o", ls="", color="#525252", markerfacecolor="white",
                                        label="gen. 0"),
                                 Line2D([], [], marker="o", ls="", color="#525252", label=f"gen. {gen}")],
                        loc="upper left", fontsize=TICK_SIZE - 1, frameon=False)
    axes_b[0, 0].legend(handles=[Line2D([], [], color="#969696", lw=1.5, ls="--", label="original prompt"),
                                 Line2D([], [], color="#252525", lw=1.5, label="reasoning")],
                        loc="upper left", fontsize=TICK_SIZE - 1, frameon=False)
    save_figure(fig, OUT_DIR / f"t2g_supplement_g{gen}")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="recompute the common divergence")
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    lr, t2g, baseline = load(args.recompute)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for gen in (30, 100):
        render_trajectories(lr, t2g, baseline, gen)
        render_conditions(lr, t2g, baseline, gen)
        render_pairwise(lr, t2g, gen)
    render_supplement(lr, t2g, 30)


if __name__ == "__main__":
    main()
