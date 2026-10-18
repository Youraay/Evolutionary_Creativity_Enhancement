"""Example prompts: the standard GA with population 100 on twelve prompts (``experiments.example_prompts``).

Output (``plots/example_prompts/``):

    distance_to_current_mean_a_cat     Fig. 1E: distance to the mean of the current generation ("a cat")
    distance_to_initial_mean           Fig. 2C: distance to the mean of generation 0, generations 0/10/20/30,
                                       the three runs pooled, bracket = Mann-Whitney U test of generation 0 vs. 30
    diversity                          Fig. 2F: diversity within the generation, mean +- SEM over the runs
    clip_scores                        Supplementary Fig. 5A: CLIP score with the random baseline (100 images of
                                       the other prompts)
    images_<prompt>_detail, _final     Supplementary Fig. 1A-C: images of generations 0/15/30 of one run and of
                                       generation 30 of every run (selection in ``data/figure_images.json``)
    noise_pca                          Supplementary Fig. 11B: PCA of the initial noise per prompt, colour =
                                       generation, arrows = course of every run (``data/noise_pca.csv``)
    tests.csv                          Mann-Whitney U tests of generation 0 vs. 30 of all prompts and metrics

The noise distributions of Supplementary Fig. 11A are drawn by ``analysis.noise_distribution``.

    python -m analysis.example_prompts [--only curves violins clip_scores images noise_pca tests]
"""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from analysis.common.clip_scores import render_clip_scores
from analysis.common.line_plots import render_line_plot, render_metric_panels
from analysis.common.paths import DATA_DIR, PLOTS_DIR, archive, tables
from analysis.common.plotting import image_grid
from analysis.common.results import load_images
from analysis.common.runs import read_run, runs_by_name
from analysis.common.statistics import generation_test, holm, stars
from analysis.common.style import (COLOR_MAP, LABEL_SIZE, PAGE_WIDTH_IN, TICK_SIZE, TITLE_SIZE, reset_style,
                                   save_figure, set_preview_dir)
from analysis.common.violins import render_pooled_violins

OUT_DIR = PLOTS_DIR / "example_prompts"
PROMPTS = ["a cat", "a dog", "a chair", "a coffee mug", "boredom", "creativity", "a capybara", "a car", "cat",
           "chaos", "knowledge", "love"]
MANUSCRIPT_PROMPTS = ["a cat", "a chair", "boredom"]
"""Prompts shown in the figures of the manuscript."""
ROW = [(0, 0), (0, 1), (0, 2)]
METRICS = {"fitness": "distance to mean of current generation",
           "global_score": "distance to mean of initial generation",
           "diversity_score": "diversity within generation"}


def _quoted(prompt: str) -> str:
    """Prompt in typographic quotes."""
    return f"“{prompt}”"


def runs(prompts: list[str] = MANUSCRIPT_PROMPTS) -> list[dict]:
    """Run configurations of the given prompts, coloured as in all figures.

    Args:
        prompts: Prompts.

    Returns:
        Three runs per prompt (see :func:`analysis.common.runs.runs_by_name`).
    """
    return runs_by_name(tables("example_prompts"), prompts, [COLOR_MAP.get(p, "#000000") for p in prompts])


def render_curves() -> None:
    """Fig. 1E and Fig. 2F."""
    reset_style()
    configs = runs()
    render_line_plot(OUT_DIR / "distance_to_current_mean_a_cat", configs, metric="fitness",
                     ylabel="distance to mean of\ncurrent generation [a.u.]", target_names=["a cat"],
                     yscale=[0.0, 0.1, 0.2], ylims=(0.0, 0.2), height_rows=3)
    render_metric_panels(OUT_DIR / "diversity", configs, MANUSCRIPT_PROMPTS, [COLOR_MAP[p] for p in MANUSCRIPT_PROMPTS],
                         "diversity_score", ncols=3, nrows=1, yscale=[0.1, 0.3], ylims=(0.1, 0.3), positions=ROW,
                         test=True)


def render_violins() -> None:
    """Fig. 2C: distance to the mean of generation 0, three runs pooled."""
    reset_style()
    configs = runs()
    panels = [{"title": _quoted(p), "configs": [c for c in configs if c["name"] == p], "color": COLOR_MAP[p]}
              for p in MANUSCRIPT_PROMPTS]
    render_pooled_violins(OUT_DIR / "distance_to_initial_mean", panels, ylims=(0.0, 0.4), yticks=(0.0, 0.4))


def render_clip() -> None:
    """Supplementary Fig. 5A: CLIP score and random baseline of the three prompts."""
    reset_style(thin_axes=False)
    render_clip_scores(OUT_DIR / "clip_scores", tables("example_prompts") / "clip_scores",
                       [COLOR_MAP[p] for p in MANUSCRIPT_PROMPTS], MANUSCRIPT_PROMPTS, ncols=3, nrows=1,
                       yscale=[0.4, 0.6, 0.8], ylims=(0.4, 0.8), positions=ROW)


def render_images() -> None:
    """Supplementary Fig. 1A-C: selected images of one run (generations 0/15/30) and of all runs (generation 30)."""
    reset_style()
    selection = json.loads((DATA_DIR / "figure_images.json").read_text())["example_prompts"]
    root = tables("example_prompts")
    for prompt in MANUSCRIPT_PROMPTS:
        chosen, slug = selection[prompt], prompt.replace(" ", "_")
        detail = archive(root / chosen["detail_run"], "example_prompts")
        rows = [load_images(detail, [Path(n).stem for n in names], size=400)
                for names in chosen["detail_images"].values()]
        image_grid(rows, [f"Gen. {g}" for g in chosen["detail_images"]], [f"{k + 1}" for k in range(len(rows[0]))],
                   _quoted(prompt), OUT_DIR / f"images_{slug}_detail")
        rows = [load_images(archive(root / run, "example_prompts"), [Path(n).stem for n in names], size=400)
                for run, names in zip(chosen["runs"], chosen["final_images"])]
        image_grid(rows, [f"Run {k + 1}" for k in range(len(rows))], [f"{k + 1}" for k in range(len(rows[0]))],
                   f"{_quoted(prompt)}, generation 30", OUT_DIR / f"images_{slug}_final")


def generation_tests() -> pd.DataFrame:
    """Mann-Whitney U tests of generation 0 vs. 30 for every prompt and metric (three runs pooled).

    Holm correction over all tests of the table.

    Returns:
        One row per prompt and metric; also written to ``tests.csv``.
    """
    rows = []
    for prompt in PROMPTS:
        configs = runs([prompt])
        data = pd.concat([read_run(c) for c in configs], ignore_index=True)
        for metric, label in METRICS.items():
            rows.append({"prompt": prompt, "runs": len(configs), "quantity": label,
                         **generation_test(data, metric, 0, 30)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"])
    table["sig"] = table["p_holm"].map(stars)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "tests.csv", index=False)
    return table


def noise_projection(prompt: str, components: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    """PCA of the flattened initial noise of all candidates of the three runs of a prompt (all generations).

    Args:
        prompt: Prompt.
        components: Number of principal components (randomised solver, as in the original noise analysis).

    Returns:
        ``(projection, explained variance ratio)``; one row per candidate with run, generation, PC1 and PC2.
    """
    archives = [archive(config["path"], "example_prompts") for config in runs([prompt])]
    members = []
    for path in archives:
        with zipfile.ZipFile(path) as zipped:
            members.append(sorted(n for n in zipped.namelist() if n.startswith("/noise/")))
    rows, noise = [], None
    for k, (path, names) in enumerate(zip(archives, members), start=1):
        with zipfile.ZipFile(path) as zipped:
            for name in names:
                values = torch.load(io.BytesIO(zipped.read(name)), map_location="cpu").float().reshape(-1).numpy()
                if noise is None:
                    noise = np.empty((sum(map(len, members)), values.size), dtype=np.float32)
                noise[len(rows)] = values
                rows.append({"prompt": prompt, "run": k, "generation": int(re.search(r"/g(\d+)_", name).group(1))})
    pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
    projection = pca.fit_transform(noise)
    return pd.DataFrame(rows).assign(pc1=projection[:, 0], pc2=projection[:, 1]), pca.explained_variance_ratio_


def render_noise_pca(recompute: bool = False) -> None:
    """Supplementary Fig. 11B: noise PCA per prompt with the course of every run.

    The projections are cached in ``data/noise_pca.csv`` (explained variance in ``data/noise_pca_variance.csv``);
    computing them reads the noise of all candidates from the archives.

    Args:
        recompute: Recompute the cache.
    """
    reset_style()
    cache, variance_csv = OUT_DIR / "data" / "noise_pca.csv", OUT_DIR / "data" / "noise_pca_variance.csv"
    if recompute or not cache.exists():
        parts, variances = [], []
        for prompt in MANUSCRIPT_PROMPTS:
            projection, ratio = noise_projection(prompt)
            parts.append(projection)
            variances.append(pd.DataFrame({"prompt": prompt, "component": np.arange(1, len(ratio) + 1),
                                           "explained_variance_ratio": ratio}))
            print(f"[INFO] noise PCA {prompt}: PC1 {ratio[0]:.1%}, PC2 {ratio[1]:.1%}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).to_csv(cache, index=False)
        pd.concat(variances).to_csv(variance_csv, index=False)
    data = pd.read_csv(cache)
    xlim = (np.floor(data["pc1"].min()), np.ceil(data["pc1"].max()))
    ylim = (np.floor(data["pc2"].min()), np.ceil(data["pc2"].max()))
    pad_x, pad_y = 0.08 * (xlim[1] - xlim[0]), 0.08 * (ylim[1] - ylim[0])

    fig, axes = plt.subplots(1, len(MANUSCRIPT_PROMPTS), figsize=(PAGE_WIDTH_IN, PAGE_WIDTH_IN / 3.2),
                             layout="constrained")
    for ax, prompt in zip(axes, MANUSCRIPT_PROMPTS):
        subset = data[data["prompt"] == prompt].sort_values("generation", kind="stable")
        points = ax.scatter(subset["pc1"], subset["pc2"], c=subset["generation"], cmap="viridis", vmin=0, vmax=30,
                            s=3, alpha=0.7, linewidths=0)
        centroids = subset.groupby(["run", "generation"])[["pc1", "pc2"]].mean()
        for run in sorted(subset["run"].unique()):
            start, end = centroids.loc[(run, 0)].to_numpy(), centroids.loc[(run, 30)].to_numpy()
            direction = (end - start) / np.linalg.norm(end - start)
            normal = np.array([-direction[1] * pad_x, direction[0] * pad_y]) * 1.2
            tail, head = start + 0.3 * (end - start) + normal, end + normal
            ax.annotate("", xy=head, xytext=tail, arrowprops=dict(arrowstyle="-|>", color="#525252", lw=0.8,
                                                                  mutation_scale=7))
            ax.text(*(tail + head) / 2 + 2.2 * normal, f"run {run}", fontsize=TICK_SIZE - 1, color="#525252",
                    ha="center", va="center")
        ax.set_xlim(xlim[0] - pad_x, xlim[1] + pad_x)
        ax.set_ylim(ylim[0] - pad_y, ylim[1] + pad_y)
        ax.set_xticks([xlim[0], 0, xlim[1]], [f"{xlim[0]:.0f}", "0", f"{xlim[1]:.0f}"], fontsize=TICK_SIZE)
        ax.set_yticks([ylim[0], 0, ylim[1]], [f"{ylim[0]:.0f}", "0", f"{ylim[1]:.0f}"], fontsize=TICK_SIZE)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_MAP[prompt])
            spine.set_linewidth(2)
        ax.set_title(_quoted(prompt), color=COLOR_MAP[prompt], fontsize=TITLE_SIZE + 1, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=LABEL_SIZE)
    axes[0].set_ylabel("PC 2", fontsize=LABEL_SIZE)
    colorbar = fig.colorbar(points, ax=axes, fraction=0.03, pad=0.02, ticks=[0, 30])
    colorbar.set_label("Generation", fontsize=LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_SIZE)
    save_figure(fig, OUT_DIR / "noise_pca")


TARGETS = {
    "curves": lambda args: render_curves(),
    "violins": lambda args: render_violins(),
    "clip_scores": lambda args: render_clip(),
    "images": lambda args: render_images(),
    "noise_pca": lambda args: render_noise_pca(args.recompute),
    "tests": lambda args: generation_tests(),
}


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    parser.add_argument("--recompute", action="store_true", help="recompute the noise PCA from the archives")
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    for target in args.only:
        TARGETS[target](args)


if __name__ == "__main__":
    main()
