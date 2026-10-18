"""PCA of the initial noise and the BLIP-2 embeddings of the PartiPrompts experiment, on three levels.

    separate   one PCA per prompt (3 seeds, all 31 generations), colour = generation
    challenge  joint PCA of all prompts of a challenge, colour = prompt, light -> dark = generation
               0 -> 30, plus the centroid trajectory of every prompt as arrow
    all        joint PCA of all prompts, trajectories per prompt and per challenge, colour = challenge

The joint noise PCAs use every 5th generation (0, 5, ..., 30), otherwise the data would be about 80 GB;
BLIP-2 uses all generations. As in the original noise analysis without standardisation, PCA with 10
components (randomised; IncrementalPCA for all noises together).

    # compute (compute node, about 1.5 h): sbatch analysis/submit_pca.sh, or
    python -m analysis.pca_parti_prompts compute
    # plot (login node):
    python -m analysis.pca_parti_prompts plot

Output: projections in ``PCA_DATA`` (``projections.parquet``, ``explained_variance.json``), figures in
``plots/parti_prompts_pca/``.
"""
import argparse
import io
import json
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

from analysis.common.paths import PCA_DATA, PLOTS_DIR, PROMPT_TABLE
from analysis.common.results import discover_parti_runs
from analysis.common.style import LABEL_SIZE, PAGE_WIDTH_IN, TICK_SIZE, TITLE_SIZE, save_figure, set_preview_dir
from analysis.parti_prompts import CHALLENGE_ORDER, prompt_colors
from experiments.common.naming import slugify_prompt

OUT_DIR = PLOTS_DIR / "parti_prompts_pca"
JOINT_GENS = list(range(0, 31, 5))
N_COMPONENTS = 10
SPACES = {"noise": "Initial noise", "blip2": "BLIP-2 embedding"}
GEN_CMAP = {"noise": "viridis", "blip2": "plasma"}
"""Colour maps of the per-prompt plots, as in the original noise analysis."""
CHALLENGE_COLORS = dict(zip(CHALLENGE_ORDER, [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"]))


# ------------------------------------------------------------------ compute
def _load_run(zip_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read noise and BLIP-2 embedding of every candidate of a run.

    Args:
        zip_path: Result archive.

    Returns:
        ``(file stems, generations, noise [N, 65536] float16, blip2 [N, 1408] float16)``; bfloat16 ->
        float32 -> float16 is lossless for standard normal noise.
    """
    import torch

    with zipfile.ZipFile(zip_path) as archive:
        stems = sorted({Path(n).stem for n in archive.namelist() if n.startswith("/noise/")})
        noise = np.empty((len(stems), 4 * 128 * 128), np.float16)
        blip = np.empty((len(stems), 1408), np.float16)
        for i, stem in enumerate(stems):
            noise[i] = torch.load(io.BytesIO(archive.read(f"/noise/{stem}.pt")),
                                  map_location="cpu").float().reshape(-1).numpy()
            blip[i] = torch.load(io.BytesIO(archive.read(f"/blip2/{stem}.pt")),
                                 map_location="cpu").float().reshape(-1).numpy()
    generations = np.array([int(s.split("_")[0][1:]) for s in stems], np.int16)
    return stems, generations, noise, blip


def _process_prompt(item: tuple[str, list[tuple[str, str]]]) -> dict:
    """Load all runs of one prompt, fit its own PCA per space and return the data for the joint PCAs.

    Args:
        item: ``(prompt, [(seed, zip path), ...])``.

    Returns:
        Dict with ``meta`` (one row per candidate with the per-prompt projections), the explained variance
        per space, the noise of the joint generations and all BLIP-2 embeddings.
    """
    from sklearn.decomposition import PCA

    prompt, runs = item
    metas, noises, blips = [], [], []
    for seed, zip_path in runs:
        stems, generations, noise, blip = _load_run(Path(zip_path))
        metas.append(pd.DataFrame({"prompt": prompt, "seed": seed, "generation": generations, "file": stems}))
        noises.append(noise)
        blips.append(blip)
    meta = pd.concat(metas, ignore_index=True)
    noise, blip = np.concatenate(noises), np.concatenate(blips)
    out = {"meta": meta}
    for space, X in (("noise", noise), ("blip2", blip)):
        pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=0)
        projection = pca.fit_transform(X.astype(np.float32))
        meta[f"sep_pc1_{space}"], meta[f"sep_pc2_{space}"] = projection[:, 0], projection[:, 1]
        out[f"evr_{space}"] = pca.explained_variance_ratio_.tolist()
    subset = meta["generation"].isin(JOINT_GENS).to_numpy()
    out["noise_sub"], out["sub"], out["blip"] = noise[subset], subset, blip
    return out


def compute(workers: int = 12, limit: int | None = None) -> None:
    """Fit all PCAs and store the first two components of every candidate.

    Args:
        workers: Number of processes that load the prompts in parallel.
        limit: Only the first ``limit`` prompts (smoke test, written to ``smoke_<limit>``).
    """
    from sklearn.decomposition import PCA, IncrementalPCA

    runs = discover_parti_runs()
    prompt_table = pd.read_csv(PROMPT_TABLE, sep="\t")
    prompt_table["slug"] = prompt_table["Prompt"].map(slugify_prompt)
    labels = prompt_table.set_index("slug")[["Category", "Challenge"]]
    items = [(p, [(r.seed, str(r.zip)) for r in g.itertuples()]) for p, g in runs.groupby("prompt", sort=True)]
    if limit:
        items = items[:limit]
    print(f"[INFO] {len(items)} prompts, {len(runs)} runs", flush=True)

    metas, noise_sub, sub_masks, blips = [], [], [], []
    evr = {"separate": {}, "challenge": {}, "all": {}}
    with ProcessPoolExecutor(workers) as executor:
        for i, result in enumerate(executor.map(_process_prompt, items)):
            prompt = result["meta"]["prompt"].iat[0]
            metas.append(result["meta"])
            noise_sub.append(result["noise_sub"])
            sub_masks.append(result["sub"])
            blips.append(result["blip"])
            evr["separate"][prompt] = {"noise": result["evr_noise"], "blip2": result["evr_blip2"]}
            print(f"[INFO] {i + 1}/{len(items)} {prompt[:60]}", flush=True)

    meta = pd.concat(metas, ignore_index=True)
    meta = meta.join(labels, on=meta["prompt"].map(slugify_prompt))
    offsets = np.cumsum([0] + [len(m) for m in metas])
    prompt_index = {m["prompt"].iat[0]: k for k, m in enumerate(metas)}
    for column in [f"{s}_pc{c}_{sp}" for s in ("ch", "all") for c in (1, 2) for sp in SPACES]:
        meta[column] = np.nan

    def rows(k, joint_only):
        """Row indices of prompt ``k`` in ``meta`` (only the joint generations if ``joint_only``)."""
        r = np.arange(offsets[k], offsets[k + 1])
        return r[sub_masks[k]] if joint_only else r

    def joint(keys, scope, name):
        """Fit a joint PCA of the prompts ``keys`` per space and store PC1/PC2 under ``<scope>_pc*``."""
        noise_X = np.concatenate([noise_sub[k] for k in keys]).astype(np.float32)
        blip_X = np.concatenate([blips[k] for k in keys]).astype(np.float32)
        result = {}
        for space, X, joint_only in (("noise", noise_X, True), ("blip2", blip_X, False)):
            pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=0)
            projection = pca.fit_transform(X)
            r = np.concatenate([rows(k, joint_only) for k in keys])
            meta.loc[r, f"{scope}_pc1_{space}"] = projection[:, 0]
            meta.loc[r, f"{scope}_pc2_{space}"] = projection[:, 1]
            result[space] = pca.explained_variance_ratio_.tolist()
        evr["challenge" if scope == "ch" else "all"][name] = result

    for challenge in CHALLENGE_ORDER:
        keys = [prompt_index[p] for p in sorted(meta.loc[meta["Challenge"] == challenge, "prompt"].unique())]
        if not keys:
            continue
        joint(keys, "ch", challenge)
        print(f"[INFO] challenge {challenge}: {len(keys)} prompts", flush=True)

    # All prompts: noise incrementally (143k x 65536 values would be about 37 GB in float32), BLIP-2 directly
    keys = list(range(len(metas)))
    ipca = IncrementalPCA(n_components=N_COMPONENTS)
    for start in range(0, len(keys), 6):           # blocks of 6 prompts (about 7.5k rows)
        ipca.partial_fit(np.concatenate([noise_sub[k] for k in keys[start:start + 6]]).astype(np.float32))
    for k in keys:
        projection = ipca.transform(noise_sub[k].astype(np.float32))
        r = rows(k, True)
        meta.loc[r, "all_pc1_noise"], meta.loc[r, "all_pc2_noise"] = projection[:, 0], projection[:, 1]
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=0)
    projection = pca.fit_transform(np.concatenate(blips).astype(np.float32))
    meta["all_pc1_blip2"], meta["all_pc2_blip2"] = projection[:, 0], projection[:, 1]
    evr["all"]["all"] = {"noise": ipca.explained_variance_ratio_.tolist(),
                         "blip2": pca.explained_variance_ratio_.tolist()}

    out = PCA_DATA if not limit else PCA_DATA / f"smoke_{limit}"
    out.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(out / "projections.parquet", index=False)
    (out / "explained_variance.json").write_text(json.dumps(evr, indent=1))
    print(f"[INFO] saved: {out}", flush=True)


# ------------------------------------------------------------------ plot
def _slug(text: str) -> str:
    """Lower-case file name component of a challenge name."""
    return re.sub(r"[^\w]+", "_", text).strip("_").lower()


def _ramp(color: str) -> LinearSegmentedColormap:
    """Colour map from light (generation 0) to the prompt colour (generation 30)."""
    c = np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list("ramp", [0.85 * np.ones(3) + 0.15 * c, c])


def _style(ax: plt.Axes) -> None:
    """Remove ticks and grid (PCA axes carry no units)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _evr_label(evr: dict, space: str) -> tuple[str, str]:
    """Axis labels with the explained variance of PC1 and PC2."""
    v = evr[space]
    return f"PC1 ({v[0]:.0%})", f"PC2 ({v[1]:.0%})"


def _outline(lw: float) -> list:
    """Dark outline path effect, so that light colours stay visible."""
    return [patheffects.withStroke(linewidth=lw + 1.0, foreground="#252525", alpha=0.7)]


def _arrow(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, color: str, lw: float = 1.2, start_marker: bool = True,
           zorder: int = 4) -> None:
    """Draw a trajectory with an arrow head in the exact prompt or challenge colour with a dark outline.

    Args:
        ax: Target axes.
        xs: x coordinates in generation order.
        ys: y coordinates in generation order.
        color: Line colour.
        lw: Line width.
        start_marker: Mark the start with a white dot.
        zorder: Drawing order.
    """
    if len(xs) < 2:
        return
    ax.plot(xs, ys, color=color, lw=lw, zorder=zorder, solid_capstyle="round", path_effects=_outline(lw))
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), zorder=zorder + 1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7, shrinkA=0, shrinkB=0,
                                path_effects=_outline(lw)))
    if start_marker:
        ax.scatter([xs[0]], [ys[0]], s=10, facecolor="white", edgecolor="#252525", linewidth=0.6,
                   zorder=zorder + 1)


def _centroids(df: pd.DataFrame, x: str, y: str, by=("prompt",), gens=JOINT_GENS) -> pd.DataFrame:
    """Mean projection per group and generation."""
    d = df[df["generation"].isin(gens)]
    return d.groupby(list(by) + ["generation"])[[x, y]].mean().reset_index()


def plot_separate(meta: pd.DataFrame, evr: dict, space: str) -> None:
    """One figure per challenge with one panel per prompt (its own PCA), colour = generation.

    Args:
        meta: Projections.
        evr: Explained variance ratios.
        space: ``"noise"`` or ``"blip2"``.
    """
    x, y = f"sep_pc1_{space}", f"sep_pc2_{space}"
    for challenge in CHALLENGE_ORDER:
        prompts = sorted(meta.loc[meta["Challenge"] == challenge, "prompt"].unique(), key=str.lower)
        if not prompts:
            continue
        ncols = 4
        nrows = int(np.ceil(len(prompts) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_WIDTH_IN, 1.45 * nrows + 0.5), squeeze=False,
                                 layout="constrained")
        for ax, prompt in zip(axes.flat, prompts):
            d = meta[meta["prompt"] == prompt].sort_values("generation")
            points = ax.scatter(d[x], d[y], c=d["generation"], cmap=GEN_CMAP[space], vmin=0, vmax=30, s=1.5,
                                alpha=0.6, linewidths=0, rasterized=True)
            _style(ax)
            title = prompt if len(prompt) <= 34 else prompt[:33].rstrip() + "…"
            v = evr["separate"][prompt][space]
            ax.set_title(f"{title}\n({v[0]:.0%} / {v[1]:.0%})", fontsize=TICK_SIZE - 2, pad=3)
        for ax in axes.flat[len(prompts):]:
            fig.delaxes(ax)
        colorbar = fig.colorbar(points, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
        colorbar.set_label("Generation", fontsize=TICK_SIZE)
        colorbar.ax.tick_params(labelsize=TICK_SIZE - 1, length=2)
        fig.suptitle(f"{challenge}: {SPACES[space]}, PCA per prompt (PC1 vs. PC2, explained variance)",
                     fontsize=TITLE_SIZE, fontweight="bold")
        save_figure(fig, OUT_DIR / "separate" / f"pca_separate_{space}_{_slug(challenge)}")


def _legend_prompts(ax: plt.Axes, prompts: list[str], colors: dict[str, str], max_chars: int = 38) -> None:
    """Prompt legend right of the axes (long prompts truncated)."""
    handles = [Line2D([], [], color=colors[p], lw=1.5, path_effects=_outline(1.5),
                      label=p if len(p) <= max_chars else p[:max_chars - 1].rstrip() + "…") for p in prompts]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=TICK_SIZE - 2.5,
              frameon=False, handlelength=1.2, borderaxespad=0)


def plot_challenge(meta: pd.DataFrame, evr: dict, space: str) -> None:
    """One figure per challenge: joint PCA, candidates light -> dark by generation, centroid arrows per prompt.

    In noise space the centroids are computed per seed, because every seed forms its own arm.

    Args:
        meta: Projections.
        evr: Explained variance ratios.
        space: ``"noise"`` or ``"blip2"``.
    """
    x, y = f"ch_pc1_{space}", f"ch_pc2_{space}"
    colors = prompt_colors(meta, "Challenge", CHALLENGE_ORDER)
    for challenge in CHALLENGE_ORDER:
        d = meta[(meta["Challenge"] == challenge) & meta[x].notna()]
        prompts = sorted(d["prompt"].unique(), key=str.lower)
        if not prompts:
            continue
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.4), layout="constrained")
        for prompt in prompts:
            dp = d[d["prompt"] == prompt].sort_values("generation")
            ax.scatter(dp[x], dp[y], c=dp["generation"], cmap=_ramp(colors[prompt]), vmin=0, vmax=30, s=1.2,
                       alpha=0.35, linewidths=0, rasterized=True, zorder=1)
        by = ("prompt", "seed") if space == "noise" else ("prompt",)
        centroids = _centroids(d, x, y, by)
        for key, c in centroids.groupby(list(by)):
            prompt = key[0] if isinstance(key, tuple) else key
            c = c.sort_values("generation")
            _arrow(ax, c[x].to_numpy(), c[y].to_numpy(), colors[prompt], lw=1.1)
        _style(ax)
        xlabel, ylabel = _evr_label(evr["challenge"][challenge], space)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(f"{challenge} ({len(prompts)} prompts): {SPACES[space]}, joint PCA", fontsize=TITLE_SIZE,
                     fontweight="bold")
        _legend_prompts(ax, prompts, colors)
        ax.text(0.0, -0.08, "points: light → dark = generation 0 → 30 · arrows: centroid per generation "
                f"({', '.join(map(str, JOINT_GENS))})" + (", per seed" if space == "noise" else ""),
                transform=ax.transAxes, fontsize=TICK_SIZE - 2.5, color="#525252", va="top")
        save_figure(fig, OUT_DIR / "challenge" / f"pca_challenge_{space}_{_slug(challenge)}")


def plot_all(meta: pd.DataFrame, evr: dict, space: str) -> None:
    """Joint PCA of all prompts: centroid trajectory per prompt (left) and per challenge (right).

    Args:
        meta: Projections.
        evr: Explained variance ratios.
        space: ``"noise"`` or ``"blip2"``.
    """
    x, y = f"all_pc1_{space}", f"all_pc2_{space}"
    d = meta[meta[x].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 3.0), sharex=True, sharey=True, layout="constrained")
    background = d[d["generation"].isin(JOINT_GENS)]
    background = background.sample(n=min(40000, len(background)), random_state=0)
    for ax in axes:
        ax.scatter(background[x], background[y], s=0.5, color="#D9D9D9", alpha=0.4, linewidths=0, rasterized=True,
                   zorder=0)
    per_seed = space == "noise"
    by = ["Challenge", "prompt"] + (["seed"] if per_seed else [])
    centroids = _centroids(d, x, y, by)
    for key, c in centroids.groupby(by):
        c = c.sort_values("generation")
        _arrow(axes[0], c[x].to_numpy(), c[y].to_numpy(), CHALLENGE_COLORS[key[0]], lw=0.5, start_marker=False)
    by_challenge = ["Challenge"] + (["seed"] if per_seed else [])
    challenge_centroids = centroids.groupby(by_challenge + ["generation"])[[x, y]].mean().reset_index()
    for key, c in challenge_centroids.groupby(by_challenge):
        challenge = key[0] if isinstance(key, tuple) else key
        c = c.sort_values("generation")
        _arrow(axes[1], c[x].to_numpy(), c[y].to_numpy(), CHALLENGE_COLORS[challenge], lw=1.4)
    xlabel, ylabel = _evr_label(evr["all"]["all"], space)
    for ax, title in zip(axes, ("per prompt" + (" × seed" if per_seed else ""),
                                "challenge mean" + (" × seed" if per_seed else ""))):
        _style(ax)
        ax.set_title(title, fontsize=TITLE_SIZE - 1, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    axes[0].set_ylabel(ylabel, fontsize=LABEL_SIZE)
    handles = [Line2D([], [], color=c, lw=1.6, label=challenge) for challenge, c in CHALLENGE_COLORS.items()]
    fig.legend(handles=handles, loc="outside right center", fontsize=TICK_SIZE - 2, frameon=False, handlelength=1.4)
    fig.suptitle(f"All {d['prompt'].nunique()} prompts: {SPACES[space]}, joint PCA", fontsize=TITLE_SIZE,
                 fontweight="bold")
    fig.supxlabel("arrows: centroid trajectory, generation 0 → 30 (every 5th) · grey: candidates",
                  fontsize=TICK_SIZE - 2, color="#525252")
    save_figure(fig, OUT_DIR / f"pca_all_{space}")


def plot() -> None:
    """Draw all PCA figures from the stored projections."""
    meta = pd.read_parquet(PCA_DATA / "projections.parquet")
    evr = json.loads((PCA_DATA / "explained_variance.json").read_text())
    for space in SPACES:
        plot_all(meta, evr, space)
        plot_challenge(meta, evr, space)
        plot_separate(meta, evr, space)
    plt.close("all")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["compute", "plot"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, help="only the first N prompts (smoke test)")
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    if args.step == "compute":
        compute(args.workers, args.limit)
    else:
        set_preview_dir(args.preview)
        plot()


if __name__ == "__main__":
    main()
