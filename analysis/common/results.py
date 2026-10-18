"""Loading of GA result directories and the random CLIP-score baseline."""
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analysis.common.paths import HF_CACHE, PROMPT_TABLE, archive, tables
from experiments.common.naming import slugify_prompt

RUN_PATTERN = re.compile(r"^(?P<slug>.+)_(?P<seed>\d{6,12})_(?P<ts>\d{8}_\d{6})$")
"""Name of a run directory: ``<prompt_slug>_<seed>_<YYYYmmdd_HHMMSS>``."""

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_PREFIX = "A photo depicts "
"""Prefix of the CLIP score, as in ``CLIPScoreEvaluator``."""
CLIP_SCALE = 2.5


def discover_runs(experiment: str, group_col: str) -> pd.DataFrame:
    """Find all runs ``<group>/<prompt>/<run>/<run>.csv`` of an experiment; the newest run per group, prompt and
    seed wins.

    Directories whose CSV is not named like the run (e.g. ``*_ABORTED``) are skipped. The expected number of
    CSV rows is ``(generations + 1) * population`` from ``<run>_experiment.json`` or the pipeline config.

    Args:
        experiment: Package name below ``experiments`` (tables from :func:`analysis.common.paths.tables`).
        group_col: Name of the column holding the first directory level (e.g. ``"variant"`` or ``"model"``).

    Returns:
        One row per run with ``group_col``, prompt, seed, timestamp, csv, zip and expected row count.
    """
    rows = []
    for csv in sorted(tables(experiment).glob("*/*/*/*.csv")):
        run = csv.parent
        match = RUN_PATTERN.match(run.name)
        if not match or csv.stem != run.name:
            continue
        config = json.loads((run / f"{run.name}.json").read_text())
        experiment_info = run / f"{run.name}_experiment.json"
        meta = json.loads(experiment_info.read_text()) if experiment_info.exists() else config
        rows.append({group_col: run.parent.parent.name, "prompt": config["prompt"], "seed": match.group("seed"),
                     "timestamp": match.group("ts"), "csv": csv, "zip": archive(run, experiment),
                     "expected": (meta["num_generations"] + 1) * meta["population_size"]})
    runs = pd.DataFrame(rows).sort_values("timestamp")
    return runs.drop_duplicates([group_col, "prompt", "seed"], keep="last").reset_index(drop=True)


def load_runs(runs: pd.DataFrame, group_col: str, rename: dict[str, str]) -> pd.DataFrame:
    """Read the candidate tables of complete runs (incomplete runs are skipped with a warning).

    Args:
        runs: Output of :func:`discover_runs`.
        group_col: Group column of ``runs``.
        rename: Column renames, e.g. ``{"model_MMD_blip2": "divergence"}``.

    Returns:
        One row per candidate and generation with generation, file name, fitness, all ``model_*`` metrics,
        the group, prompt and seed.
    """
    frames = []
    for run in runs.itertuples():
        df = pd.read_csv(run.csv)
        if len(df) < run.expected:
            print(f"[WARN] incomplete, skipped ({len(df)}/{run.expected}): {run.csv.name}")
            continue
        columns = ["generation", "file_name", "fitness"] + [c for c in df.columns if c.startswith("model_")]
        df = df[columns].rename(columns=rename)
        frames.append(df.assign(**{group_col: getattr(run, group_col), "prompt": run.prompt, "seed": run.seed}))
    return pd.concat(frames, ignore_index=True)


def load_embedding_experiment(experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an experiment with metrics in four embedding spaces (``long_runs``, ``tournament_size`` or
    ``fitness_embedding``).

    Divergence and diversity are the BLIP-2 metrics (``model_MMD_blip2``, ``model_Diversity_blip2``).

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        ``(candidates, runs)``.
    """
    runs = discover_runs(experiment, "variant")
    data = load_runs(runs, "variant", {"model_MMD_blip2": "divergence", "model_Diversity_blip2": "diversity"})
    print(f"[INFO] {experiment}: {data.groupby(['variant', 'prompt', 'seed']).ngroups} runs, "
          f"variants {sorted(data['variant'].unique())}")
    return data, runs


def load_model_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all complete runs of the model comparison.

    Returns:
        ``(candidates, runs)`` with the group column ``model``.
    """
    runs = discover_runs("model_comparison", "model")
    data = load_runs(runs, "model", {"model_LocalMaxMeanDivergence": "divergence",
                                     "model_DiversityWithinPopulation": "diversity"})
    print(f"[INFO] model comparison: {data['model'].nunique()} models, "
          f"{data.groupby(['model', 'prompt', 'seed']).ngroups} runs")
    return data, runs


def discover_parti_runs() -> pd.DataFrame:
    """Find the runs of the PartiPrompts experiment (``<prompt_slug>/<run>/<run>.json``).

    Returns:
        One row per prompt and seed (the newest run wins) with slug, prompt, seed, timestamp, csv and zip.
    """
    rows = []
    for config_path in sorted(tables("parti_prompts").glob("*/*/*.json")):
        run = config_path.parent
        config = json.loads(config_path.read_text())
        rows.append({"prompt_slug": run.parent.name, "prompt": config["prompt"], "seed": config["random_seed"],
                     "timestamp": config["timestamp"], "csv": run / f"{run.name}.csv",
                     "zip": archive(run, "parti_prompts")})
    runs = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates(["prompt_slug", "seed"], keep="last")
    return runs.reset_index(drop=True)


def attach_prompt_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Category`` and ``Challenge`` of the PartiPrompts selection (matched by prompt slug).

    Args:
        df: Table with a ``prompt`` column.

    Returns:
        ``df`` with the two columns (replaced if present).

    Raises:
        ValueError: If a prompt is missing in the prompt table.
    """
    meta = pd.read_csv(PROMPT_TABLE, sep="\t")
    meta = meta.assign(slug=meta["Prompt"].map(slugify_prompt)).set_index("slug")[["Category", "Challenge"]]
    out = df.drop(columns=meta.columns, errors="ignore").join(meta, on=df["prompt"].map(slugify_prompt))
    if out[meta.columns].isna().any().any():
        raise ValueError(f"Prompts without metadata: {out.loc[out['Category'].isna(), 'prompt'].unique()}")
    return out


def load_images(zip_path: Path, file_names: list[str], size: int = 384) -> list[np.ndarray]:
    """Read images of a run archive and downscale them.

    Args:
        zip_path: Result archive of a run.
        file_names: File stems (``g<gen>_id<id>_f<fitness>``).
        size: Maximum edge length in pixels.

    Returns:
        RGB arrays in the order of ``file_names``.
    """
    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(n).stem: n for n in archive.namelist() if "images/" in n}
        images = []
        for name in file_names:
            image = Image.open(io.BytesIO(archive.read(members[name]))).convert("RGB")
            image.thumbnail((size, size), Image.LANCZOS)
            images.append(np.asarray(image))
    return images


def random_candidates(data: pd.DataFrame, generation: int, n: int, rng: np.random.Generator) -> list[str]:
    """Draw ``n`` candidates of a generation at random (no selection by score).

    Args:
        data: Candidates of one run.
        generation: Generation to draw from.
        n: Number of candidates.
        rng: Random generator.

    Returns:
        File stems of the drawn candidates.
    """
    names = sorted(data.loc[data["generation"] == generation, "file_name"])
    return list(rng.choice(names, n, replace=False))


def clip_text_features(prompts: list[str]):
    """Normalised CLIP text embeddings of the prompts (with the CLIP-score prefix).

    Args:
        prompts: Prompts to embed.

    Returns:
        Tensor of shape ``[len(prompts), 512]`` with unit rows.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL, cache_dir=HF_CACHE, local_files_only=True).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, cache_dir=HF_CACHE, local_files_only=True)
    features = []
    with torch.inference_mode():
        for prompt in prompts:
            text = prompt if prompt.startswith(CLIP_PREFIX) else CLIP_PREFIX + prompt
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            output = model.get_text_features(**inputs)
            if not isinstance(output, torch.Tensor):
                output = getattr(output, "text_embeds", getattr(output, "pooler_output", output[0]))
            features.append(output[0])
    features = torch.stack(features)
    return features / features.norm(dim=-1, keepdim=True)


def random_clip_baseline(runs: pd.DataFrame, candidates: pd.DataFrame, pool_per_run: int, samples: int = 100,
                         seed: int = 0) -> pd.DataFrame:
    """CLIP score of every prompt against random images of the other prompts.

    A pool of ``pool_per_run`` random images is drawn from every run (all generations); every prompt is
    then scored against ``samples`` pool images of other prompts. The image embeddings are read from the
    result archives (``/clip/*.pt``); only the text embeddings are computed. As a check, the pool images
    are also scored against their own prompt, which must reproduce the CLIP score of the CSV.

    Args:
        runs: Runs with the columns prompt, seed and zip.
        candidates: Candidates with prompt, seed, file_name and ``model_CLIPScore`` (for the check).
        pool_per_run: Number of pool images per run.
        samples: Number of baseline images per prompt.
        seed: Seed of the random generator.

    Returns:
        One row per prompt and baseline image with ``baseline_clip_score``.
    """
    import torch

    rng = np.random.default_rng(seed)
    pool, embeddings = [], []
    for run in runs.itertuples():
        with zipfile.ZipFile(run.zip) as archive:
            names = sorted(n for n in archive.namelist() if n.startswith("/clip/") and n.endswith(".pt"))
            for name in rng.choice(names, pool_per_run, replace=False):
                embeddings.append(torch.load(io.BytesIO(archive.read(name)), map_location="cpu").reshape(-1).float())
                pool.append({"prompt": run.prompt, "seed": run.seed, "file_name": Path(name).stem})
    pool = pd.DataFrame(pool)
    embeddings = torch.stack(embeddings)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    prompts = sorted(runs["prompt"].unique())
    scores = (CLIP_SCALE * torch.clamp(clip_text_features(prompts) @ embeddings.T, min=0.0)).numpy()

    index = {p: i for i, p in enumerate(prompts)}
    own = scores[pool["prompt"].map(index).to_numpy(), np.arange(len(pool))]
    reference = pool.merge(candidates[["prompt", "seed", "file_name", "model_CLIPScore"]].drop_duplicates(
        ["prompt", "seed", "file_name"]), on=["prompt", "seed", "file_name"], how="left")["model_CLIPScore"]
    print(f"[INFO] check of the CLIP score against the CSV: max |difference| = "
          f"{np.nanmax(np.abs(own - reference.to_numpy())):.2e} ({reference.notna().sum()} images)")

    rows = []
    for prompt in prompts:
        others = np.flatnonzero(pool["prompt"].to_numpy() != prompt)
        for j in rng.choice(others, samples, replace=False):
            rows.append({"prompt": prompt, "source_prompt": pool.at[j, "prompt"], "source_seed": pool.at[j, "seed"],
                         "file_name": pool.at[j, "file_name"], "baseline_clip_score": float(scores[index[prompt], j])})
    return pd.DataFrame(rows)
