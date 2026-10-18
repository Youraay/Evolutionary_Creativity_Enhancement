"""CLIP scores of finished runs whose archives contain only the images (experiments of ``experiments.common.variants``).

The score is that of ``CLIPScoreEvaluator``: ``2.5 * max(cos(image, "A photo depicts <prompt>"), 0)`` with CLIP
ViT-B/32. Two modes write CSV files into ``<result dir>/clip_scores`` (or ``--out``):

    prompts   every image of the runs of the example prompts against their own prompt
              (``scores_<prompt>_seed_<label>.csv``) and, as random baseline, 100 images drawn from the runs of
              the other prompts (all generations) against every prompt (``baseline_scores_<prompt>.csv``)
    runs      every image of every run of a comparison against "a cat" (``scores_<run>.csv``)

    python -m experiments.common.clip_scores prompts example_prompts
    python -m experiments.common.clip_scores runs archive_size crossover_function fitness_function
"""
import argparse
import io
import random
import re
import zipfile
from pathlib import Path

import pandas as pd

from experiments.common.settings import DEVICE, HF_CACHE, result_dir

PROMPTS = ["a cat", "a dog", "a chair", "a coffee mug", "boredom", "creativity"]
SEED_LABELS = ["1", "2", "3"]
COMPARISON_PROMPT = "a cat"
BASELINE_SIZE = 100
BATCH_SIZE = 32


def archive_images(zip_path: Path) -> list[str]:
    """Image members of a run archive in archive order.

    Args:
        zip_path: Run archive.

    Returns:
        Member names below ``/images/``.
    """
    with zipfile.ZipFile(zip_path) as archive:
        return [n for n in archive.namelist() if "images/" in n and n.lower().endswith((".png", ".jpg", ".jpeg"))]


def score_images(evaluator, items: list[tuple[Path, str]], prompt: str) -> list[float]:
    """CLIP scores of images from run archives against a prompt.

    Args:
        evaluator: ``CLIPScoreEvaluator``.
        items: ``(archive, member)`` per image.
        prompt: Prompt without the CLIPScore prefix.

    Returns:
        One score per item.
    """
    from PIL import Image

    scores = []
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        images = []
        for zip_path, member in batch:
            with zipfile.ZipFile(zip_path) as archive:
                images.append(Image.open(io.BytesIO(archive.read(member))).convert("RGB"))
        scores += [result["score"] for result in evaluator.evaluate(images, prompt)]
    return scores


def find_run(directory: Path, prompt: str, label: str) -> Path | None:
    """Archive of the run ``<prompt>_Seed_<label>_*`` (case-insensitive) in ``directory``.

    Args:
        directory: Result directory of the experiment.
        prompt: Prompt of the run.
        label: Seed label.

    Returns:
        Path of the archive, or ``None``.
    """
    pattern = re.compile(rf"^{re.escape(prompt.replace(' ', '_'))}_Seed_{label}(_|$)", re.IGNORECASE)
    runs = sorted(p for p in directory.iterdir()
                  if p.is_dir() and pattern.match(p.name.replace("CoffeeMug", "Coffee_Mug")))
    archives = [z for run in runs for z in sorted(run.glob("*.zip"))]
    return archives[0] if archives else None


def score_prompts(directory: Path, out: Path, evaluator, seed: int) -> None:
    """Scores of the example prompts against their own prompt and the random baseline of every prompt.

    Args:
        directory: Result directory of ``example_prompts``.
        out: Output directory.
        evaluator: ``CLIPScoreEvaluator``.
        seed: Seed of the random baseline sample.
    """
    catalog: dict[str, list[tuple[Path, str]]] = {prompt: [] for prompt in PROMPTS}
    for prompt in PROMPTS:
        for label in SEED_LABELS:
            zip_path = find_run(directory, prompt, label)
            if zip_path is None:
                print(f"[WARN] no run of {prompt!r} seed {label}")
                continue
            items = [(zip_path, member) for member in archive_images(zip_path)]
            catalog[prompt] += items
            frame = pd.DataFrame({"image_path": [m for _, m in items],
                                  "clip_score": score_images(evaluator, items, prompt)})
            frame.to_csv(out / f"scores_{prompt.replace(' ', '_')}_seed_{label}.csv", index=False)
            print(f"[INFO] {prompt!r} seed {label}: {len(frame)} images", flush=True)
    rng = random.Random(seed)
    for prompt in PROMPTS:
        others = [item for other, items in catalog.items() if other != prompt for item in items]
        sample = rng.sample(others, min(BASELINE_SIZE, len(others)))
        frame = pd.DataFrame({"source_zip": [z.name for z, _ in sample], "image_path": [m for _, m in sample],
                              "baseline_clip_score": score_images(evaluator, sample, prompt)})
        frame.to_csv(out / f"baseline_scores_{prompt.replace(' ', '_')}.csv", index=False)
        print(f"[INFO] baseline {prompt!r}: {len(frame)} images", flush=True)


def score_runs(directory: Path, out: Path, evaluator, prompt: str = COMPARISON_PROMPT) -> None:
    """Scores of every image of every run in ``directory`` against ``prompt``.

    Args:
        directory: Result directory of a comparison.
        out: Output directory.
        evaluator: ``CLIPScoreEvaluator``.
        prompt: Prompt of the comparison.
    """
    for zip_path in sorted(directory.glob("*/*.zip")):
        items = [(zip_path, member) for member in archive_images(zip_path)]
        frame = pd.DataFrame({"image_path": [m for _, m in items],
                              "clip_score": score_images(evaluator, items, prompt)})
        frame.to_csv(out / f"scores_{zip_path.parent.name}.csv", index=False)
        print(f"[INFO] {zip_path.parent.name}: {len(frame)} images", flush=True)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["prompts", "runs"])
    parser.add_argument("experiments", nargs="+", help="package names below experiments/")
    parser.add_argument("--out", type=Path, help="output directory (default: <result dir>/clip_scores)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random baseline sample")
    args = parser.parse_args()

    from evolutionary_creativity.evaluators import CLIPScoreEvaluator
    from evolutionary_creativity.models.embeddings import ClipEmbeddingModel

    evaluator = CLIPScoreEvaluator(ClipEmbeddingModel(DEVICE, HF_CACHE), DEVICE)
    for experiment in args.experiments:
        directory = result_dir(experiment)
        out = args.out or directory / "clip_scores"
        out.mkdir(parents=True, exist_ok=True)
        if args.mode == "prompts":
            score_prompts(directory, out, evaluator, args.seed)
        else:
            score_runs(directory, out, evaluator)


if __name__ == "__main__":
    main()
