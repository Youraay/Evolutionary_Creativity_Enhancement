"""Render every candidate of an initial population mutated with every combination of the grid.

    python -m experiments.mutation_grid_search.worker [--prompt "a cat"] [--seed 1943746817] [--candidates 5]
        [--fractions 0.1 0.2 ...] [--strengths 0.2 0.5 ...]
"""
import argparse
import logging
from pathlib import Path

from experiments.common.ga import configure_logging, log_environment, set_all_seeds
from experiments.common.naming import slugify_prompt
from experiments.common.settings import DEVICE, HF_CACHE
from experiments.mutation_grid_search.config import (CANDIDATES, FRACTIONS, INFERENCE_STEPS, PROMPT, RESULT_DIR, SEED,
                                                     STRENGTHS)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    parser.add_argument("--strengths", type=float, nargs="+", default=STRENGTHS)
    parser.add_argument("--out", type=Path, default=RESULT_DIR)
    return parser.parse_args()


def run(prompt: str, seed: int, candidates: int, fractions: list[float], strengths: list[float], out: Path) -> Path:
    """Render the original and all mutated images of every candidate and record their BLIP-2 distances.

    Every mutation starts from the unmutated noise of the candidate; the random generator is seeded per
    candidate and grid point, so every image can be reproduced on its own.

    Args:
        prompt: Text prompt.
        seed: Seed of the initial population.
        candidates: Number of candidates of the initial population.
        fractions: Mutated fractions ``p_m``.
        strengths: Standard deviations ``sigma_m``.
        out: Directory that receives the run directory.

    Returns:
        The run directory.
    """
    import pandas as pd
    import torch
    from torch.nn import functional as F

    from evolutionary_creativity.models.embeddings import Blip2EmbeddingModel
    from evolutionary_creativity.models.text_to_image.sdxl import StableDiffusionXLModel
    from evolutionary_creativity.noise_factory import NoiseFactory
    from evolutionary_creativity.operators import UniformGaussianMutator

    name = f"{slugify_prompt(prompt)}_{seed}"
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)
    sdxl = StableDiffusionXLModel(DEVICE, HF_CACHE, num_inference_steps=INFERENCE_STEPS, torch_dtype=torch.float16)
    blip2 = Blip2EmbeddingModel(DEVICE, HF_CACHE)
    population = NoiseFactory().create_population(sdxl.noise_config(), candidates)
    grid = [(p, s) for p in fractions for s in strengths]
    rows = []
    for k, candidate in enumerate(population, start=1):
        noises = [candidate.initial_noise]
        for i, (fraction, strength) in enumerate(grid):
            torch.manual_seed(seed + 1000 * k + i)
            noises.append(UniformGaussianMutator(fraction, strength).mutate(candidate.initial_noise))
        images = []
        for start in range(0, len(noises), 5):
            images += sdxl.generate_batch(noises[start:start + 5], prompt)
        images[0].save(directory / f"original_c{k}.JPEG")
        embeddings = blip2.image_features(images).float()
        distances = ((1 - F.cosine_similarity(embeddings[1:], embeddings[:1])) / 2).tolist()
        for (fraction, strength), image, distance in zip(grid, images[1:], distances):
            image.save(directory / f"mutated_c{k}_p{fraction:g}_s{strength:g}.JPEG")
            rows.append({"candidate": k, "fraction": fraction, "strength": strength, "blip2_distance": distance})
        logger.info("Candidate %d done", k)
    pd.DataFrame(rows).to_csv(directory / f"{name}.csv", index=False)
    return directory


if __name__ == "__main__":
    configure_logging()
    log_environment()
    arguments = parse_args()
    logger.info("Results: %s", run(arguments.prompt, arguments.seed, arguments.candidates, arguments.fractions,
                                   arguments.strengths, arguments.out))
