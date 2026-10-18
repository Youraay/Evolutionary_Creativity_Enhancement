"""Seed optimisation with a prompt rewritten by Think-Then-Generate.

The prompt is rewritten beforehand (``data/prompts/t2g_rewritten_prompts.json``, created with
``rewrite_prompts.py``). The GA then runs with exactly the setup of the PartiPrompts experiment (SDXL with
30 steps, population 60, tournament 3, uniform crossover, Gaussian mutation, fitness = novelty in BLIP-2
space), only with more generations. Variant A (``--generator t2g``) uses the T2G DiT with 30 steps, true
CFG 4 and batch size 4 instead of SDXL.

Recorded metrics:
    model_LocalMaxMeanDivergence / model_DiversityWithinPopulation  as before (BLIP-2)
    model_CLIPScore         CLIP score against the ORIGINAL prompt (fidelity to the user's intent, same
                            meaning as in the PartiPrompts runs)
    model_CLIPScoreRevised  CLIP score against the rewritten prompt (the generation prompt)

    python -m experiments.think_then_generate.worker --prompt "a cat" --seed 1943746817 --generations 100
    python -m experiments.think_then_generate.worker --prompt "a cat" --seed 1943746817 --condition original
    python -m experiments.think_then_generate.worker --prompt "a cat" --seed 1943746817 --generator t2g --generations 30
"""
import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from evolutionary_creativity.evaluators import (BaseImageEvaluator, CLIPScoreEvaluator,
                                                DiversityWithinPopulationEvaluator, EvaluationResult,
                                                MaxMeanDivergenceEvaluator)
from evolutionary_creativity.models.embeddings import Blip2EmbeddingModel, ClipEmbeddingModel
from evolutionary_creativity.models.text_to_image.sdxl import StableDiffusionXLModel
from evolutionary_creativity.models.text_to_image.think_then_generate import ThinkThenGenerate
from experiments.common.ga import (build_pipeline, configure_logging, log_environment, new_run_id, set_all_seeds,
                                   success_marker)
from experiments.common.settings import DEVICE, HF_CACHE
from experiments.think_then_generate.config import GENERATORS, RESULT_DIR, REWRITES

logger = logging.getLogger(__name__)


class FixedPromptEvaluator(BaseImageEvaluator):
    """Evaluates against a fixed prompt, regardless of the prompt the pipeline passes."""

    def __init__(self, evaluator: BaseImageEvaluator, prompt: str, name: str) -> None:
        """
        Args:
            evaluator: Wrapped evaluator; it is renamed to ``name``.
            prompt: Prompt used for every evaluation.
            name: Metric name.
        """
        self.evaluator = evaluator
        self.prompt = prompt
        self.name = name
        evaluator.name = name

    def evaluate(self, images: list[Image.Image], prompt: str, update_reference: bool = False) -> list[EvaluationResult]:
        """Evaluate ``images`` against the fixed prompt (``prompt`` is ignored)."""
        return self.evaluator.evaluate(images, self.prompt, update_reference=update_reference)


def load_rewrites() -> dict:
    """Read the rewritten prompts (see ``config.REWRITES``)."""
    return json.loads(REWRITES.read_text())


def run(prompt: str, seed: int, generations: int, population: int, condition: str, generator: str = "sdxl",
        steps: int | None = None) -> str:
    """Run the GA with the rewritten or the original prompt.

    Besides the standard result files, ``<run>_t2g.json`` records condition, generator, both prompts and
    the reasoning of the rewrite.

    Args:
        prompt: Original prompt (key in the rewrite file).
        seed: Seed of the run.
        generations: Number of evolution steps.
        population: Population size.
        condition: ``"t2g"`` (generate with the rewritten prompt) or ``"original"``.
        generator: ``"sdxl"`` or ``"t2g"`` (T2G DiT).
        steps: Number of denoising steps (default 30).

    Returns:
        Result directory of the run (or of the prompt, if the run had already finished).

    Raises:
        KeyError: If ``condition`` is ``"t2g"`` and the prompt has no rewrite.
    """
    rewrites = load_rewrites()
    if condition == "t2g":
        if prompt not in rewrites["prompts"]:
            raise KeyError(f"No rewritten prompt for {prompt!r} in {REWRITES}")
        generation_prompt = rewrites["prompts"][prompt]["revised_prompt"]
    else:
        generation_prompt = prompt

    slug, run_id = new_run_id(prompt, seed)
    directory = RESULT_DIR / f"{GENERATORS[generator]}_{condition}_g{generations}" / slug
    directory.mkdir(parents=True, exist_ok=True)
    marker = success_marker(directory, seed)
    if marker.exists():
        logger.info("Run already finished (%s), skipping", marker)
        return str(directory)
    logger.info("Original prompt: %s", prompt)
    logger.info("Generation prompt: %s", generation_prompt)
    logger.info("Seed %s | generations %s | population %s | generator %s", seed, generations, population, generator)

    set_all_seeds(seed)
    if generator == "sdxl":
        model, batch_size = StableDiffusionXLModel(DEVICE, HF_CACHE, num_inference_steps=steps or 30), 8
    else:
        model, batch_size = ThinkThenGenerate(DEVICE, HF_CACHE, num_inference_steps=steps or 30), 4
    clip = ClipEmbeddingModel(DEVICE, HF_CACHE)
    blip2 = Blip2EmbeddingModel(DEVICE, HF_CACHE)

    fitness = [MaxMeanDivergenceEvaluator(blip2, DEVICE)]
    metrics = [MaxMeanDivergenceEvaluator(blip2, DEVICE),
               FixedPromptEvaluator(CLIPScoreEvaluator(clip, DEVICE), prompt, "CLIPScore"),
               DiversityWithinPopulationEvaluator(blip2, DEVICE)]
    if condition == "t2g":
        metrics.insert(2, FixedPromptEvaluator(CLIPScoreEvaluator(clip, DEVICE), generation_prompt, "CLIPScoreRevised"))

    pipe = build_pipeline(model, generation_prompt, fitness, metrics, run_id, generations, population, directory,
                          batch_size=batch_size)
    pipe.save_config()
    run_dir = Path(pipe.result_path)
    (run_dir / f"{run_id}_t2g.json").write_text(json.dumps({
        "condition": condition, "generator": generator, "num_inference_steps": steps or 30,
        "original_prompt": prompt, "generation_prompt": generation_prompt,
        "seed": seed, "num_generations": generations, "population_size": population,
        "rewrite_source": rewrites["_source"] if condition == "t2g" else None,
        "reasoning": rewrites["prompts"].get(prompt, {}).get("reasoning") if condition == "t2g" else None,
    }, indent=2, ensure_ascii=False))
    pipe.run()
    marker.write_text("ok\n", encoding="utf-8")
    return str(run_dir)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="original prompt (key in the rewrite file)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--condition", choices=["t2g", "original"], default="t2g")
    parser.add_argument("--generator", choices=list(GENERATORS), default="sdxl")
    parser.add_argument("--steps", type=int, default=None, help="denoising steps (default 30)")
    args = parser.parse_args()
    configure_logging()
    log_environment()
    result = run(args.prompt, args.seed, args.generations, args.population, args.condition, args.generator, args.steps)
    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
