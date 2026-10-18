"""Setup shared by all genetic algorithm runs: hyperparameters, seeding, run directories and job handling."""
import argparse
import logging
import os
import random
import sqlite3
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch

from evolutionary_creativity.evaluators import (BaseImageEvaluator, CLIPScoreEvaluator,
                                                DiversityWithinPopulationEvaluator, MaxMeanDivergenceEvaluator)
from evolutionary_creativity.models.embeddings import EmbeddingModel
from evolutionary_creativity.models.text_to_image.base import GenerativeModel
from evolutionary_creativity.noise_factory import NoiseFactory
from evolutionary_creativity.operators import TournamentSelector, UniformCrossover, UniformGaussianMutator
from evolutionary_creativity.pipeline import GeneticAlgorithmPipeline
from experiments.common.job_queue import JobQueue
from experiments.common.naming import slugify_prompt

logger = logging.getLogger(__name__)

TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.05
"""Probability that a child is mutated."""
MUTATED_FRACTION = 0.1
"""Fraction of the latent values perturbed by one mutation."""
MUTATION_STRENGTH = 0.2
"""Standard deviation of the Gaussian mutation."""
CROSSOVER_RATE = 0.9
ELITE_SIZE = 1
BATCH_SIZE = 8


def configure_logging() -> None:
    """Log to stdout (level from ``LOGLEVEL``, default ``INFO``) and hide the Diffusers progress bars."""
    from diffusers.utils.logging import disable_progress_bar

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    disable_progress_bar()


def log_environment() -> None:
    """Log the versions of the deep learning stack and the GPU in use."""
    import diffusers
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    logger.info("torch %s (CUDA %s) | diffusers %s | transformers %s | GPU %s", torch.__version__,
                torch.version.cuda, diffusers.__version__, transformers.__version__, gpu)


def set_all_seeds(seed: int) -> None:
    """Seed Python's ``random`` and all torch generators.

    Args:
        seed: Seed of the run.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def new_run_id(prompt: str, seed: int) -> tuple[str, str]:
    """Create the prompt slug and a unique run id ``<slug>_<seed>_<YYYYmmdd_HHMMSS>``.

    Args:
        prompt: Text prompt.
        seed: Seed of the run.

    Returns:
        ``(prompt_slug, run_id)``.
    """
    slug = slugify_prompt(prompt)
    return slug, f"{slug}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def success_marker(directory: Path, key: int | str) -> Path:
    """Path of the marker file that records a finished run in ``directory``.

    A requeued job finds the marker and skips the recomputation.

    Args:
        directory: Directory of the runs of one prompt or experiment.
        key: Identifies the run within ``directory`` (the seed, or the run name).
    """
    return directory / f"SUCCESS_{key}"


def standard_evaluators(blip2: EmbeddingModel, clip: EmbeddingModel,
                        device: str) -> tuple[list[BaseImageEvaluator], list[BaseImageEvaluator]]:
    """Evaluators of the standard setup.

    Fitness is the novelty in BLIP-2 space (distance to the mean of the current population). Recorded
    for analysis are the distance to the mean of generation 0 (``model_LocalMaxMeanDivergence``), the
    CLIP score (``model_CLIPScore``) and the diversity within the generation
    (``model_DiversityWithinPopulation``).

    Args:
        blip2: BLIP-2 embedding model.
        clip: CLIP embedding model.
        device: Device of the evaluators.

    Returns:
        ``(fitness_evaluators, model_evaluators)``.
    """
    fitness = [MaxMeanDivergenceEvaluator(blip2, device)]
    metrics = [MaxMeanDivergenceEvaluator(blip2, device), CLIPScoreEvaluator(clip, device),
               DiversityWithinPopulationEvaluator(blip2, device)]
    return fitness, metrics


def build_pipeline(generative_model: GenerativeModel, prompt: str, fitness_evaluators: list[BaseImageEvaluator],
                   model_evaluators: list[BaseImageEvaluator], run_id: str, num_generations: int,
                   population_size: int, result_dir: Path, tournament_size: int = TOURNAMENT_SIZE,
                   batch_size: int = BATCH_SIZE) -> GeneticAlgorithmPipeline:
    """Create the genetic algorithm with the operators and rates shared by all experiments.

    Args:
        generative_model: Text-to-image model.
        prompt: Prompt used for generation.
        fitness_evaluators: Evaluators that define the fitness.
        model_evaluators: Evaluators recorded for analysis.
        run_id: Name of the run directory.
        num_generations: Number of evolution steps.
        population_size: Number of candidates per generation.
        result_dir: Directory in which the run directory is created.
        tournament_size: Size of the selection tournaments.
        batch_size: Images per call of the text-to-image model.

    Returns:
        The configured pipeline (not yet run).
    """
    return GeneticAlgorithmPipeline(
        generative_model=generative_model,
        prompt=prompt,
        crossover_operation=UniformCrossover(),
        selector=TournamentSelector(tournament_size=tournament_size),
        mutator=UniformGaussianMutator(mutation_rate=MUTATED_FRACTION, mutation_strength=MUTATION_STRENGTH),
        noise_factory=NoiseFactory(),
        fitness_evaluator=fitness_evaluators,
        model_evaluator=model_evaluators,
        experiment_id=run_id,
        num_generations=num_generations,
        population_size=population_size,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        elite_size=ELITE_SIZE,
        batch_size=batch_size,
        result_path=str(result_dir),
    )


def run_queued_job(queue: JobQueue, run: Callable[[sqlite3.Row], str], description: str) -> None:
    """Command line entry point of a worker: run the job given as argument and record the outcome.

    Args:
        queue: Job database of the experiment.
        run: Function that executes one job and returns its result directory.
        description: Help text of the command line interface.

    Raises:
        SystemExit: If the job id does not exist.
        Exception: Any error of ``run`` is re-raised after the job was marked as failed.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("job_id", type=int, help="id of the job in the job database")
    job_id = parser.parse_args().job_id

    configure_logging()
    log_environment()
    job = queue.get_job(job_id)
    if job is None:
        raise SystemExit(f"Job {job_id} not found in {queue.db_file}")
    try:
        output_path = run(job)
    except Exception as error:
        queue.mark_failed(job_id, str(error))
        logger.error("Job %s FAILED: %s", job_id, error)
        raise
    queue.mark_done(job_id, output_path)
    logger.info("Job %s DONE", job_id)
