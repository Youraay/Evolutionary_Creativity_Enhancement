"""Experiments that record novelty and diversity in four embedding spaces (fitness embedding, tournament size,
long runs).

The GA is the standard setup (SDXL with 30 steps, uniform crossover, Gaussian mutation, elitism 1) on
"a cat", "a chair" and "boredom" with three seeds each; a job sets the fitness embedding, the tournament size
and the number of generations. Every run records the distance to the mean of generation 0
(``MMD_<embedding>``) and the diversity within the generation (``Diversity_<embedding>``) in BLIP-2, CLIP,
DINOv2 and SigLIP 2 space plus CLIP and SigLIP scores, so all three experiments share the same metrics.

Run directories: ``<result dir>/<variant>/<prompt_slug>/<prompt_slug>_<seed>_<timestamp>/``.
"""
import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from experiments.common.job_queue import JobQueue
from experiments.common.settings import NUM_GENERATIONS, POPULATION_SIZE, SEEDS, submit_script

logger = logging.getLogger(__name__)

PROMPTS = ["a cat", "a chair", "boredom"]

BASE = {"fitness_embedding": "blip2", "tournament_size": 3, "num_generations": NUM_GENERATIONS}
"""Configuration of the standard setup; every variant changes one entry."""

COLUMNS = [
    ("variant", "TEXT NOT NULL"),
    ("prompt", "TEXT NOT NULL"),
    ("seed", "TEXT NOT NULL"),
    ("fitness_embedding", "TEXT NOT NULL"),
    ("tournament_size", "INTEGER NOT NULL"),
    ("num_generations", "INTEGER NOT NULL"),
    ("population_size", "INTEGER NOT NULL"),
    ("note", "TEXT"),
]
"""Every row holds the complete GA configuration, so the worker does not depend on the environment."""

JOB_INFO_KEYS = ("id", "variant", "prompt", "seed", "fitness_embedding", "tournament_size", "num_generations",
                 "population_size")

TEXT_ALIGNED = {"clip": "CLIPScore", "siglip": "SigLIPScore"}
"""Image-text models whose prompt fidelity is recorded, with the metric name."""


@dataclass
class EmbeddingExperiment:
    """Paths and variants of one experiment.

    Attributes:
        name: Package name below ``experiments``.
        variants: ``(variant, deviation from BASE)`` per variant.
        db_file: Job database.
        result_dir: Directory of the variant directories.
        time_limit: Slurm time limit that replaces the one of the batch script, if set.
    """

    name: str
    variants: list[tuple[str, dict]]
    db_file: Path
    result_dir: Path
    time_limit: str | None = None

    def job_queue(self, db_file: Path | None = None) -> JobQueue:
        """Return the job database.

        Args:
            db_file: Database path; by default :attr:`db_file`.
        """
        return JobQueue(str(db_file or self.db_file), COLUMNS)


def create_jobs_cli(experiment: EmbeddingExperiment, description: str) -> None:
    """Command line entry point of ``experiments.<name>.create_jobs``: variant x prompt x seed.

    The dispatcher claims jobs by id, so the variants finish one after another and can be analysed early.

    Args:
        experiment: The experiment.
        description: Help text (the module docstring).
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=experiment.db_file,
                        help=f"SQLite job database (default: {experiment.name.upper()}_DB_FILE)")
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--append", action="store_true", help="add jobs to an existing database")
    args = parser.parse_args()
    if args.db.exists() and not args.append:
        raise SystemExit(f"{args.db} already exists. Choose another --db or use --append.")

    jobs = [{"variant": variant, "prompt": prompt, "seed": seed, "population_size": args.population_size, "note": "",
             **BASE, **overrides}
            for variant, overrides in experiment.variants for prompt in args.prompts for seed in args.seeds]
    queue = experiment.job_queue(args.db)
    queue.create_table()
    print(f"{queue.add_jobs(jobs)} jobs written to {args.db} ({', '.join(v for v, _ in experiment.variants)})")
    print(f"Start with: {experiment.name.upper()}_DB_FILE={args.db} python -m experiments.{experiment.name}.dispatch")


def dispatch_cli(experiment: EmbeddingExperiment) -> None:
    """Command line entry point of ``experiments.<name>.dispatch``.

    Args:
        experiment: The experiment.
    """
    from experiments.common.dispatcher import run_dispatcher

    run_dispatcher(experiment.job_queue(), submit_script(experiment.name), time_limit=lambda job: experiment.time_limit,
                   describe=lambda job: f"({job['variant']})")


def named(evaluator, name: str):
    """Rename an evaluator so that several embeddings give distinct CSV columns.

    Args:
        evaluator: Evaluator to rename.
        name: New metric name.

    Returns:
        The same evaluator.
    """
    evaluator.name = name
    return evaluator


def build_evaluators(embeddings: dict, fitness_embedding: str) -> tuple[list, list]:
    """Create the fitness evaluator and the metrics of all embedding spaces.

    Args:
        embeddings: Loaded embedding models by name.
        fitness_embedding: Name of the embedding that defines the fitness.

    Returns:
        ``(fitness_evaluators, model_evaluators)``; the metrics are ``MMD_<name>`` and ``Diversity_<name>``
        for every embedding plus ``CLIPScore`` and ``SigLIPScore``.
    """
    from evolutionary_creativity.evaluators import (CLIPScoreEvaluator, DiversityWithinPopulationEvaluator,
                                                    MaxMeanDivergenceEvaluator)
    from experiments.common.settings import DEVICE

    fitness = [named(MaxMeanDivergenceEvaluator(embeddings[fitness_embedding], DEVICE), f"MMD_{fitness_embedding}")]
    metrics = []
    for name, embedding in embeddings.items():
        metrics.append(named(MaxMeanDivergenceEvaluator(embedding, DEVICE), f"MMD_{name}"))
        metrics.append(named(DiversityWithinPopulationEvaluator(embedding, DEVICE), f"Diversity_{name}"))
    for name, label in TEXT_ALIGNED.items():
        metrics.append(named(CLIPScoreEvaluator(embeddings[name], DEVICE), label))
    return fitness, metrics


def run_job(experiment: EmbeddingExperiment, job: sqlite3.Row) -> str:
    """Run the GA for one job.

    Besides the standard result files, ``<run>_experiment.json`` records the job configuration, which the
    pipeline config does not contain (tournament size, fitness embedding).

    Args:
        experiment: The experiment.
        job: Row of the job database.

    Returns:
        Result directory of the run (or of the prompt, if the run had already finished).
    """
    from evolutionary_creativity.models.embeddings import (Blip2EmbeddingModel, ClipEmbeddingModel,
                                                           DinoEmbeddingModel, SiglipEmbeddingModel)
    from evolutionary_creativity.models.text_to_image.sdxl import StableDiffusionXLModel
    from experiments.common.ga import build_pipeline, new_run_id, set_all_seeds, success_marker
    from experiments.common.settings import DEVICE, HF_CACHE

    embedding_models = {
        "blip2": Blip2EmbeddingModel,     # Salesforce/blip2-opt-2.7b, vision encoder (EVA ViT-g)
        "clip": ClipEmbeddingModel,       # openai/clip-vit-base-patch32
        "dino": DinoEmbeddingModel,       # facebook/dinov2-base
        "siglip": SiglipEmbeddingModel,   # google/siglip2-base-patch16-224
    }
    prompt, seed = job["prompt"], int(job["seed"])
    slug, run_id = new_run_id(prompt, seed)
    directory = experiment.result_dir / job["variant"] / slug
    directory.mkdir(parents=True, exist_ok=True)
    marker = success_marker(directory, seed)
    if marker.exists():
        logger.info("Job %s already finished (%s), skipping", job["id"], marker)
        return str(directory)
    logger.info("Job %s | %s | output %s", job["id"], {key: job[key] for key in JOB_INFO_KEYS[1:]}, directory)

    set_all_seeds(seed)
    sdxl = StableDiffusionXLModel(DEVICE, HF_CACHE, num_inference_steps=30)
    embeddings = {name: cls(DEVICE, HF_CACHE) for name, cls in embedding_models.items()}
    fitness, metrics = build_evaluators(embeddings, job["fitness_embedding"])

    pipe = build_pipeline(sdxl, prompt, fitness, metrics, run_id, int(job["num_generations"]),
                          int(job["population_size"]), directory, tournament_size=int(job["tournament_size"]))
    pipe.save_config()
    info = {key: job[key] for key in JOB_INFO_KEYS}
    info["model_evaluators"] = [e.name for e in metrics]
    (Path(pipe.config_path).parent / f"{run_id}_experiment.json").write_text(json.dumps(info, indent=4),
                                                                           encoding="utf-8")
    pipe.run()
    marker.write_text("ok\n", encoding="utf-8")
    return str(pipe.result_path)


def worker_cli(experiment: EmbeddingExperiment) -> None:
    """Command line entry point of ``experiments.<name>.worker``.

    Args:
        experiment: The experiment.
    """
    from experiments.common.ga import run_queued_job

    run_queued_job(experiment.job_queue(), lambda job: run_job(experiment, job),
                   f"Run one job of the experiment {experiment.name}.")
