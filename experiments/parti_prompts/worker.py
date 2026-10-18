"""Run one job of the PartiPrompts experiment (called by ``submit_job.sh``).

Setup: SDXL with 30 steps, population 60, 30 generations, tournament size 3, uniform crossover,
Gaussian mutation, elitism 1; fitness = novelty in BLIP-2 space.

    python -m experiments.parti_prompts.worker <job_id>
"""
import logging
import sqlite3

from evolutionary_creativity.models.embeddings import Blip2EmbeddingModel, ClipEmbeddingModel
from evolutionary_creativity.models.text_to_image.sdxl import StableDiffusionXLModel
from experiments.common.ga import (build_pipeline, new_run_id, run_queued_job, set_all_seeds, standard_evaluators,
                                   success_marker)
from experiments.common.settings import DEVICE, HF_CACHE, NUM_GENERATIONS, POPULATION_SIZE
from experiments.parti_prompts.config import RESULT_DIR, job_queue

logger = logging.getLogger(__name__)


def run(job: sqlite3.Row) -> str:
    """Run the GA for one prompt and seed.

    Args:
        job: Row of the job database.

    Returns:
        Result directory of the run (or of the prompt, if the run had already finished).
    """
    prompt, seed = job["prompt"], int(job["seed"])
    slug, run_id = new_run_id(prompt, seed)
    directory = RESULT_DIR / slug
    directory.mkdir(parents=True, exist_ok=True)
    marker = success_marker(directory, seed)
    if marker.exists():
        logger.info("Job %s already finished (%s), skipping", job["id"], marker)
        return str(directory)
    logger.info("Job %s | prompt %r | seed %s | output %s", job["id"], prompt, seed, directory)

    set_all_seeds(seed)
    sdxl = StableDiffusionXLModel(DEVICE, HF_CACHE, num_inference_steps=30)
    clip = ClipEmbeddingModel(DEVICE, HF_CACHE)
    blip2 = Blip2EmbeddingModel(DEVICE, HF_CACHE)
    fitness, metrics = standard_evaluators(blip2, clip, DEVICE)

    pipe = build_pipeline(sdxl, prompt, fitness, metrics, run_id, NUM_GENERATIONS, POPULATION_SIZE, directory)
    pipe.save_config()
    pipe.run()
    marker.write_text("ok\n", encoding="utf-8")
    return str(pipe.result_path)


if __name__ == "__main__":
    run_queued_job(job_queue(), run, "Run one job of the PartiPrompts experiment.")
