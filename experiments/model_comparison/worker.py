"""Run one job of the model comparison.

Same GA as the PartiPrompts experiment; the text-to-image model comes from the ``model`` column.

    python -m experiments.model_comparison.worker <job_id>
"""
import logging
import sqlite3

from evolutionary_creativity.models.embeddings import Blip2EmbeddingModel, ClipEmbeddingModel
from experiments.common.ga import (build_pipeline, new_run_id, run_queued_job, set_all_seeds, standard_evaluators,
                                   success_marker)
from experiments.common.settings import DEVICE, HF_CACHE, NUM_GENERATIONS, POPULATION_SIZE
from experiments.model_comparison.config import RESULT_DIR, job_queue
from experiments.model_comparison.models import load_model

logger = logging.getLogger(__name__)


def run(job: sqlite3.Row) -> str:
    """Run the GA for one model, prompt and seed.

    Args:
        job: Row of the job database.

    Returns:
        Result directory of the run (or of the prompt, if the run had already finished).
    """
    prompt, seed, model_name = job["prompt"], int(job["seed"]), job["model"]
    slug, run_id = new_run_id(prompt, seed)
    directory = RESULT_DIR / model_name / slug
    directory.mkdir(parents=True, exist_ok=True)
    marker = success_marker(directory, seed)
    if marker.exists():
        logger.info("Job %s already finished (%s), skipping", job["id"], marker)
        return str(directory)
    logger.info("Job %s | model %s | prompt %r | seed %s | output %s", job["id"], model_name, prompt, seed, directory)

    set_all_seeds(seed)
    generator = load_model(model_name, DEVICE, HF_CACHE)
    clip = ClipEmbeddingModel(DEVICE, HF_CACHE)
    blip2 = Blip2EmbeddingModel(DEVICE, HF_CACHE)
    for component in ("unet", "transformer", "vae"):     # UNet models have .unet, DiT models .transformer
        module = getattr(generator.model, component, None)
        if module is not None:
            parameter = next(module.parameters())
            logger.info("%s device: %s | dtype: %s", component, parameter.device, parameter.dtype)
    fitness, metrics = standard_evaluators(blip2, clip, DEVICE)

    pipe = build_pipeline(generator, prompt, fitness, metrics, run_id, NUM_GENERATIONS, POPULATION_SIZE, directory)
    pipe.save_config()
    pipe.run()
    marker.write_text("ok\n", encoding="utf-8")
    return str(pipe.result_path)


if __name__ == "__main__":
    run_queued_job(job_queue(), run, "Run one job of the model comparison.")
