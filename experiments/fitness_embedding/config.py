"""Paths and variants of the experiment fitness_embedding."""
from experiments.common.embedding_spaces import EmbeddingExperiment
from experiments.common.settings import db_file, result_dir

EXPERIMENT = EmbeddingExperiment(
    name="fitness_embedding",
    variants=[(f"fitness_{embedding}", {"fitness_embedding": embedding})
              for embedding in ("blip2", "clip", "dino", "siglip")],
    db_file=db_file("fitness_embedding"),
    result_dir=result_dir("fitness_embedding"),
)
