"""Paths and variants of the experiment tournament_size."""
from experiments.common.embedding_spaces import EmbeddingExperiment
from experiments.common.settings import db_file, result_dir

EXPERIMENT = EmbeddingExperiment(
    name="tournament_size",
    variants=[(f"tsize_{size}", {"tournament_size": size}) for size in (2, 3, 5, 10)],
    db_file=db_file("tournament_size"),
    result_dir=result_dir("tournament_size"),
)
