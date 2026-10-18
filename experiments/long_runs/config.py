"""Paths and variants of the experiment long_runs."""
from experiments.common.embedding_spaces import EmbeddingExperiment
from experiments.common.settings import db_file, result_dir

EXPERIMENT = EmbeddingExperiment(
    name="long_runs",
    variants=[("gen_300", {"num_generations": 300})],
    db_file=db_file("long_runs"),
    result_dir=result_dir("long_runs"),
    time_limit="3-00:00:00",
)
"""300 SDXL generations take about 15 h; the time limit replaces the 15 h of the batch script."""
