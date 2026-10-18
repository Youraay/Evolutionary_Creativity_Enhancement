"""Paths and variants of the experiment mutation_only."""
from experiments.common.settings import db_file, result_dir
from experiments.common.variants import Variant, VariantExperiment

MUTATION_ONLY = {"mutator": "gaussian:0.5:0.2", "mutation_rate": 1.0, "crossover_rate": 0.0, "elite_size": 0}
"""Every child is mutated; half of the values are changed."""

EXPERIMENT = VariantExperiment(
    name="mutation_only",
    variants=[
        Variant("gaussean_only", seeds="7-9", overrides=MUTATION_ONLY),
        Variant("uniform_only", seeds="7-9", overrides={**MUTATION_ONLY, "mutator": "uniform:0.5"}),
    ],
    db_file=db_file("mutation_only"),
    result_dir=result_dir("mutation_only"),
)
