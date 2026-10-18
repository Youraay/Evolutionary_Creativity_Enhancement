"""Paths and variants of the experiment operator_ablation."""
from experiments.common.settings import db_file, result_dir
from experiments.common.variants import Variant, VariantExperiment
from experiments.mutation_only.config import MUTATION_ONLY

EXPERIMENT = VariantExperiment(
    name="operator_ablation",
    variants=[
        Variant("No_Crossover", seeds="7-9", overrides=MUTATION_ONLY, rerun=False),
        Variant("No_Elites", seeds="7-9", overrides={"elite_size": 0}),
        Variant("No_Mutation", seeds="7-9", overrides={"mutation_rate": 0.0, "crossover_rate": 1.0, "elite_size": 0}),
    ],
    db_file=db_file("operator_ablation"),
    result_dir=result_dir("operator_ablation"),
)
