"""Paths and variants of the experiment fitness_function."""
from experiments.common.settings import db_file, result_dir
from experiments.common.variants import Variant, VariantExperiment

EXPERIMENT = VariantExperiment(
    name="fitness_function",
    variants=[
        Variant("Centroid", seeds="4-6"),
        Variant("kNN_10", seeds="4-6", overrides={"fitness": "knn:10"}),
        Variant("kNN_50", seeds="4-6", overrides={"fitness": "knn:50"}),
        Variant("kNN_100", seeds="4-6", overrides={"fitness": "knn:99"}),
    ],
    db_file=db_file("fitness_function"),
    result_dir=result_dir("fitness_function"),
)
