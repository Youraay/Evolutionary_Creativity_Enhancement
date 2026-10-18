"""Paths and variants of the experiment crossover_function."""
from experiments.common.settings import db_file, result_dir
from experiments.common.variants import Variant, VariantExperiment

EXPERIMENT = VariantExperiment(
    name="crossover_function",
    variants=[
        Variant(f"{n}Point{suffix}", overrides={"crossover": f"n_point:{n}"}, rerun=not suffix)
        for n in (1, 3, 7) for suffix in ("", "_Pixel")
    ] + [
        Variant("Arithmetic", overrides={"crossover": "arithmetic"}),
        Variant("Uniform"),
        Variant("Uniform_Pixel", overrides={"crossover": "uniform_pixel"}),
    ],
    db_file=db_file("crossover_function"),
    result_dir=result_dir("crossover_function"),
)
