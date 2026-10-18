"""Paths and variants of the experiment seed_transfer."""
from experiments.common.settings import db_file, result_dir
from experiments.common.variants import VariantExperiment, transfer_variants

TRANSFER_PROMPTS = ["a cat", "a dog", "a chair", "a coffee mug", "boredom", "creativity"]

EXPERIMENT = VariantExperiment(
    name="seed_transfer",
    variants=transfer_variants(TRANSFER_PROMPTS),
    db_file=db_file("seed_transfer"),
    result_dir=result_dir("seed_transfer"),
    transfer_source_dir=result_dir("example_prompts"),
)
