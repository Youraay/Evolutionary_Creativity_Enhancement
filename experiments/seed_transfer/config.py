"""Paths and variants of the experiment seed_transfer."""
from experiments.common.settings import db_file, result_dir
from experiments.common.variants import VariantExperiment, transfer_variants

TRANSFER_PROMPTS = ["a cat", "a dog", "a chair", "a coffee mug", "boredom", "creativity"]

SOURCE_CORRECTIONS = {"a_chair_Seed_1": "a coffee mug", "a_coffee_mug_Seed_1": "a chair"}
"""When the transfer runs were computed, the ``example_prompts`` runs of seed 1 of "a chair" and "a coffee mug" were
stored in each other's directory. The runs ``a_chair_Seed_1_to_*`` therefore render the noise of "a coffee mug"
seed 1 and ``a_coffee_mug_Seed_1_to_*`` that of "a chair" seed 1 (verified by comparing the noise tensors of the
archives); the job database lists them with their actual source prompt."""

EXPERIMENT = VariantExperiment(
    name="seed_transfer",
    variants=transfer_variants(TRANSFER_PROMPTS),
    db_file=db_file("seed_transfer"),
    result_dir=result_dir("seed_transfer"),
    transfer_source_dir=result_dir("example_prompts"),
    source_corrections=SOURCE_CORRECTIONS,
)
