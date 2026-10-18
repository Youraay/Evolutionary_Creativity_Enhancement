"""Paths and variants of the experiment example_prompts."""
from experiments.common.settings import db_file, result_dir
from experiments.common.variants import Variant, VariantExperiment, prompt_name

PROMPTS = ["a cat", "a dog", "a chair", "a coffee mug", "boredom", "creativity", "a capybara", "a car", "cat",
           "chaos", "knowledge", "love"]

EXPERIMENT = VariantExperiment(
    name="example_prompts",
    variants=[Variant(prompt_name(prompt), prompt) for prompt in PROMPTS],
    db_file=db_file("example_prompts"),
    result_dir=result_dir("example_prompts"),
)
