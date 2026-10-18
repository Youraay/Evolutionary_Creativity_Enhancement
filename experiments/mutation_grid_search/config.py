"""Grid and paths of the mutation grid search."""
from experiments.common.settings import result_dir

RESULT_DIR = result_dir("mutation_grid_search")
PROMPT = "a cat"
SEED = 1943746817
CANDIDATES = 5
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
"""Mutated fraction ``p_m`` of the noise values."""
STRENGTHS = [0.2, 0.5, 0.8, 1.0, 2.0]
"""Standard deviation ``sigma_m`` of the added Gaussian noise."""
INFERENCE_STEPS = 50
