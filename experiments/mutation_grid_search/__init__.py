"""Grid search of the Gaussian mutation (Methods, section Mutation).

For every candidate of an initial population of "a cat" (SDXL in float16, 50 steps, as the experiments with
population 100), the noise is mutated once with every combination of the mutated fraction ``p_m`` and the
standard deviation ``sigma_m`` of ``UniformGaussianMutator``, and the mutated image is rendered. The images are
compared visually (``analysis.mutation_grid_search``); the table records the BLIP-2 distance
``(1 - cos) / 2`` of every mutated image to the image of the unmutated noise.

    python -m experiments.mutation_grid_search.worker
    python -m experiments.mutation_grid_search.worker --fractions 0.1 0.3 0.5 --strengths 0.1 0.2 0.3 0.5 0.8 1.3

Results: ``<MUTATION_GRID_SEARCH_RESULT_DIR>/<prompt_slug>_<seed>/`` with ``original_c<k>.JPEG``,
``mutated_c<k>_p<p_m>_s<sigma_m>.JPEG`` and ``<prompt_slug>_<seed>.csv``.
"""
