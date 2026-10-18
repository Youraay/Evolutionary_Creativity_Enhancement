"""Crossover function: n-point, uniform and arithmetic crossover ("a cat", seeds 1-3).

Supplementary Fig. 7.

Variants (configuration of every run in ``config.py`` and in the job database):

    1Point, 3Point, 7Point   n-point crossover on the flattened noise (equal segments)
    *_Pixel                  the same operator (all channels of a position switch together in both cases);
                             the runs are pooled with the plain variant and not scheduled again
    Arithmetic               weighted mean of the parents, renormalised to unit variance
    Uniform                  every value from a random parent (standard)
    Uniform_Pixel            every position (all channels) from a random parent

Standard configuration: population 100, 30 generations, SDXL with 50 steps, tournament size 3, uniform
crossover (rate 0.9), Gaussian mutation of 10 % of the values with standard deviation 0.2 (probability 0.05),
elitism 1, novelty fitness in BLIP-2 space (see ``experiments.common.variants``).

CLIP scores of the images are in ``<result dir>/clip_scores``.

    python -m experiments.crossover_function.create_jobs --db new_jobs.db       # open jobs for all variants
    CROSSOVER_FUNCTION_DB_FILE=new_jobs.db python -m experiments.crossover_function.dispatch

``data/jobs/crossover_function.db`` lists the analysed runs (``create_jobs --register``).
"""
