"""Fitness function: distance to the population centroid vs. k nearest neighbours ("a cat", seeds 4-6).

Supplementary Fig. 4.

Variants (configuration of every run in ``config.py`` and in the job database):

    Centroid              distance to the mean embedding of the population (standard)
    kNN_10, kNN_50        mean distance to the 10 / 50 nearest neighbours in the population
    kNN_100               mean distance to all 99 other candidates

Standard configuration: population 100, 30 generations, SDXL with 50 steps, tournament size 3, uniform
crossover (rate 0.9), Gaussian mutation of 10 % of the values with standard deviation 0.2 (probability 0.05),
elitism 1, novelty fitness in BLIP-2 space (see ``experiments.common.variants``).

CLIP scores of the images are in ``<result dir>/clip_scores``.

    python -m experiments.fitness_function.create_jobs --db new_jobs.db       # open jobs for all variants
    FITNESS_FUNCTION_DB_FILE=new_jobs.db python -m experiments.fitness_function.worker <job id>
``data/jobs/fitness_function.db`` lists the analysed runs (``create_jobs --register``).
"""
