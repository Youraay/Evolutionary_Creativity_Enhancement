"""Mutation only: every child is a mutated copy of its parent, no crossover and no elitism ("a cat", seeds 7-9).

Supplementary Fig. 9.

Variants (configuration of every run in ``config.py`` and in the job database):

    gaussean_only         Gaussian noise (standard deviation 0.2) added to half of the values
    uniform_only          half of the values replaced by new standard normal values

Standard configuration: population 100, 30 generations, SDXL with 50 steps, tournament size 3, uniform
crossover (rate 0.9), Gaussian mutation of 10 % of the values with standard deviation 0.2 (probability 0.05),
elitism 1, novelty fitness in BLIP-2 space (see ``experiments.common.variants``).

    python -m experiments.mutation_only.create_jobs --db new_jobs.db       # open jobs for all variants
    MUTATION_ONLY_DB_FILE=new_jobs.db python -m experiments.mutation_only.dispatch

``data/jobs/mutation_only.db`` lists the analysed runs (``create_jobs --register``).
"""
