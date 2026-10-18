"""Operator ablation: the GA without crossover, without elitism or without mutation ("a cat", seeds 7-9).

Not part of the manuscript.

Variants (configuration of every run in ``config.py`` and in the job database):

    No_Crossover          no crossover, every child mutated (Gaussian); the configuration of
                          ``mutation_only/gaussean_only``, hence not scheduled again
    No_Elites             standard configuration without elitism
    No_Mutation           crossover for every child, no mutation, no elitism

Standard configuration: population 100, 30 generations, SDXL with 50 steps, tournament size 3, uniform
crossover (rate 0.9), Gaussian mutation of 10 % of the values with standard deviation 0.2 (probability 0.05),
elitism 1, novelty fitness in BLIP-2 space (see ``experiments.common.variants``).

    python -m experiments.operator_ablation.create_jobs --db new_jobs.db       # open jobs for all variants
    OPERATOR_ABLATION_DB_FILE=new_jobs.db python -m experiments.operator_ablation.worker <job id>
``data/jobs/operator_ablation.db`` lists the analysed runs (``create_jobs --register``).
"""
