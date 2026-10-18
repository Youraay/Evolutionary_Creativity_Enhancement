"""Seed transfer: noise optimised for one prompt rendered with another prompt.

Fig. 3 and Supplementary Fig. 10.

Variants (configuration of every run in ``config.py`` and in the job database):

    <source>_to_<target>  generations 0, 10, 20 and 30 of the ``example_prompts`` run of <source> (same seed)
                          rendered and evaluated with <target>; no evolution

Standard configuration: population 100, 30 generations, SDXL with 50 steps, tournament size 3, uniform
crossover (rate 0.9), Gaussian mutation of 10 % of the values with standard deviation 0.2 (probability 0.05),
elitism 1, novelty fitness in BLIP-2 space (see ``experiments.common.variants``).

    python -m experiments.seed_transfer.create_jobs --db new_jobs.db       # open jobs for all variants
    SEED_TRANSFER_DB_FILE=new_jobs.db python -m experiments.seed_transfer.dispatch

``data/jobs/seed_transfer.db`` lists the analysed runs (``create_jobs --register``).
"""
