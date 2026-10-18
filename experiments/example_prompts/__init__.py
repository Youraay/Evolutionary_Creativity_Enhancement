"""Example prompts: the standard GA (population 100) on twelve prompts with three seeds each.

Fig. 1E, Fig. 2A-C, 2E-F, Supplementary Figs. 1A-C, 5A and 11; the noise of these runs is the source of
``seed_transfer``.

Variants (configuration of every run in ``config.py`` and in the job database):

    a_cat, a_dog, ...     one variant per prompt, standard configuration, seeds 1-3

Standard configuration: population 100, 30 generations, SDXL with 50 steps, tournament size 3, uniform
crossover (rate 0.9), Gaussian mutation of 10 % of the values with standard deviation 0.2 (probability 0.05),
elitism 1, novelty fitness in BLIP-2 space (see ``experiments.common.variants``).

CLIP scores of the images and random baselines (computed after the runs) are in ``<result dir>/clip_scores``.

    python -m experiments.example_prompts.create_jobs --db new_jobs.db       # open jobs for all variants
    EXAMPLE_PROMPTS_DB_FILE=new_jobs.db python -m experiments.example_prompts.dispatch

``data/jobs/example_prompts.db`` lists the analysed runs (``create_jobs --register``).
"""
