"""Long runs: 300 generations of the standard GA.

Supplementary Fig. 1D.

Variants, each on "a cat", "a chair" and "boredom" with the seeds 1943746817, 1394004965 and 3030801653:

    gen_300               300 generations (fitness BLIP-2, tournament size 3)

Standard configuration: population 60, SDXL with 30 steps, uniform crossover (rate 0.9), Gaussian mutation of
10 % of the values with standard deviation 0.2 (probability 0.05), elitism 1. Every run records novelty and
diversity in BLIP-2, CLIP, DINOv2 and SigLIP 2 space (see ``experiments.common.embedding_spaces``).

    python -m experiments.long_runs.create_jobs --db new_jobs.db
    LONG_RUNS_DB_FILE=new_jobs.db python -m experiments.long_runs.worker <job id>

``data/jobs/long_runs.db`` lists the analysed runs.
"""
