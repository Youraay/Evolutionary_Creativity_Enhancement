"""Tournament size: selection tournaments of 2, 3, 5 or 10 candidates.

Supplementary Fig. 8.

Variants, each on "a cat", "a chair" and "boredom" with the seeds 1943746817, 1394004965 and 3030801653:

    tsize_2, tsize_3, tsize_5, tsize_10   tournament size (fitness BLIP-2, 30 generations)

Standard configuration: population 60, SDXL with 30 steps, uniform crossover (rate 0.9), Gaussian mutation of
10 % of the values with standard deviation 0.2 (probability 0.05), elitism 1. Every run records novelty and
diversity in BLIP-2, CLIP, DINOv2 and SigLIP 2 space (see ``experiments.common.embedding_spaces``).

    python -m experiments.tournament_size.create_jobs --db new_jobs.db
    TOURNAMENT_SIZE_DB_FILE=new_jobs.db python -m experiments.tournament_size.dispatch
    experiments/common/start_experiments.sh tournament_size start      # alternative: run under another Slurm user

``data/jobs/tournament_size.db`` lists the analysed runs.
"""
