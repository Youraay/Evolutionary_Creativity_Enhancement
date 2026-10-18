"""Fitness embedding: novelty measured with BLIP-2, CLIP, DINOv2 or SigLIP 2.

Supplementary Fig. 3C-D.

Variants, each on "a cat", "a chair" and "boredom" with the seeds 1943746817, 1394004965 and 3030801653:

    fitness_blip2, fitness_clip, fitness_dino, fitness_siglip   embedding of the fitness (tournament size 3,
                                                                30 generations)

Standard configuration: population 60, SDXL with 30 steps, uniform crossover (rate 0.9), Gaussian mutation of
10 % of the values with standard deviation 0.2 (probability 0.05), elitism 1. Every run records novelty and
diversity in BLIP-2, CLIP, DINOv2 and SigLIP 2 space (see ``experiments.common.embedding_spaces``).

    python -m experiments.fitness_embedding.create_jobs --db new_jobs.db
    FITNESS_EMBEDDING_DB_FILE=new_jobs.db python -m experiments.fitness_embedding.worker <job id>

``data/jobs/fitness_embedding.db`` lists the analysed runs.
"""
