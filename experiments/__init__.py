"""Job databases and workers of all experiments.

Every experiment follows the same pattern:

1. ``create_jobs`` writes one row per run (prompt x seed x variant) into an SQLite job database.
2. ``worker <job id>`` runs the genetic algorithm of one row on a GPU and marks the job as done or failed; the
   jobs can be started by hand or by any batch system.

Experiments (see the README for the figures of the manuscript they belong to):
    example_prompts      SDXL on twelve example prompts, population 100.
    parti_prompts        SDXL on the PartiPrompts selection (114 prompts).
    seed_transfer        Noise optimised for one prompt rendered with another prompt.
    model_comparison     The standard GA with 14 text-to-image models.
    fitness_embedding    Novelty measured with BLIP-2, CLIP, DINOv2 or SigLIP 2.
    fitness_function     Distance to the population centroid vs. k nearest neighbours.
    archive_size         Novelty fitness with an archive of earlier candidates.
    crossover_function   n-point, uniform and arithmetic crossover.
    tournament_size      Selection tournaments of 2, 3, 5 or 10 candidates.
    mutation_only        Gaussian or uniform mutation without crossover and elitism.
    long_runs            300 generations.
    operator_ablation    The GA without crossover, elitism or mutation.
    think_then_generate  Prompt rewriting with Think-Then-Generate, followed by seed optimisation.
    mutation_grid_search Mutated fraction and strength of the Gaussian mutation (Methods).

Shared code is in ``common``: settings, job database, the standard GA (``ga``) and the two families
of configurable experiments (``variants``: population 100, operators per job; ``embedding_spaces``:
population 60, metrics in four embedding spaces).
"""
