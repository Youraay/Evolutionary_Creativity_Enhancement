"""Analysis of the GA results: all figures and tables (written to ``plots/<module>/``).

One module per experiment (each runnable with ``python -m analysis.<module>``):

    example_prompts        Fig. 1E, 2A-C, 2E-F; Supplementary Figs. 1A-C, 5A, 11B
    parti_prompts          Fig. 2D, 2G; Supplementary Figs. 2, 5B
    seed_transfer          Fig. 3; Supplementary Fig. 10
    model_comparison       Supplementary Fig. 3A-B
    fitness_embedding      Supplementary Fig. 3C-D
    fitness_function       Supplementary Fig. 4
    archive_size           Supplementary Fig. 6
    crossover_function     Supplementary Fig. 7
    tournament_size        Supplementary Fig. 8
    mutation_only          Supplementary Fig. 9
    long_runs              Supplementary Fig. 1D
    noise_distribution     Supplementary Fig. 11A (and the same figure for the long runs and the PartiPrompts)
    pca_parti_prompts      PCA of initial noise and BLIP-2 embeddings of the PartiPrompts
    operator_ablation      not part of the manuscript
    think_then_generate    not part of the manuscript
    gemma_ratings          Gemma ratings across generations (114/108 source prompts)
    gemma_evolution        human/VLM agreement and Gemma-guided image evolution

``python -m analysis`` runs all of them. Shared code is in ``common``: paths, style, loading of runs, statistical
tests and the plot types used by several figures. Error bands are the standard error of the mean over runs,
except gemma_evolution, which retains the sample standard deviation of its original analysis.
"""
