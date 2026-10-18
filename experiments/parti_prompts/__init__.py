"""Main experiment: SDXL with the standard GA on the PartiPrompts selection (114 prompts x 3 seeds).

    python -m experiments.parti_prompts.create_jobs          # job database from the prompt selection
    python -m experiments.parti_prompts.dispatch             # submit all runs to Slurm

Results: ``<PARTI_PROMPTS_RESULT_DIR>/<prompt_slug>/<prompt_slug>_<seed>_<timestamp>/``.
"""
