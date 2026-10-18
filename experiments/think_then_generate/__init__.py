"""Think-Then-Generate (T2G, arXiv 2601.10332) followed by seed optimisation (Supplementary Fig. 10).

1. ``rewrite_prompts.py`` lets the T2G language model rewrite the prompts (reasoning + revised prompt).
2. ``worker.py`` runs the standard GA with SDXL and the rewritten prompt (``--condition t2g``) or the
   original prompt (``--condition original``).

    python -m experiments.think_then_generate.rewrite_prompts --out <output dir>
    python -m experiments.think_then_generate.worker --prompt "a cat" --seed 1943746817 --generations 100

Results:
``<THINK_THEN_GENERATE_RESULT_DIR>/sdxl_<condition>_g<generations>/<prompt_slug>/<prompt_slug>_<seed>_<timestamp>/``.
"""
