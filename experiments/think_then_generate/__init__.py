"""Think-Then-Generate (T2G, arXiv 2601.10332) followed by seed optimisation.

1. ``rewrite_prompts.py`` lets the T2G language model rewrite the prompts (reasoning + revised prompt),
   measures run time and memory of the T2G DiT and renders example images.
2. ``worker.py`` runs the standard GA with the rewritten prompt (``--condition t2g``) or the original
   prompt (``--condition original``), using SDXL (variant B) or the T2G DiT (variant A) as generator.

    sbatch experiments/think_then_generate/submit_rewrite.sh <output dir>
    sbatch experiments/think_then_generate/submit_sdxl.sh "a cat" 1943746817 100
    sbatch experiments/think_then_generate/submit_t2g_dit.sh "a cat" 1943746817 t2g

Results: ``<THINK_THEN_GENERATE_RESULT_DIR>/<sdxl|t2gdit>_<condition>_g<generations>/<prompt_slug>/<prompt_slug>_<seed>_<timestamp>/``.
"""
