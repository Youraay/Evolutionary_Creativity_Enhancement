"""Model comparison: the standard GA with 14 text-to-image models on "a cat", "a chair" and "boredom".

    python -m experiments.model_comparison.create_jobs --prompts "a cat" "a chair" boredom --models all
    python -m experiments.model_comparison.worker <job id>

Results: ``<MODEL_COMPARISON_RESULT_DIR>/<model>/<prompt_slug>/<prompt_slug>_<seed>_<timestamp>/``.
"""
