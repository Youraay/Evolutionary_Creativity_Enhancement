"""Model comparison: the standard GA with 14 text-to-image models on "a cat", "a chair" and "boredom".

    python -m experiments.model_comparison.create_jobs --prompts "a cat" "a chair" boredom --models all
    python -m experiments.model_comparison.dispatch

Long-running models can also be processed in fixed per-GPU queues inside one Slurm job
(``submit_gpu_queue.sh``), which saves QOS job slots.

Results: ``<MODEL_COMPARISON_RESULT_DIR>/<model>/<prompt_slug>/<prompt_slug>_<seed>_<timestamp>/``.
"""
