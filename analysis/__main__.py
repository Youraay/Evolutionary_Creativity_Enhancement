"""Run every analysis module (``python -m analysis``); the PCA of the PartiPrompts is only plotted.

    python -m analysis [--only example_prompts seed_transfer ...] [--preview DIR]
"""
import argparse
import importlib
import sys
from pathlib import Path

MODULES = ["example_prompts", "parti_prompts", "seed_transfer", "model_comparison", "fitness_embedding",
           "fitness_function", "archive_size", "crossover_function", "tournament_size", "mutation_only", "long_runs",
           "noise_distribution", "operator_ablation", "think_then_generate", "mutation_grid_search",
           "pca_parti_prompts"]


def main() -> None:
    """Call the command line entry point of every selected module."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=MODULES, default=MODULES)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    for name in args.only:
        print(f"[INFO] ===== analysis.{name}", flush=True)
        sys.argv = [name] + (["plot"] if name == "pca_parti_prompts" else []) + (
            ["--preview", str(args.preview)] if args.preview else [])
        importlib.import_module(f"analysis.{name}").main()


if __name__ == "__main__":
    main()
