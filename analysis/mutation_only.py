"""Mutation only: mutated copies without crossover and elitism (Supplementary Fig. 9).

Output (``plots/mutation_only/``):

    examples_generation_8          Supplementary Fig. 9A: one random candidate of generation 8 of every run with
                                   Gaussian mutation (fixed random seed, not a selection)
    distance_to_initial_mean       Supplementary Fig. 9B: distance to the mean of generation 0, Gaussian and
                                   uniform mutation, the three runs pooled; bracket = Mann-Whitney U test of
                                   generation 0 vs. 30
    tests.csv                      tests of distance and diversity per variant, Holm-corrected

    python -m analysis.mutation_only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common.paths import PLOTS_DIR, archive
from analysis.common.plotting import image_grid
from analysis.common.results import load_images
from analysis.common.runs import read_run, variant_runs
from analysis.common.statistics import group_tests
from analysis.common.style import COLOR_MAP, reset_style, set_preview_dir
from analysis.common.violins import render_pooled_violins

OUT_DIR = PLOTS_DIR / "mutation_only"
VARIANTS = {"gaussean_only": "Gaussian noise", "uniform_only": "Uniform noise"}


def render() -> None:
    """Example images, pooled violins and tests."""
    reset_style()
    color = COLOR_MAP["a cat"]
    runs = {name: variant_runs("mutation_only", name, color) for name in VARIANTS}
    render_pooled_violins(OUT_DIR / "distance_to_initial_mean",
                          [{"title": title, "configs": runs[name], "color": color} for name, title in VARIANTS.items()],
                          nrows=2)
    groups = {title: pd.concat([read_run(c) for c in runs[name]], ignore_index=True)
              for name, title in VARIANTS.items()}
    group_tests(groups, ["global_score", "diversity_score"]).to_csv(OUT_DIR / "tests.csv", index=False)

    rng = np.random.default_rng(0)
    images = []
    for config in runs["gaussean_only"]:
        candidates = read_run(config).query("generation == 8")["file_name"].sort_values().to_list()
        images += load_images(archive(config["path"], "mutation_only"), [str(rng.choice(candidates))], size=400)
    image_grid([images], [""], [f"Run {k + 1}" for k in range(len(images))],
               "Gaussian mutation only, generation 8", OUT_DIR / "examples_generation_8")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render()


if __name__ == "__main__":
    main()
