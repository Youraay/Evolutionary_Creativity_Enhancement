"""Seed transfer: noise optimised for one prompt rendered with another prompt (Fig. 3, Supplementary Fig. 10).

Output (``plots/seed_transfer/``):

    <source>_to_<target>_distance     Fig. 3B (a chair -> a cat), Supplementary Fig. 10C (a dog -> a cat):
                                      distance to the mean of generation 0, transfer vs. direct optimisation
    <source>_to_<target>_diversity    Fig. 3C, Supplementary Fig. 10D: diversity within the generation
    *_tests.csv                       Mann-Whitney U tests of generation 0 vs. 30 of both curves
    examples_<source>_to_<target>     Fig. 3A, Supplementary Fig. 10A: images of generations 0-30 of a run and the
                                      same candidates rendered with the target prompt (``data/figure_images.json``)

    python -m analysis.seed_transfer
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.common.paths import DATA_DIR, PLOTS_DIR
from analysis.common.runs import transfer_runs
from analysis.common.style import reset_style, set_preview_dir
from analysis.common.transfer_plots import render_transfer_curves, render_transfer_examples

OUT_DIR = PLOTS_DIR / "seed_transfer"
CURVES = [("a chair", "a cat"), ("a dog", "a cat")]
"""Transfers shown as curves (Fig. 3B-C, Supplementary Fig. 10C-D)."""
METRICS = {
    "distance": ("global_score", "distance to mean of\ninitial generation [a.u.]",
                 "Transferability of divergence\nfrom initial conditions", (0.0, 0.2)),
    "diversity": ("diversity_score", "diversity within\ngeneration [a.u.]",
                  "Transferability of diversity\nwithin generations", (0.10, 0.25)),
}


def _slug(prompt: str) -> str:
    """File name part of a prompt."""
    return prompt.replace(" ", "_")


def render() -> None:
    """Curves, tests and example images of all transfers of the manuscript."""
    tests = []
    for source, target in CURVES:
        configs = transfer_runs([source], [target])
        for key, (metric, ylabel, title, limits) in METRICS.items():
            reset_style()
            tests.append(render_transfer_curves(OUT_DIR / f"{_slug(source)}_to_{_slug(target)}_{key}", configs, metric,
                                                ylabel, title, yticks=limits, ylims=limits))
    pd.concat(tests).to_csv(OUT_DIR / "tests.csv", index=False)
    selection = json.loads((DATA_DIR / "figure_images.json").read_text())["seed_transfer"]
    for pair, chosen in selection.items():
        source, target = pair.split(" -> ")
        reset_style()
        render_transfer_examples(OUT_DIR / f"examples_{_slug(source)}_to_{_slug(target)}", source, target, chosen)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render()


if __name__ == "__main__":
    main()
