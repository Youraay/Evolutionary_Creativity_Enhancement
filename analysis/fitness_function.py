"""Fitness function: distance to the population centroid vs. k nearest neighbours (Supplementary Fig. 4).

Output in ``plots/fitness_function/`` (see :mod:`analysis.common.comparison`): ``runs`` (centroid and 10
nearest neighbours as in the manuscript, plus 50 and all 99 neighbours), ``overview`` and ``tests.csv``.

    python -m analysis.fitness_function
"""
import argparse
from pathlib import Path

from analysis.common.comparison import render_comparison
from analysis.common.style import set_preview_dir


def render() -> None:
    """All figures and tests of the fitness function comparison."""
    render_comparison("fitness_function",
                      {"Centroid": "Centroid", "kNN_10": "10 Nearest Neighbors", "kNN_50": "50 Nearest Neighbors",
                       "kNN_100": "99 Nearest Neighbors"},
                      overview={"Centroid": "Centroid based Fitness Function",
                                "kNN_10": "10 Nearest-Neighbor based Fitness Function"},
                      overview_seeds=["Seed 4", "Seed 5", "Seed 6"], ylims=(0.0, 0.32))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render()


if __name__ == "__main__":
    main()
