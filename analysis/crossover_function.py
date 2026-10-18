"""Crossover function: n-point, uniform and arithmetic crossover (Supplementary Fig. 7).

Output in ``plots/crossover_function/`` (see :mod:`analysis.common.comparison`): ``runs`` (1-, 3-, 7-point and
uniform crossover as in the manuscript, plus arithmetic and per-pixel uniform crossover), ``overview`` (as
in the manuscript; the n-point rows pool the runs of the identical ``_Pixel`` variants) and ``tests.csv``.

    python -m analysis.crossover_function
"""
import argparse
from pathlib import Path

from analysis.common.comparison import render_comparison
from analysis.common.style import set_preview_dir


def render() -> None:
    """All figures and tests of the crossover comparison."""
    render_comparison("crossover_function",
                      {"1Point": "1-Point crossover", "3Point": "3-Point crossover", "7Point": "7-Point crossover",
                       "Uniform": "Uniform crossover", "Uniform_Pixel": "Uniform crossover per pixel",
                       "Arithmetic": "Arithmetic crossover"},
                      overview={"1Point": "1-Point Crossover", "3Point": "3-Point Crossover",
                                "7Point": "7-Point Crossover", "Uniform": "Uniform Crossover"},
                      overview_seeds=["Seed 1", "Seed 2", "Seed 3"])


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render()


if __name__ == "__main__":
    main()
