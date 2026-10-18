"""Operator ablation: the GA without crossover, without elitism or without mutation (not part of the manuscript).

Output in ``plots/operator_ablation/`` (see :mod:`analysis.common.comparison`): ``runs`` and ``tests.csv``.

    python -m analysis.operator_ablation
"""
import argparse
from pathlib import Path

from analysis.common.comparison import render_comparison
from analysis.common.style import set_preview_dir


def render() -> None:
    """All figures and tests of the operator ablation."""
    render_comparison("operator_ablation", {"No_Crossover": "No crossover", "No_Elites": "No elitism",
                                            "No_Mutation": "No mutation"}, ylims=(0.0, 0.4))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    args = parser.parse_args()
    set_preview_dir(args.preview)
    render()


if __name__ == "__main__":
    main()
