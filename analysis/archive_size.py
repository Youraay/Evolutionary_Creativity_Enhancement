"""Archive size: novelty fitness with an archive of earlier candidates (Supplementary Fig. 6).

Output in ``plots/archive_size/`` (see :mod:`analysis.common.comparison`): ``runs`` with the rows of the
manuscript (archive of 200 candidates with threshold 0, no archive, archive of all 3000 candidates of a run;
the run used an archive size of 6000, which holds every candidate), ``overview`` and ``tests.csv``.

    python -m analysis.archive_size
"""
import argparse
from pathlib import Path

from analysis.common.comparison import render_comparison
from analysis.common.style import set_preview_dir


def render() -> None:
    """All figures and tests of the archive size comparison."""
    render_comparison("archive_size",
                      {"Archiv_Low": "Archive size: 200", "No_Archiv": "Archive size: 0",
                       "Full_Archiv": "Archive size: 3000"},
                      overview={"No_Archiv": "Archive size: 0", "Archiv_Low": "Archive size: 200",
                                "Full_Archiv": "Archive size: 3000"},
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
