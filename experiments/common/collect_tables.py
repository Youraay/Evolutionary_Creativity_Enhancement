"""Copy the candidate tables and run configurations of an experiment into the repository.

For every finished run of the job database (status ``DONE``; without a database: every run directory below
the result directory) its CSV and JSON files (candidate table and run configuration) are copied from the result
directory to ``data/results/<experiment>`` with the same relative path, together with the CSV files directly
in the result directory (e.g. summaries). Archives and images stay in the result directory.

    python -m experiments.common.collect_tables model_comparison
    python -m experiments.common.collect_tables model_comparison --db /path/to/live_jobs.db   # also copies the database
"""
import argparse
import shutil
import sqlite3
from pathlib import Path

from experiments.common.settings import db_file, result_dir, table_dir


def run_directories(results: Path, db: Path) -> list[Path]:
    """Run directories of an experiment.

    Args:
        results: Result directory of the experiment.
        db: Job database; if it does not exist, every directory with ``<name>/<name>.csv`` counts.

    Returns:
        Resolved run directories in sorted order.

    Raises:
        ValueError: If a finished job points outside the result directory.
    """
    if not db.exists():
        return sorted({csv.parent.resolve() for csv in results.rglob("*.csv") if csv.stem == csv.parent.name})
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    paths = [Path(p).resolve() for (p,) in connection.execute(
        "SELECT output_path FROM jobs WHERE status = 'DONE' ORDER BY id")]
    connection.close()
    outside = [p for p in paths if not p.is_relative_to(results.resolve())]
    if outside:
        raise ValueError(f"{len(outside)} runs outside {results}, e.g. {outside[0]}")
    missing = [p for p in paths if not (p / f"{p.name}.csv").exists()]
    for path in missing:
        print(f"[WARN] no candidate table in {path}")
    return sorted(p for p in paths if p not in missing)


def collect(experiment: str, source_db: Path | None = None) -> None:
    """Copy the tables of one experiment (files that are unchanged are skipped).

    Args:
        experiment: Package name below ``experiments``.
        source_db: Job database to copy to ``data/jobs/<experiment>.db`` first (e.g. the live database of
            running jobs).
    """
    results, tables, db = result_dir(experiment).resolve(), table_dir(experiment), db_file(experiment)
    if source_db is not None:
        source = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        target = sqlite3.connect(db)
        source.backup(target)
        source.close()
        target.close()
        print(f"{source_db} -> {db}")
    runs = run_directories(results, db)
    files = [f for run in runs for f in sorted(run.glob("*")) if f.suffix in (".csv", ".json")]
    files += sorted(results.glob("*.csv"))
    copied = 0
    for source in files:
        target = tables / source.relative_to(results)
        if target.exists() and target.stat().st_size == source.stat().st_size \
                and target.stat().st_mtime == source.stat().st_mtime:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    print(f"{experiment}: {len(runs)} runs, {len(files)} files, {copied} copied to {tables}")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", help="package names below experiments/")
    parser.add_argument("--db", type=Path, help="job database to copy into data/jobs first (one experiment only)")
    args = parser.parse_args()
    if args.db and len(args.experiments) > 1:
        parser.error("--db works with a single experiment")
    for experiment in args.experiments:
        collect(experiment, args.db)


if __name__ == "__main__":
    main()
