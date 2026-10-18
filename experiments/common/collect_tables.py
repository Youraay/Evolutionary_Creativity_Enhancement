"""Copy the candidate tables and run configurations of an experiment into the repository.

For every finished run of the job database (status ``DONE``; without a database: every run directory below
the result directory) its CSV and JSON files (candidate table and run configuration) are copied from the result
directory to ``data/results/<experiment>`` with the same relative path, together with the CSV files directly
in the result directory (e.g. summaries). Archives and images stay in the result directory.

The run configurations are cleaned on copying (:func:`clean_config`): the absolute paths ``result_path`` and
``stat_path`` become paths relative to the result directory of the experiment, and notes in German are
replaced by their translation (:data:`TRANSLATIONS`). The copies of the job databases in ``data/jobs`` are
cleaned likewise (:func:`clean_database`): ``output_path`` becomes relative and the user names in
``claimed_by`` are removed.

    python -m experiments.common.collect_tables model_comparison
    python -m experiments.common.collect_tables model_comparison --db /path/to/live_jobs.db   # also copies the database
    python -m experiments.common.collect_tables --clean archive_size        # clean the copies already in the repository
"""
import argparse
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path, PurePosixPath

from experiments.common.settings import DATA_DIR, db_file, result_dir, table_dir

TRANSLATIONS = {
    "src_t2g/test_t2g.py (System-Prompt und greedy Decoding wie inference.py des T2G-Repos), Slurm-Job 1793120":
        "experiments/think_then_generate/rewrite_prompts.py (system prompt and greedy decoding as in inference.py "
        "of the T2G repository; produced as src_t2g/test_t2g.py), Slurm job 1793120",
}
"""Notes in German in run configurations and their translation."""

ABSOLUTE_PATH = re.compile(r'"/(work|home|panfs|tmp)/')


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
    paths = [(results / p if not Path(p).is_absolute() else Path(p)).resolve() for (p,) in connection.execute(
        "SELECT output_path FROM jobs WHERE status = 'DONE' ORDER BY id")]
    connection.close()
    outside = [p for p in paths if not p.is_relative_to(results.resolve())]
    if outside:
        raise ValueError(f"{len(outside)} runs outside {results}, e.g. {outside[0]}")
    missing = [p for p in paths if not (p / f"{p.name}.csv").exists()]
    for path in missing:
        print(f"[WARN] no candidate table in {path}")
    return sorted(p for p in paths if p not in missing)


def clean_config(text: str, run: Path) -> str:
    """Run configuration without absolute paths and with notes in English.

    Only the affected values are replaced, so formatting and all other values stay as written by the run.

    Args:
        text: Content of the JSON file.
        run: Run directory relative to the result directory of the experiment.

    Returns:
        The cleaned content.

    Raises:
        ValueError: If an absolute path remains.
    """
    config = json.loads(text)
    replacements = dict(TRANSLATIONS)
    if isinstance(config.get("result_path"), str):
        replacements[config["result_path"]] = run.as_posix()
    if isinstance(config.get("stat_path"), str):
        replacements[config["stat_path"]] = (PurePosixPath(run.as_posix()) / PurePosixPath(config["stat_path"]).name
                                             ).as_posix()
    for old, new in replacements.items():
        text = text.replace(json.dumps(old), json.dumps(new))
    if ABSOLUTE_PATH.search(text):
        raise ValueError(f"absolute path left in the configuration of {run}")
    return text


def write_config(source: Path, target: Path, run: Path) -> bool:
    """Write the cleaned configuration ``source`` to ``target`` (with the timestamp of ``source``).

    Args:
        source: Run configuration in the result directory (or the copy itself, to clean it in place).
        target: Copy in ``data/results``.
        run: Run directory relative to the result directory of the experiment.

    Returns:
        True if ``target`` was written, False if it was already up to date.
    """
    text, status = clean_config(source.read_text(), run), source.stat()
    if target.exists() and target.read_text() == text and target.stat().st_mtime == status.st_mtime:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    os.utime(target, (status.st_atime, status.st_mtime))
    return True


def clean_database(db: Path, results: Path) -> int:
    """Remove user names and absolute paths from the copy of a job database in ``data/jobs``.

    ``claimed_by`` (user or worker that claimed the job) is cleared and ``output_path`` becomes the run
    directory relative to the result directory. The file is rebuilt afterwards (``VACUUM``), so that no old
    values remain in free pages.

    Args:
        db: Job database in ``data/jobs``.
        results: Result directory of the experiment.

    Returns:
        Number of changed jobs.

    Raises:
        ValueError: If ``db`` is not in ``data/jobs`` or an output path lies outside ``results``.
    """
    if not db.resolve().is_relative_to((DATA_DIR / "jobs").resolve()):
        raise ValueError(f"{db} is not a copy in {DATA_DIR / 'jobs'}")
    connection = sqlite3.connect(db)
    changed = 0
    rows = connection.execute("SELECT id, output_path, claimed_by FROM jobs").fetchall()
    for job_id, output_path, claimed_by in rows:
        relative = output_path
        if output_path and Path(output_path).is_absolute():
            path = Path(output_path).resolve()
            if not path.is_relative_to(results.resolve()):
                raise ValueError(f"{db}: job {job_id} outside {results}: {output_path}")
            relative = path.relative_to(results.resolve()).as_posix()
        if relative != output_path or claimed_by is not None:
            connection.execute("UPDATE jobs SET output_path = ?, claimed_by = NULL WHERE id = ?", (relative, job_id))
            changed += 1
    connection.commit()
    connection.execute("VACUUM")
    connection.close()
    return changed


def clean(experiment: str) -> None:
    """Clean the run configurations in ``data/results/<experiment>`` and the job database in ``data/jobs``.

    Args:
        experiment: Package name below ``experiments``.
    """
    tables, db = table_dir(experiment), db_file(experiment)
    configs = sorted(p for p in tables.rglob("*.json") if p.parent != tables)
    written = sum(write_config(path, path, path.parent.relative_to(tables)) for path in configs)
    jobs = clean_database(db, result_dir(experiment)) if db.exists() else 0
    print(f"{experiment}: {len(configs)} configurations, {written} cleaned; {jobs} jobs cleaned in {db.name}")


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
        print(f"{source_db} -> {db} ({clean_database(db, results)} jobs cleaned)")
    runs = run_directories(results, db)
    files = [f for run in runs for f in sorted(run.glob("*")) if f.suffix in (".csv", ".json")]
    files += sorted(results.glob("*.csv"))
    copied = 0
    for source in files:
        target = tables / source.relative_to(results)
        if source.suffix == ".json":
            copied += write_config(source, target, source.parent.relative_to(results))
            continue
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
    parser.add_argument("--clean", action="store_true", help="only clean the configurations already in data/results")
    args = parser.parse_args()
    if args.db and len(args.experiments) > 1:
        parser.error("--db works with a single experiment")
    for experiment in args.experiments:
        if args.clean:
            clean(experiment)
        else:
            collect(experiment, args.db)


if __name__ == "__main__":
    main()
