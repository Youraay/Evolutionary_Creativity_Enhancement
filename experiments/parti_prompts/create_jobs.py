"""Create the job database of the PartiPrompts experiment: every prompt of the selection x every seed.

    python -m experiments.parti_prompts.create_jobs
    python -m experiments.parti_prompts.create_jobs --append --prompts "a red cube on top of a blue cube"
"""
import argparse
import csv
from pathlib import Path

from experiments.common.settings import SEEDS
from experiments.parti_prompts.config import DB_FILE, PROMPT_FILE, job_queue


def read_prompts(path: Path) -> list[dict]:
    """Read a prompt table with the columns ``Prompt``, ``Category``, ``Challenge`` and ``Note``.

    Args:
        path: Tab-separated file.

    Returns:
        One dict per row.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DB_FILE, help="SQLite job database (default: PARTI_PROMPTS_DB_FILE)")
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_FILE, help="prompt table (TSV)")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="only these prompts of the prompt table (exact text)")
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    parser.add_argument("--append", action="store_true", help="add jobs to an existing database")
    return parser.parse_args()


def main() -> None:
    """Create the jobs in prompt order, three seeds per prompt."""
    args = parse_args()
    if args.db.exists() and not args.append:
        raise SystemExit(f"{args.db} already exists. Choose another --db or use --append.")

    rows = read_prompts(args.prompt_file)
    if args.prompts:
        missing = set(args.prompts) - {row["Prompt"] for row in rows}
        if missing:
            raise SystemExit(f"Not in {args.prompt_file}: {sorted(missing)}")
        rows = [row for row in rows if row["Prompt"] in args.prompts]

    jobs = [{"prompt": row["Prompt"], "seed": seed, "category": row.get("Category"),
             "challenge": row.get("Challenge"), "note": row.get("Note") or ""}
            for row in rows for seed in args.seeds]
    queue = job_queue(args.db)
    queue.create_table()
    print(f"{queue.add_jobs(jobs)} jobs written to {args.db} ({len(rows)} prompts x {len(args.seeds)} seeds)")
    print(f"Start with: PARTI_PROMPTS_DB_FILE={args.db} python -m experiments.parti_prompts.dispatch")


if __name__ == "__main__":
    main()
