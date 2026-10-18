"""Create the job database of the model comparison: every model x every prompt x every seed.

    python -m experiments.model_comparison.create_jobs --prompts "a cat" "a chair" boredom --models all
    python -m experiments.model_comparison.create_jobs --models sd15 ssd_1b --max-prompts 10
    python -m experiments.model_comparison.create_jobs --append --models z_image --prompts "a cat"
"""
import argparse
from pathlib import Path

from experiments.common.settings import SEEDS
from experiments.model_comparison.config import DB_FILE, MODEL_NAMES, PROMPT_FILE, job_queue
from experiments.parti_prompts.create_jobs import read_prompts

DEFAULT_MODELS = [m for m in MODEL_NAMES if m not in ("sdxl", "sdxl_turbo")]
"""SDXL already ran for all prompts in the PartiPrompts experiment and is not scheduled by default."""


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DB_FILE, help="SQLite job database (default: MODEL_COMPARISON_DB_FILE)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=MODEL_NAMES + ["all"],
                        help="'all' = all models including sdxl and sdxl_turbo")
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    parser.add_argument("--prompts", nargs="+", default=None, help="prompts given directly instead of --prompt-file")
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_FILE,
                        help="TSV with the columns Prompt, Category, Challenge, Note")
    parser.add_argument("--max-prompts", type=int, default=None, help="only the first N prompts (e.g. for a test)")
    parser.add_argument("--append", action="store_true", help="add jobs to an existing database")
    return parser.parse_args()


def main() -> None:
    """Create the jobs ordered by model, prompt and seed, so that models finish one after another."""
    args = parse_args()
    if args.db.exists() and not args.append:
        raise SystemExit(f"{args.db} already exists. Choose another --db or use --append.")

    models = MODEL_NAMES if "all" in args.models else args.models
    rows = [{"Prompt": p} for p in args.prompts] if args.prompts else read_prompts(args.prompt_file)
    rows = rows[:args.max_prompts]

    jobs = [{"prompt": row["Prompt"], "seed": seed, "model": model, "category": row.get("Category"),
             "challenge": row.get("Challenge"), "note": row.get("Note") or ""}
            for model in models for row in rows for seed in args.seeds]
    queue = job_queue(args.db)
    queue.create_table()
    print(f"{queue.add_jobs(jobs)} jobs written to {args.db} "
          f"({len(models)} models x {len(rows)} prompts x {len(args.seeds)} seeds)")
    print(f"Run a job with: MODEL_COMPARISON_DB_FILE={args.db} python -m experiments.model_comparison.worker <job id>")


if __name__ == "__main__":
    main()
