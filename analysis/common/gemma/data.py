"""Load hash-checked figure inputs and reconcile the saved scientific summaries."""

import hashlib
from pathlib import Path

import pandas as pd

from analysis.common.gemma import aggregation, evolution, ratings_plot, statistics
from analysis.common.paths import DATA_DIR

DATA = DATA_DIR / "results/gemma"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconcile(saved, calculated, keys, label):
    require(not saved.duplicated(keys).any(), f"Repeated saved {label}")
    require(not calculated.duplicated(keys).any(), f"Repeated calculated {label}")
    a = saved.set_index(keys).sort_index()
    b = calculated.set_index(keys).sort_index()
    require(a.index.equals(b.index), f"Mismatched {label} identities")
    require(set(a.columns) == set(b.columns), f"Mismatched {label} fields")
    pd.testing.assert_frame_equal(
        a, b[a.columns], check_dtype=False, check_exact=False, rtol=0, atol=2e-12
    )


def load_posthoc(root=DATA, deep=False):
    source = root / "posthoc"
    prompt = pd.read_csv(source / "source_prompt_generation_summary.csv")
    question = pd.read_csv(source / "question_generation_summary.csv")
    ratings_plot._validated_summaries(prompt, question)
    # The primary and sensitivity tests also validate the full prompt cohort.
    tests, paired = statistics.analyze(prompt)
    sensitivity, _ = statistics.analyze(prompt, complete_only=True)
    stats_source = root / "statistics"
    reconcile(
        pd.read_csv(stats_source / "generation_0_vs_30_tests.csv"),
        tests,
        ["prompt_id"],
        "directional tests",
    )
    reconcile(
        pd.read_csv(stats_source / "paired_prompt_values.csv"),
        paired,
        ["prompt_id", "prompt_folder"],
        "paired prompt values",
    )
    reconcile(
        pd.read_csv(stats_source / "complete_endpoint_sensitivity.csv"),
        sensitivity,
        ["prompt_id"],
        "complete-endpoint sensitivity",
    )
    if deep:
        raw = pd.read_csv(source / "ratings.csv.gz", dtype={"content_sha256": str})
        require(len(raw) == 475200, "Unexpected number of image-question observations")
        require(not raw.duplicated(["prompt_id", "image_id"]).any(), "Duplicate rating")
        require(raw.status.isin(["valid", "parse_invalid"]).all(), "Missing/failed inference")
        require(int(raw.status.eq("parse_invalid").sum()) == 50, "Invalid count changed")
        require(
            raw.loc[raw.status.eq("parse_invalid"), "score"].isna().all(),
            "Invalid scores must be missing",
        )
        require(raw.loc[raw.status.eq("valid"), "score"].between(1, 5).all(), "Invalid score")
        require(raw.image_id.nunique() == 82080, "Image cohort changed")
        run, derived_prompt, derived_question = aggregation.summaries(raw)
        reconcile(
            prompt,
            derived_prompt,
            ["prompt_id", "prompt_folder", "generation"],
            "source prompt summaries",
        )
        reconcile(question, derived_question, ["prompt_id", "generation"], "question summaries")
        reconcile(
            pd.read_csv(source / "run_generation_summary.csv"),
            run,
            ["prompt_id", "prompt_folder", "experiment_id", "generation"],
            "run summaries",
        )
    return prompt, question, tests, paired, sensitivity


def load_evolution(root=DATA):
    source = root / "evolution"
    runs = pd.read_csv(source / "run_generation_summary.csv")
    evolution.validate_runs(runs)
    prompt_curves = evolution.summarize_runs(runs)
    seed_means, target_curves = evolution.pool_prompts(runs)
    reconcile(
        pd.read_csv(source / "pooled_generation_summary.csv"),
        prompt_curves,
        ["construct", "prompt", "generation"],
        "per-prompt seed summaries",
    )
    reconcile(
        pd.read_csv(source / "prompt_pooled_seed_means.csv"),
        seed_means,
        ["construct", "prompt", "seed", "generation"],
        "pooled seed means",
    )
    reconcile(
        pd.read_csv(source / "prompt_pooled_curves.csv"),
        target_curves,
        ["construct", "prompt", "generation"],
        "pooled curves",
    )
    examples = pd.read_csv(source / "selected_examples.csv")
    require(len(examples) == 9 and set(examples.construct) == {"creativity"}, "Example scope")
    expected = {(p, s) for p in evolution.PROMPTS for s in (2025, 2026, 2027)}
    require(set(zip(examples.prompt, examples.seed, strict=True)) == expected, "Example identities")
    for record in examples.itertuples(index=False):
        path = (root / record.original_image).resolve()
        require(path.is_relative_to(root.resolve()), "Unsafe example path")
        require(sha256(path) == record.content_sha256, "Example image changed")
        expected_gen = 29 if record.seed == 2025 and record.prompt in {"a cat", "boredom"} else 30
        require(record.generation == expected_gen, "Incorrect example generation")
        require(record.saved_fitness == 5 and record.rank == 1, "Unexpected selected example")
    return prompt_curves, target_curves, examples
