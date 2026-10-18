"""Portable checks and derivations for the saved evolution and benchmark tables.

Each population contributes one row, not 100 independent image observations.
Prompt curves use mean and sample SD across seeds. Seven-target comparisons first
average cat/chair/boredom equally within each seed, then summarize those seed
averages. The incomplete creativity/2025 generation-30 block is excluded only from
that pooled view; generation 29 is never relabeled as generation 30.

The benchmark compares VLM scores with per-image human aggregates, not individual
participants. Its four invalid LLaVA unusualness responses stay omitted. Saved
BLIP2 divergence/diversity values can be redrawn, but this module does not refit
embeddings or claim to recompute those metrics from raw features: full embedding
matrices are not included in the compact figure package.
"""

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TARGETS = (
    "creativity",
    "unusualness",
    "uncommonness",
    "innovation",
    "novelty",
    "originality",
    "uniqueness",
)
PROMPTS = ("a cat", "a chair", "boredom")
METRICS = ("fitness_mean", "divergence_mean", "diversity")
SEEDS = (2025, 2026, 2027)
POOLED_PROMPT = "cat_chair_boredom_equal_weight"
BENCHMARK_MODELS = ("gemma4-26b-a4b", "llava-onevision-7b", "qwen2.5-vl-7b")
BENCHMARK_QUESTIONS = (
    "study-boredom",
    "study-creativity",
    "study-excitement",
    "study-pleasantness",
    "study-suitability",
    "study-unusualness",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _columns(frame, names, label):
    _require(isinstance(frame, pd.DataFrame), f"{label} must be a pandas DataFrame")
    _require(frame.columns.is_unique, f"{label} has duplicate column names")
    missing = set(names) - set(frame.columns)
    _require(not missing, f"{label} is missing columns: {sorted(missing)}")


def _numeric(frame, names, *, integral=()):
    result = frame.copy()
    for column in names:
        try:
            values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric values in {column}") from error
        _require(np.isfinite(values).all(), f"Non-finite values in {column}")
        if column in integral:
            _require(np.equal(values, np.floor(values)).all(), f"Non-integral {column}")
            result[column] = values.astype(np.int64)
        else:
            result[column] = values
    return result


def _last_generation(target, prompt, seed):
    return 29 if target == "creativity" and seed == 2025 and prompt != "a chair" else 30


def validate_runs(runs: pd.DataFrame):
    """Reject incomplete, duplicated, out-of-scope, or numerically invalid runs.

    The accepted scope is exactly seven targets, three prompts and three seeds:
    63 trajectories / 1,951 complete 100-member populations. This validates the
    supplied table without mutating it; no filesystem or cluster access occurs.
    """
    keys = ["construct", "prompt", "seed", "generation"]
    _columns(runs, keys + ["run_name", "population", *METRICS], "Run table")
    _require(len(runs) == 1951, "Expected exactly 1,951 population rows")
    table = _numeric(
        runs,
        ["seed", "generation", "population", *METRICS],
        integral=("seed", "generation", "population"),
    )
    _require(not table[keys].isna().any().any(), "Missing population identity")
    _require(not table.duplicated(keys).any(), "Duplicate population identity")
    expected = {
        (target, prompt, seed, generation)
        for target, prompt, seed in product(TARGETS, PROMPTS, SEEDS)
        for generation in range(_last_generation(target, prompt, seed) + 1)
    }
    observed = set(table[keys].itertuples(index=False, name=None))
    _require(observed == expected, "Run identities or generation coverage differ")
    _require(table["population"].eq(100).all(), "Expected complete populations of 100")
    _require(table["run_name"].notna().all(), "Missing run name")
    _require(
        table["run_name"].map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
        "Empty or invalid run name",
    )
    names = table.groupby(["construct", "prompt", "seed"])["run_name"].nunique()
    _require(len(names) == 63 and names.eq(1).all(), "Mixed run names within a trajectory")
    _require(table["run_name"].nunique() == 63, "Run name reused across trajectories")
    for metric in METRICS:
        lower, upper = (1, 5) if metric == "fitness_mean" else (0, 1)
        _require(table[metric].between(lower, upper).all(), f"Out-of-range {metric}")


def _summaries(table):
    rows = []
    for (target, prompt, generation), group in table.groupby(
        ["construct", "prompt", "generation"], sort=True
    ):
        group = group.sort_values("seed")
        _require(group["seed"].is_unique, "Repeated seed within a summary cell")
        result = {
            "construct": target,
            "prompt": prompt,
            "generation": int(generation),
            "n_seeds": len(group),
            "seeds": ";".join(str(int(seed)) for seed in group["seed"]),
            "images": int(group["population"].sum()),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(result)
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Return per-target/prompt/generation seed means and sample SDs.

    Every seed has equal weight. Cat/boredom creativity at generation 30 retain
    the two available seeds; chair retains all three. Returns 651 summary cells.
    """
    validate_runs(runs)
    table = _numeric(runs, ["seed", "generation", "population", *METRICS])
    return _summaries(table)


def pool_prompts(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return equal-prompt seed means and their between-seed mean/sample SD.

    These match the saved ``prompt_pooled_seed_means.csv`` (650 rows) and
    ``prompt_pooled_curves.csv`` (217 rows). Only creativity/2025/g30 is excluded:
    its chair population exists, but the matching cat and boredom populations do
    not. No interpolation, imputation, or generation-29 substitution is applied.
    """
    validate_runs(runs)
    table = _numeric(
        runs,
        ["seed", "generation", "population", *METRICS],
        integral=("seed", "generation", "population"),
    )
    seed_rows, excluded = [], []
    for (target, seed, generation), group in table.groupby(
        ["construct", "seed", "generation"], sort=True
    ):
        prompts = set(group["prompt"])
        if prompts != set(PROMPTS):
            _require(
                (target, seed, generation) == ("creativity", 2025, 30)
                and prompts == {"a chair"}
                and len(group) == 1,
                "Unexpected incomplete three-prompt seed block",
            )
            excluded.append((target, seed, generation))
            continue
        _require(len(group) == 3, "Duplicate prompt in pooled seed block")
        row = {
            "construct": target,
            "prompt": POOLED_PROMPT,
            "seed": int(seed),
            "generation": int(generation),
            "population": int(group["population"].sum()),
        }
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        seed_rows.append(row)
    _require(excluded == [("creativity", 2025, 30)], "Expected one incomplete seed block")
    seed_means = pd.DataFrame(seed_rows)
    curves = _summaries(seed_means)
    _require(len(seed_means) == 650 and len(curves) == 217, "Pooled coverage differs")
    return seed_means, curves


def validate_benchmark(metrics: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Recompute all 18 Spearman correlations and check the saved coefficients.

    The 644 paired observations are distinct images within each model/question.
    LLaVA unusualness has 32 valid pairs; the other 17 cells each have 36. The
    provided human values are per-image aggregates, not participant-level data.
    Returns model/question identities, Spearman coefficients and valid pair counts.
    """
    keys = ["model_id", "prompt_id"]
    _columns(metrics, keys + ["spearman_r", "n_analysis_units"], "Benchmark metrics")
    _columns(units, keys + ["image_id", "vlm_score", "human_rating"], "Benchmark pairs")
    _require(len(metrics) == 18 and len(units) == 644, "Benchmark row counts differ")
    _require(not metrics[keys].isna().any().any(), "Missing benchmark metric identity")
    _require(not metrics.duplicated(keys).any(), "Duplicate benchmark metric cell")
    identity = keys + ["image_id"]
    _require(not units[identity].isna().any().any(), "Missing benchmark pair identity")
    _require(not units.duplicated(identity).any(), "Duplicate benchmark image pair")
    _require(
        units["image_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
        "Empty or invalid benchmark image identity",
    )
    if "analysis_unit_id" in units:
        _require(units["analysis_unit_id"].notna().all(), "Missing benchmark analysis unit")
        _require(
            not units.duplicated(keys + ["analysis_unit_id"]).any(),
            "Duplicate benchmark analysis unit",
        )
    expected = set(product(BENCHMARK_MODELS, BENCHMARK_QUESTIONS))
    _require(
        set(metrics[keys].itertuples(index=False, name=None)) == expected,
        "Benchmark model/question scope differs",
    )
    _require(
        set(units[keys].itertuples(index=False, name=None)) == expected,
        "Benchmark pairs have unexpected model/question scope",
    )
    values = _numeric(units, ["vlm_score", "human_rating"])
    saved = _numeric(metrics, ["spearman_r", "n_analysis_units"], integral=("n_analysis_units",))
    _require(saved["spearman_r"].between(-1, 1).all(), "Invalid saved Spearman coefficient")
    for score in ("vlm_score", "human_rating"):
        _require(values[score].between(1, 5).all(), f"Out-of-range benchmark {score}")
    saved = saved.set_index(keys)
    results = []
    for (model, question), group in values.groupby(keys, sort=True):
        expected_count = 32 if (model, question) == ("llava-onevision-7b", "study-unusualness") else 36
        _require(len(group) == expected_count, "Benchmark valid-pair coverage differs")
        _require(
            int(saved.loc[(model, question), "n_analysis_units"]) == expected_count,
            "Saved benchmark pair count differs",
        )
        rho = float(spearmanr(group["vlm_score"], group["human_rating"]).statistic)
        _require(np.isfinite(rho), "Undefined benchmark correlation")
        _require(
            np.isclose(rho, float(saved.loc[(model, question), "spearman_r"]), rtol=0, atol=1e-12),
            f"Saved Spearman coefficient differs for {model}/{question}",
        )
        results.append(
            {
                "model_id": model,
                "prompt_id": question,
                "spearman_r": rho,
                "n_analysis_units": len(group),
            }
        )
    return pd.DataFrame(results)
