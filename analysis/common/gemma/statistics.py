"""One-sided paired prompt tests; Holm adjustment across six questions."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

QUESTIONS = (
    ("study-suitability", "Suitability", 108),
    ("study-unusualness", "Unusualness", 114),
    ("study-boredom", "Boredom", 108),
    ("study-pleasantness", "Pleasantness", 108),
    ("study-excitement", "Excitement", 108),
    ("study-creativity", "Creativity", 114),
)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if not (np.isfinite(pvalues).all() and ((pvalues >= 0) & (pvalues <= 1)).all()):
        raise ValueError("Invalid p-values")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1))
    )
    return adjusted


def paired_test(differences):
    # Avoid spurious rank distinctions caused by CSV/subtraction roundoff.
    differences = np.round(np.asarray(differences, dtype=float), 12)
    if not np.isfinite(differences).all() or len(differences) == 0:
        raise ValueError("Need finite paired differences")
    nonzero = differences[differences != 0]
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    if len(nonzero):
        result = wilcoxon(
            differences,
            alternative="greater",
            zero_method="wilcox",
            correction=False,
            method="asymptotic",
            nan_policy="raise",
        )
        assert np.isclose(result.statistic, positive)
        pvalue = float(result.pvalue)
        sign_p = float(
            binomtest(int((nonzero > 0).sum()), len(nonzero), p=0.5, alternative="greater").pvalue
        )
    else:
        pvalue = sign_p = 1.0
    return {
        "n_pairs": len(differences),
        "n_increased": int((differences > 0).sum()),
        "n_decreased": int((differences < 0).sum()),
        "n_unchanged": int((differences == 0).sum()),
        "W_positive": positive,
        "W_negative": negative,
        "matched_rank_biserial": (positive - negative) / (positive + negative)
        if positive + negative
        else 0.0,
        "p_one_sided": pvalue,
        "sign_test_p_one_sided": sign_p,
    }


def analyze(summary, complete_only=False):
    if summary.duplicated(["prompt_id", "prompt_folder", "generation"]).any():
        raise ValueError("Duplicate prompt/generation records")
    if set(summary.prompt_id) != {q for q, _, _ in QUESTIONS}:
        raise ValueError("Unexpected questions")
    results, pairs = [], []
    for question, label, expected in QUESTIONS:
        group = summary[summary.prompt_id.eq(question)]
        if set(group.generation) != {0, 10, 20, 30}:
            raise ValueError("Unexpected generations")
        if len(group) != expected * 4 or not group.groupby("generation").size().eq(expected).all():
            raise ValueError("Incomplete prompt cohort")
        if not (group.run_count.eq(3).all() and group.expected_image_count.eq(180).all()):
            raise ValueError("Unexpected population design")
        if not (group.image_count + group.invalid_image_count).eq(180).all():
            raise ValueError("Invalid observation counts")
        if not group["mean"].between(1, 5).all():
            raise ValueError("Invalid means")
        endpoints = []
        for gen in (0, 30):
            endpoint = (
                group[group.generation.eq(gen)][
                    ["prompt_folder", "mean", "image_count", "invalid_image_count"]
                ]
                .set_index("prompt_folder")
                .add_suffix(f"_g{gen}")
            )
            endpoints.append(endpoint)
        paired = endpoints[0].join(endpoints[1], how="outer").sort_index()
        if paired.isna().any().any():
            raise ValueError("Unpaired source prompts")
        if complete_only:
            paired = paired[
                paired.invalid_image_count_g0.eq(0) & paired.invalid_image_count_g30.eq(0)
            ].copy()
        paired["difference_g30_minus_g0"] = paired.mean_g30 - paired.mean_g0
        paired["difference_used_for_ranks"] = paired.difference_g30_minus_g0.round(12)
        paired.insert(0, "prompt_id", question)
        paired.insert(1, "rating", label)
        pairs.append(paired.reset_index())
        row = {
            "prompt_id": question,
            "rating": label,
            "mean_generation_0": float(paired.mean_g0.mean()),
            "mean_generation_30": float(paired.mean_g30.mean()),
            "mean_paired_change": float(paired.difference_g30_minus_g0.mean()),
            "median_paired_change": float(paired.difference_g30_minus_g0.median()),
            "invalid_scores_generation_0": int(paired.invalid_image_count_g0.sum()),
            "invalid_scores_generation_30": int(paired.invalid_image_count_g30.sum()),
        }
        row.update(paired_test(paired.difference_g30_minus_g0))
        results.append(row)
    results = pd.DataFrame(results)
    results["p_holm_six_tests"] = holm(results.p_one_sided)
    results["sign_test_p_holm_six_tests"] = holm(results.sign_test_p_one_sided)
    results["significant_higher_alpha_0_05"] = results.p_holm_six_tests.lt(0.05)
    results["significance"] = results.p_holm_six_tests.map(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    )
    return results, pd.concat(pairs, ignore_index=True)
