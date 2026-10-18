"""Render the six-question post-hoc figure from portable prompt summaries.

This renderer neither rerates images nor reconstructs prompt means from raw scores.
Each dot is a supplied source-prompt mean; all prompts receive equal visual and
overall-mean weight. The accompanying statistical analysis is kept separate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common.gemma import style

GENERATIONS = (0, 10, 20, 30)
STEM = "ratings_across_generations"
EXPECTED_PROMPTS = {
    "study-suitability": 108,
    "study-unusualness": 114,
    "study-boredom": 108,
    "study-pleasantness": 108,
    "study-excitement": 108,
    "study-creativity": 114,
}
EXPECTED_INVALID = {
    "study-suitability": 48,
    "study-unusualness": 0,
    "study-boredom": 1,
    "study-pleasantness": 0,
    "study-excitement": 1,
    "study-creativity": 0,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validated_summaries(
    prompt_summary: pd.DataFrame, question_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt = prompt_summary.copy()
    overall = question_summary.copy()
    prompt_columns = {
        "prompt_id",
        "prompt_folder",
        "generation",
        "mean",
        "image_count",
        "expected_image_count",
        "invalid_image_count",
        "run_count",
    }
    overall_columns = {
        "prompt_id",
        "generation",
        "source_prompt_count",
        "valid_rating_count",
        "expected_rating_count",
        "invalid_rating_count",
        "mean_of_source_prompt_means",
    }
    _require(prompt_columns <= set(prompt), "Incomplete source-prompt summary columns")
    _require(overall_columns <= set(overall), "Incomplete question summary columns")
    _require(prompt[list(prompt_columns)].notna().all().all(), "Missing prompt summary values")
    _require(overall[list(overall_columns)].notna().all().all(), "Missing question summary values")
    _require(set(prompt.prompt_id) == set(EXPECTED_PROMPTS), "Unexpected rating questions")
    _require(set(overall.prompt_id) == set(EXPECTED_PROMPTS), "Unexpected summary questions")
    _require(
        not prompt.duplicated(["prompt_id", "prompt_folder", "generation"]).any(),
        "Duplicate source-prompt/generation records",
    )
    _require(
        len(overall) == 24 and not overall.duplicated(["prompt_id", "generation"]).any(),
        "Expected 24 unique question/generation summary cells",
    )
    for frame, names in (
        (
            prompt,
            (
                "generation",
                "image_count",
                "expected_image_count",
                "invalid_image_count",
                "run_count",
            ),
        ),
        (
            overall,
            (
                "generation",
                "source_prompt_count",
                "valid_rating_count",
                "expected_rating_count",
                "invalid_rating_count",
            ),
        ),
    ):
        for name in names:
            values = pd.to_numeric(frame[name], errors="raise")
            _require(
                np.isfinite(values).all() and values.eq(np.floor(values)).all(),
                f"Non-integer or non-finite counts: {name}",
            )
            frame[name] = values.astype("int64")
    prompt["mean"] = pd.to_numeric(prompt["mean"], errors="raise")
    overall["mean_of_source_prompt_means"] = pd.to_numeric(
        overall["mean_of_source_prompt_means"], errors="raise"
    )
    _require(prompt["mean"].between(1, 5).all(), "Prompt means outside the 1–5 scale")
    _require(prompt.run_count.eq(3).all(), "Expected three seed runs per prompt")
    _require(prompt.expected_image_count.eq(180).all(), "Expected 180 images per prompt/generation")
    _require(
        prompt.image_count.gt(0).all()
        and prompt.invalid_image_count.ge(0).all()
        and (prompt.image_count + prompt.invalid_image_count).eq(180).all(),
        "Invalid image coverage or missing/failed observations",
    )
    for question, count in EXPECTED_PROMPTS.items():
        subset = prompt[prompt.prompt_id.eq(question)]
        _require(
            len(subset) == count * 4
            and subset.prompt_folder.nunique() == count
            and subset.groupby("prompt_folder").size().eq(4).all()
            and set(subset.generation) == set(GENERATIONS)
            and subset.groupby("generation").size().eq(count).all(),
            f"Incomplete prompt cohort: {question}",
        )
        _require(
            int(subset.invalid_image_count.sum()) == EXPECTED_INVALID[question],
            f"Changed historical invalid-score count: {question}",
        )
        question_rows = overall[overall.prompt_id.eq(question)]
        _require(
            set(question_rows.generation) == set(GENERATIONS),
            f"Incomplete question-summary generations: {question}",
        )
        for generation in GENERATIONS:
            values = subset[subset.generation.eq(generation)]
            saved = question_rows[question_rows.generation.eq(generation)].iloc[0]
            for name, expected in (
                ("source_prompt_count", count),
                ("valid_rating_count", int(values.image_count.sum())),
                ("expected_rating_count", count * 180),
                ("invalid_rating_count", int(values.invalid_image_count.sum())),
            ):
                _require(
                    int(saved[name]) == expected,
                    f"Question summary mismatch: {question}/{generation}/{name}",
                )
            _require(
                np.isclose(saved.mean_of_source_prompt_means, values["mean"].mean(), rtol=0, atol=2e-14),
                f"Question mean differs from equally weighted prompt means: {question}/{generation}",
            )
    _require(
        int(overall.expected_rating_count.sum()) == 475200
        and int(overall.valid_rating_count.sum()) == 475150
        and int(overall.invalid_rating_count.sum()) == 50,
        "Unexpected total rating coverage",
    )
    return prompt, overall


def render(prompt_summary: pd.DataFrame, question_summary: pd.DataFrame, output: Path) -> str:
    """Save SVG/PNG figures; return the output stem without its extension."""
    prompt, overall = _validated_summaries(prompt_summary, question_summary)
    questions = tuple(style.RATING_QUESTIONS)
    _require(
        tuple(question for question, _, _ in questions) == tuple(EXPECTED_PROMPTS),
        "Shared style must preserve the six-question panel order",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(style.FIGURE_WIDTH, 7.5), sharey=True)
    rng = np.random.default_rng(20261014)
    for panel, (question, label, color) in enumerate(questions):
        ax = axes.flat[panel]
        subset = prompt[prompt.prompt_id.eq(question)]
        pivot = subset.pivot(index="prompt_folder", columns="generation", values="mean")
        for values in pivot[list(GENERATIONS)].to_numpy():
            ax.plot(range(4), values, color="#777777", alpha=0.07, linewidth=0.6, zorder=1)
        for position, generation in enumerate(GENERATIONS):
            values = subset.loc[subset.generation.eq(generation), "mean"].to_numpy(float)
            if np.ptp(values) > 1e-12:
                violin = ax.violinplot(
                    [values],
                    positions=[position],
                    widths=0.8,
                    showextrema=False,
                    bw_method="scott",
                    points=150,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.20)
            ax.scatter(
                position + rng.uniform(-0.13, 0.13, len(values)),
                values,
                s=8,
                color=color,
                alpha=0.50,
                linewidths=0,
                zorder=3,
            )
            ax.plot(
                [position - 0.17, position + 0.17],
                [values.mean()] * 2,
                color="#2f2f2f",
                linewidth=1.5,
                zorder=4,
            )
        counts = overall[overall.prompt_id.eq(question)]
        valid = int(counts.valid_rating_count.sum())
        expected = int(counts.expected_rating_count.sum())
        style.style_axis(ax)
        ax.set_title(
            f"{label} ({subset.prompt_folder.nunique()} prompts)\n{valid:,}/{expected:,} valid ratings",
            fontsize=style.TITLE_SIZE,
        )
        ax.set_xticks(range(4), labels=GENERATIONS)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(0.85, 5.15)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Generation")
        if panel % 3 == 0:
            ax.set_ylabel("Gemma rating (1–5)")
    fig.suptitle("Gemma ratings across generations", fontsize=15, y=0.99)
    fig.text(
        0.5,
        0.02,
        "Dots: source-prompt means (3 runs × 60 images expected); lines pair prompts.\n"
        "Horizontal marks equally weight the prompts in each panel.\n"
        "Coverage differs by question. Invalid scores are missing; statistical tests are supplied separately.\n"
        "Higher boredom scores mean more boring.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95), h_pad=1.5, w_pad=1.3)
    style.save_figure(fig, output, STEM)
    plt.close(fig)
    return STEM
