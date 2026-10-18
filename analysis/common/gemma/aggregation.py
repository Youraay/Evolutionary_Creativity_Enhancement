"""Aggregate valid image ratings without dropping repeated elites."""

import pandas as pd


def summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.assign(valid_content_sha256=frame["content_sha256"].where(frame["status"] == "valid"))
    run = (
        frame.groupby(
            [
                "prompt_id",
                "prompt_folder",
                "experiment_id",
                "experiment_seed",
                "generation",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            image_count=("score", "count"),
            expected_image_count=("score", "size"),
            invalid_image_count=("status", lambda values: values.eq("parse_invalid").sum()),
            mean=("score", "mean"),
            median=("score", "median"),
            sample_sd=("score", "std"),
            unique_image_count=("valid_content_sha256", "nunique"),
        )
        .sort_values(["prompt_id", "prompt_folder", "experiment_id", "generation"])
    )
    source_prompt = (
        frame.groupby(["prompt_id", "prompt_folder", "generation"], as_index=False)
        .agg(
            image_count=("score", "count"),
            expected_image_count=("score", "size"),
            invalid_image_count=("status", lambda values: values.eq("parse_invalid").sum()),
            run_count=("experiment_id", "nunique"),
            mean=("score", "mean"),
            median=("score", "median"),
            sample_sd=("score", "std"),
            unique_image_count=("valid_content_sha256", "nunique"),
        )
        .sort_values(["prompt_id", "prompt_folder", "generation"])
    )
    question = (
        source_prompt.groupby(["prompt_id", "generation"], as_index=False)
        .agg(
            source_prompt_count=("prompt_folder", "size"),
            valid_rating_count=("image_count", "sum"),
            expected_rating_count=("expected_image_count", "sum"),
            invalid_rating_count=("invalid_image_count", "sum"),
            mean_of_source_prompt_means=("mean", "mean"),
            median_of_source_prompt_means=("mean", "median"),
            sample_sd_of_source_prompt_means=("mean", "std"),
            minimum_source_prompt_mean=("mean", "min"),
            maximum_source_prompt_mean=("mean", "max"),
        )
        .sort_values(["prompt_id", "generation"])
    )
    return run, source_prompt, question
