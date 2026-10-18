"""Scientific and repository-integration regressions for the two Gemma figures."""

import base64
import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from analysis.common import style as common_style
from analysis.common.gemma import data, evolution, ratings_plot, statistics, style, workflow
from analysis.common.paths import REPO_ROOT
from PIL import Image


def test_source_hashes_and_posthoc_values():
    workflow.check_hashes()
    prompt, overall, result, paired, sensitivity = data.load_posthoc(deep=True)
    ratings_plot._validated_summaries(prompt, overall)
    assert len(prompt) == 2640 and len(overall) == 24 and len(paired) == 660
    assert result.loc[result.significant_higher_alpha_0_05, "rating"].tolist() == ["Unusualness"]
    assert sensitivity.loc[sensitivity.rating.eq("Suitability"), "n_pairs"].item() == 107
    with pytest.raises(ValueError):
        ratings_plot._validated_summaries(prompt.iloc[:-1], overall)


def test_holm_and_direction():
    np.testing.assert_allclose(statistics.holm([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])
    assert statistics.paired_test([0] * 20)["p_one_sided"] == 1
    assert statistics.paired_test(np.arange(1, 31))["p_one_sided"] < 0.001
    assert statistics.paired_test(-np.arange(1, 31))["p_one_sided"] > 0.99


def test_curves_keep_actual_generations_and_balanced_seed_blocks():
    prompt, pooled, examples = data.load_evolution()
    assert len(prompt) == 651 and len(pooled) == 217 and len(examples) == 9
    cell = pooled[pooled.construct.eq("creativity") & pooled.generation.eq(30)].iloc[0]
    assert cell.n_seeds == 2 and cell.seeds == "2026;2027"
    assert examples.generation.eq(29).sum() == 2
    runs = pd.read_csv(data.DATA / "evolution/run_generation_summary.csv")
    with pytest.raises(ValueError):
        evolution.validate_runs(pd.concat([runs.iloc[:-1], runs.iloc[:1]]))


def test_benchmark_coverage_and_recalculated_correlations():
    metrics = pd.read_csv(data.DATA / "benchmark/metrics.csv")
    pairs = pd.read_csv(data.DATA / "benchmark/headline_analysis_units.csv")
    result = evolution.validate_benchmark(metrics, pairs)
    assert len(result) == 18 and result.n_analysis_units.sum() == 644
    cell = result[
        result.model_id.eq("llava-onevision-7b") & result.prompt_id.eq("study-unusualness")
    ].iloc[0]
    assert cell.n_analysis_units == 32
    bad = metrics.copy()
    bad.loc[0, "spearman_r"] = 0
    with pytest.raises(ValueError):
        evolution.validate_benchmark(bad, pairs)


def test_source_tampering_is_detected(tmp_path):
    (tmp_path / "manifest.json").write_bytes((data.DATA / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="Hash mismatch"):
        workflow.check_hashes(tmp_path, {"benchmark"})


def test_shared_style_and_launcher_registration():
    from analysis.__main__ import MODULES

    assert {"gemma_ratings", "gemma_evolution"} <= set(MODULES)
    assert style.PROMPT_COLORS == {p: common_style.COLOR_MAP[p] for p in evolution.PROMPTS}
    assert style.TITLE_SIZE == common_style.TITLE_SIZE


def test_launcher_outputs_and_original_examples(tmp_path):
    import os

    env = dict(os.environ, PLOTS_DIR=str(tmp_path / "plots"), MPLBACKEND="Agg")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis",
            "--only",
            "gemma_ratings",
            "gemma_evolution",
            "--preview",
            str(tmp_path / "previews"),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    ns = "{http://www.w3.org/2000/svg}"
    for module, stem in (
        ("gemma_ratings", "ratings_across_generations"),
        ("gemma_evolution", "evaluation_and_evolution"),
    ):
        root = tmp_path / "plots" / module
        assert (root / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        assert (tmp_path / "previews" / module / f"{stem}.png").is_file()
        svg = ET.parse(root / f"{stem}.svg").getroot()
        assert len(list(svg.iter(ns + "text"))) > 30
        images = list(svg.iter(ns + "image"))
        if module == "gemma_ratings":
            assert not images
            continue
        assert len(images) == 9
        examples = pd.read_csv(data.DATA / "evolution/selected_examples.csv")
        records = [
            record
            for prompt in evolution.PROMPTS
            for record in examples[examples.prompt.eq(prompt)]
            .sort_values("seed")
            .itertuples(index=False)
        ]
        for image, record in zip(images, records, strict=True):
            prefix, payload = image.attrib["{http://www.w3.org/1999/xlink}href"].split(",", 1)
            assert prefix == "data:image/jpeg;base64"
            assert hashlib.sha256(base64.b64decode(payload)).hexdigest() == record.content_sha256
            with Image.open(data.DATA / record.original_image) as original:
                assert original.size == (1024, 1024)
