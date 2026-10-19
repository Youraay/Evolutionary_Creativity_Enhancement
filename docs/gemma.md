# VLM evaluation and rating-guided evolution

The VLM analyses compare automated ratings with human ratings, evaluate images
from distance-based optimisation, and use image ratings as evolutionary fitness.
The human-rating comparison includes Gemma 4, LLaVA-OneVision and Qwen2.5-VL.
Gemma is used for the broader image evaluation and for rating-guided evolution.

`analysis.gemma_ratings` generates the six-dimensional ratings plot.
`analysis.gemma_evolution` generates the VLM comparison and evolution panels of
Supplementary Figure 14. Their inputs are in `data/results/gemma/`; derived
tables and figures are written to the corresponding `plots/<module>/` directory.

## Reproducing the analyses

Install the repository's analysis dependencies and run the modules from the
repository root:

```bash
pip install -e ".[analysis]"
python -m analysis.gemma_ratings --deep --verify-only
python -m analysis.gemma_evolution --verify-only
python -m analysis --only gemma_ratings gemma_evolution --preview previews
```

The modules use the common plotting theme, prompt colours and PDF/SVG save
helper. `--preview DIR` adds PNG previews, `--output DIR` overrides a module's
output directory, and `PLOTS_DIR` sets the common plot root. `--verify-only`
checks the inputs without writing outputs. `--deep` also rebuilds post-hoc
rating summaries from the individual scores. Rendering reads the included data
and does not require a GPU or model downloads.

| Analysis | Inputs under `data/results/gemma/` | Outputs |
| --- | --- | --- |
| `analysis/gemma_ratings.py` | `posthoc/`, `statistics/` | `plots/gemma_ratings/ratings_across_generations.pdf`, `.svg`, `.csv`, `_tests.csv`, `_pairs.csv`, `_sensitivity.csv` |
| `analysis/gemma_evolution.py` | `benchmark/`, `evolution/`, `examples/` | `plots/gemma_evolution/evaluation_and_evolution.pdf`, `.svg`, `human_vlm_correlations.csv`, `prompt_curves.csv`, `target_curves.csv` |

The CSV files are recalculated from the source tables. Each output directory
also contains `render_manifest.json` with software versions and input, code and
output hashes. Shared calculations are in `analysis/common/gemma/`: aggregation
of image ratings, paired statistical tests, correlations and pooling across
seeds. The figure-to-panel mapping is listed in the [README](../README.md).

The analysis was checked with Python 3.12, NumPy 2.5.2, pandas 2.3.3, SciPy 1.18.1,
Matplotlib 3.11.1, Pillow 11.3.0 and seaborn 0.13.2. These plotting dependencies
are separate from the historical inference environment described below.

## Comparison with human ratings

The benchmark contains 36 study images from cat, chair and boredom at generations
0, 15 and 30. Each model rates the same images on suitability, creativity,
unusualness, boredom, pleasantness and excitement. Human ratings are rescaled
to 1-5 and averaged per image. The analysis unit is the image.

| Model | Hugging Face model | Valid ratings / expected |
| --- | --- | --- |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` | 216 / 216 |
| LLaVA-OneVision 7B | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | 212 / 216 |
| Qwen2.5-VL 7B | `Qwen/Qwen2.5-VL-7B-Instruct` | 216 / 216 |

The preserved [question configuration](../data/results/gemma/methods/study_question_configuration.yaml)
defines all three models and the exact six questions. All models used BF16,
greedy decoding, a maximum of 64 new tokens, seed 2025 and one response per
image/question. Suitability includes the image's generation prompt; the other
five questions use the image alone. The original runtime was Python 3.12.14,
Torch 2.11.0+cu128, Transformers 5.16.1 and Accelerate 1.14.0 on an A100 80GB GPU.

The complete response export is
[`benchmark/latest_responses.csv`](../data/results/gemma/benchmark/latest_responses.csv).
It contains all 648 responses, including exact rendered questions, raw model
outputs, parsed scores, parse errors and runtime provenance. Resolved model
revisions are recorded for each response:

| Model | Resolved revision |
| --- | --- |
| Gemma 4 | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| LLaVA-OneVision | `0d50680527681998e456c7b78950205bedd8a068` |
| Qwen2.5-VL | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |

Four LLaVA unusualness responses fall outside the allowed scale and remain
invalid. Their exclusion leaves 644 valid pairs in
[`benchmark/headline_analysis_units.csv`](../data/results/gemma/benchmark/headline_analysis_units.csv).
All 18 Spearman correlations in the heatmap are recalculated from these pairs:
17 cells have 36 images and LLaVA unusualness has 32. The original `metrics.csv`
also records Pearson correlations, bootstrap intervals, prediction errors and
coverage. Completeness and dataset metadata are stored alongside the tables.

This benchmark supplies panel A of `evaluation_and_evolution.pdf` and `.svg`.
Its 36 images are a separate dataset from the broader post-hoc evaluation below;
adding source prompts to that evaluation does not enlarge the human benchmark.
Individual participant records are not included here.

## Ratings across generations

`ratings_across_generations.pdf` and `.svg` show Gemma ratings of images produced
by distance-based optimisation. Generations 0, 10, 20 and 30 are included, with
three runs and 60 image observations per generation for each source prompt.
Repeated elite images remain observations in their populations.

The included data record two completed stages: all six questions were first
applied to 108 source prompts, then six additional prompts were rated for
creativity and unusualness. Consequently, creativity and unusualness currently
cover 114 prompts, while suitability, boredom, pleasantness and excitement cover
108. A cluster workflow has been prepared for the remaining four questions on
the six added prompts (17,280 additional ratings). After those results have been
verified, the source tables, figure and statistical tests can be updated to the
same 114-prompt cohort for all dimensions.

The current snapshot contains 475,200 expected scores, 475,150 valid scores and
50 parse-invalid scores retained as missing values. Each dot is a source-prompt
mean across valid image observations; faint lines connect the same prompt across
generations. Violins describe the distribution of prompt means, and horizontal
marks give their equally weighted mean. All prompts use circular markers.

The separate exploratory tests ask whether generation 30 is higher than
generation 0. They use paired source-prompt means, one-sided Wilcoxon signed-rank
tests and Holm correction across six dimensions. In the included snapshot, only
unusualness passes the corrected 0.05 threshold. Creativity has raw p=0.0201 and
adjusted p=0.1003. These results apply to the current cohorts and must be
recomputed when the remaining ratings are added. The figure itself is descriptive
and does not display significance stars.

The tests treat source prompts as independent units and assume reasonably
symmetric paired differences. `n.s.` denotes no demonstrated increase, rather
than equivalence or absence of a decrease. Exact sign-test and complete-endpoint
sensitivity results are supplied separately. Individual images are not treated
as independent replications, and these tests are not analyses of human
participants.

The question IDs preserve the original wording: `study-unusualness` asks
"How unusual do you find the image?" and corresponds to human uncommonness;
`study-pleasantness` corresponds to pleasure/affect and `study-excitement` to
arousal. The full scale instructions and suitability wording are in the linked
question configuration.

## Rating-guided evolution

The remaining panels of `evaluation_and_evolution.pdf` and `.svg` use Gemma
ratings as fitness. The supplied population metrics cover seven targets
(creativity, unusualness, uncommonness, innovation, novelty, originality and
uniqueness), three image prompts (cat, chair and boredom) and three seeds
(2025, 2026 and 2027): 63 trajectories with populations of 100 images.
Unusualness and uncommonness are distinct fitness questions in these runs.

Generated panels B-D show creativity fitness, BLIP2 divergence and BLIP2 diversity
for each prompt. Lines show the mean across seed runs; shaded bands show the
sample standard deviation. Panel E shows the best-rated final-population image
from each prompt and seed. The creativity/seed-2025 cat and boredom runs end at
generation 29, and their example labels retain g29. At generation 30 those two curves have two
seeds; chair retains three.

Generated panels F-H compare the seven fitness targets. Cat, chair and boredom
are first averaged with equal weights within each seed and generation; the seed
averages then determine the mean and sample SD. The incomplete
creativity/seed-2025/generation-30 block is excluded because only chair is
available. Generation 29 retains its original index. Fitness is the score from
each target's own question. The SD bands describe variability across seeds and
are not confidence intervals or significance tests.

BLIP2 divergence and diversity use half-cosine distances. The encoder identity,
formulae and relation to the manuscript equation are given in
[metric definitions](../data/results/gemma/methods/metric_definitions.txt).
The six additional fitness question strings are preserved in
[`fitness_construct_questions.py`](../data/results/gemma/methods/fitness_construct_questions.py);
the creativity question follows the study creativity wording.

## Data and original inference

The repository includes all individual scores needed for the post-hoc ratings
analysis, the full three-model response export and paired human image aggregates
for the benchmark, and the population metrics and nine original JPEG examples
used in the evolution figure. The JPEGs are embedded at their original
resolution in the SVG. Source hashes are checked before calculations.

The VLM inference implementation is maintained in
[`pdaech/vlm-creativity`](https://github.com/pdaech/vlm-creativity), under
`src/creativity_eval/`; the benchmark records the producing revision
`7a7d7a283ab181d669ae32b013e322f4e2648968`. The preserved YAML records the original
questions and settings; its historical data paths are not configured for a new
inference run in this repository. Replotting uses the included response, score
and population tables. Recomputing BLIP2 metrics or rerunning evolution also
requires the original image/embedding/noise archives and inference environment.

[`source_inventory.json`](../data/results/gemma/methods/source_inventory.json)
records source files, archive members and hashes. Its `package_file` paths are
relative to `data/results/gemma/`; `original_package_file` fields retain earlier
export paths. These historical paths document provenance and are not required
at runtime. `manifest.json` contains the input hashes used by the analysis.
