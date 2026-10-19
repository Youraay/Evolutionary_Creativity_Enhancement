# VLM evaluation and rating-guided evolution

The VLM analyses address three questions: how model ratings agree with human
ratings, how ratings change during distance-based optimisation, and how images
evolve when a model rating is used as fitness.

## Models and rating questions

The human-rating comparison uses three vision-language models:

| Model | Hugging Face model | Resolved revision |
| --- | --- | --- |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| LLaVA-OneVision 7B | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | `0d50680527681998e456c7b78950205bedd8a068` |
| Qwen2.5-VL 7B | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |

All models use BF16 weights, greedy decoding, at most 64 output tokens, seed 2025
and one response per image and question. They return a continuous score from
1 to 5 as JSON. Higher scores indicate more of the rated quality.

| Dimension | Question |
| --- | --- |
| Suitability | How appropriate do you find the image for the prompt "{study_prompt}"? |
| Creativity | How creative do you find the image? |
| Unusualness | How unusual do you find the image? |
| Boredom | How boring do you find the image? |
| Pleasantness | How pleasant do you find the image? |
| Excitement | How exciting do you find the image? |

The [question configuration](../data/results/gemma/methods/study_question_configuration.yaml)
contains the full instructions, scale definitions and model settings.
Suitability includes the generation prompt; the other five questions use the
image alone. Unusualness corresponds to the human uncommonness rating,
pleasantness to pleasure/affect, and excitement to arousal.

## Agreement with human ratings

Each model rates the same 36 study images: cat, chair and boredom at generations
0, 15 and 30. Human ratings are rescaled to 1–5 and averaged per image.
Spearman correlations compare these image averages with the model scores for
each of the six dimensions.

The benchmark contains 648 responses. Four LLaVA unusualness responses are
outside the allowed scale, leaving 644 valid pairs: 216 for Gemma, 216 for Qwen
and 212 for LLaVA. The LLaVA unusualness correlation uses 32 images; each of the
other 17 correlations uses 36.

The [response export](../data/results/gemma/benchmark/latest_responses.csv)
contains rendered questions, raw outputs, parsed scores, parse errors and
runtime provenance. The
[paired values](../data/results/gemma/benchmark/headline_analysis_units.csv)
are the inputs to the correlation calculation. Individual human participant
records are not included.

`analysis.gemma_evolution` recalculates all 18 Spearman correlations and plots
them in panel A of `evaluation_and_evolution.pdf` and `.svg`
(Supplementary Figure 14A). The
[benchmark metrics](../data/results/gemma/benchmark/metrics.csv) also include
Pearson correlations, bootstrap intervals, prediction errors and valid counts.

## Ratings across generations

Gemma rates distance-optimised images on the same six dimensions at generations
0, 10, 20 and 30. Each source prompt has three runs and 60 image observations per
generation. Repeated elite images remain part of their populations.

`analysis.gemma_ratings` produces `ratings_across_generations.pdf` and `.svg`.
Each dot is a source-prompt mean over valid image ratings, and lines connect
the same prompt across generations. Violins show the distribution of prompt
means; horizontal marks show their equally weighted mean. Sample sizes, valid
counts and missing counts are recorded per dimension and generation in the
[generation summary](../data/results/gemma/posthoc/question_generation_summary.csv).
Invalid responses remain missing.

The generation-0/30 comparison uses a one-sided Wilcoxon signed-rank test on
paired source-prompt means, with Holm correction across six dimensions.
The alternative hypothesis is that generation 30 has higher ratings.
Source prompts are treated as independent units, and the signed-rank test
assumes approximately symmetric paired differences. These are exploratory tests.

The [test results](../data/results/gemma/statistics/generation_0_vs_30_tests.csv)
report sample sizes, changes and adjusted p-values. Paired values and
sign-test/complete-endpoint sensitivity results are supplied in `statistics/`.
The descriptive figure has no significance stars.

## Rating-guided evolution

Gemma provides fitness scores for seven separate conditions: creativity,
unusualness, uncommonness, innovation, novelty, originality and uniqueness.
The analysis covers cat, chair and boredom, seeds 2025, 2026 and 2027, and
populations of 100 images: 63 trajectories in total. Each condition uses its
own image-only question; unusualness and uncommonness are distinct conditions.
Exact fitness questions are in the
[study configuration](../data/results/gemma/methods/study_question_configuration.yaml)
and [fitness question definitions](../data/results/gemma/methods/fitness_construct_questions.py).

The three population measures are:

- **Fitness:** mean Gemma rating within the population.
- **BLIP2 divergence:** mean half-cosine distance from each image embedding to
  the fixed generation-0 centroid of the same run.
- **BLIP2 diversity:** mean half-cosine distance between distinct image pairs
  within the population. Self-pairs are excluded; repeated elites are retained.

Half-cosine distance is `0.5 × (1 − cosine similarity)`. The encoder is
`Salesforce/blip2-opt-2.7b` at revision
`59a1ef6c1e5117b3f65523d1c6066825bcf315e3`, using the 1,408-dimensional vision
pooler output. Full definitions are in
[metric definitions](../data/results/gemma/methods/metric_definitions.txt).

`analysis.gemma_evolution` plots creativity-target fitness, divergence and
diversity by image prompt in generated panels B–D, and one best-rated final
image per prompt and seed in panel E. Lines show means across seeds; shaded
bands show sample SD.

Generated panels F–H compare all seven targets. The three image prompts receive
equal weights within each seed and generation, followed by mean and sample SD
across seeds. Fitness values are scores from each target's own question.

The creativity/seed-2025 cat and boredom trajectories end at generation 29.
Their examples retain g29, and their generation-30 prompt curves use two seeds.
The corresponding generation-30 block is excluded from prompt-pooled curves
because it contains only chair. Other complete blocks use all three seeds.

## Reproducing the figures

Install the analysis dependencies and run from the repository root:

```bash
pip install -e ".[analysis]"
python -m analysis.gemma_ratings --deep --verify-only
python -m analysis.gemma_evolution --verify-only
python -m analysis --only gemma_ratings gemma_evolution --preview previews
```

| Module | Inputs under `data/results/gemma/` | Outputs |
| --- | --- | --- |
| `analysis.gemma_ratings` | `posthoc/`, `statistics/` | `plots/gemma_ratings/ratings_across_generations.pdf`, `.svg`, `.csv`, `_tests.csv`, `_pairs.csv`, `_sensitivity.csv` |
| `analysis.gemma_evolution` | `benchmark/`, `evolution/`, `examples/` | `plots/gemma_evolution/evaluation_and_evolution.pdf`, `.svg`, `human_vlm_correlations.csv`, `prompt_curves.csv`, `target_curves.csv` |

The modules use the shared plotting theme and PDF/SVG save helper.
`--preview DIR` adds PNG previews, `--output DIR` overrides the plot directory,
and `PLOTS_DIR` sets the common plot root. `--verify-only` validates inputs
without writing outputs. `--deep` rebuilds generation-rating summaries from
individual scores. The figure-to-panel mapping is in the [README](../README.md).

Figures are generated from the included scores, human image averages and
population metrics. This requires no GPU inference. Recomputing image embeddings
or rerunning evolution requires the image, embedding and noise archives.
Nine original JPEG examples are included and embedded at full resolution in
the evolution SVG.

The VLM scoring implementation is in
[`pdaech/vlm-creativity`](https://github.com/pdaech/vlm-creativity/tree/7a7d7a283ab181d669ae32b013e322f4e2648968/src/creativity_eval).
Benchmark inference used Python 3.12.14, Torch 2.11.0+cu128, Transformers 5.16.1
and Accelerate 1.14.0 on an A100 80GB GPU. The configuration YAML contains the
scoring templates; its paths refer to the inference dataset.

`manifest.json` records input hashes,
[`source_inventory.json`](../data/results/gemma/methods/source_inventory.json)
records source files and archive members, and each output directory contains
`render_manifest.json` with software versions and input, code and output hashes.
