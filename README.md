# Evolutionary Creativity Enhancement

A genetic algorithm that optimises the **initial noise** of text-to-image diffusion models towards novel
images. A population of noise tensors is rendered by a frozen text-to-image model (SDXL by default), every
image is embedded with BLIP-2, and the fitness of a candidate is its distance to the mean embedding of the
current population. Tournament selection, crossover and mutation act directly on the noise tensors.
We also compare VLM ratings with human image ratings and use Gemma ratings as an alternative fitness
function for the evolutionary algorithm.

This repository contains the evolutionary algorithm, experiment code, job databases, candidate tables and
figure analyses. The run archives (images, embeddings and initial noise of the candidates) are on
G-Node GIN, in two data repositories:

* [Evolutionary_Creativity_Enhancement_1](https://gin.g-node.org/yourisinziri/Evolutionary_Creativity_Enhancement_1):
  the runs with population 100 (`example_prompts`, `seed_transfer`, `archive_size`, `crossover_function`,
  `fitness_function`, `mutation_only`, `operator_ablation`)
* [Evolutionary_Creativity_Enhancement_2](https://gin.g-node.org/yourisinziri/Evolutionary_Creativity_Enhancement_2):
  the runs with population 60 (`parti_prompts`, `model_comparison`, `fitness_embedding`, `tournament_size`,
  `long_runs`, `think_then_generate`)

To reproduce the figures from them, point the result directories in `.env` to the data repositories (commented
block in `.env.example`).

## Repository layout

```
evolutionary_creativity/     the method
├── pipeline.py              GeneticAlgorithmPipeline (evaluate, evolve, save every generation)
├── candidate.py             one individual: noise, image, scores, lineage
├── noise_factory.py         initial population and candidate ids
├── operators/               tournament selection, crossover (uniform, n-point, arithmetic), mutation
├── evaluators/              novelty (distance to mean embedding, k nearest neighbours), diversity, CLIP score
└── models/                  text-to-image models (SDXL, UNet and DiT models of the model comparison)
                             and image embeddings (BLIP-2, CLIP, DINOv2, SigLIP 2)
experiments/                 one package per experiment: create_jobs -> worker
└── common/                  settings, SQLite job database, standard GA, configurable experiment families
                             (variants, embedding_spaces), collect_tables, data_release
analysis/                    one module per experiment, writes plots/<module>/
└── common/                  paths, style, loading of runs, statistical tests, shared plot types
data/
├── jobs/<experiment>.db     job database of every experiment: one row per analysed run with its configuration
├── results/<experiment>/    candidate tables (<run>.csv) and run configurations (<run>.json) of all runs
├── figure_images.json       images shown in the figures (run and file name in the archive)
└── prompts/                 PartiPrompts selection, Think-Then-Generate rewrites
plots/                       all generated figures (PDF + SVG) with their tables and data caches
```

## Experiments and figures of the manuscript

| Experiment | Setup | Manuscript | Analysis output |
|---|---|---|---|
| `example_prompts` | 12 prompts x 3 seeds, population 100 | Fig. 1E, 2A-C, 2E-F; Supp. Fig. 1A-C, 5A, 12 | `plots/example_prompts`, `plots/noise_distribution` |
| `parti_prompts` | 114 PartiPrompts x 3 seeds | Fig. 2D, 2G; Supp. Fig. 2, 5B; Supp. Table 1 | `plots/parti_prompts` |
| `seed_transfer` | noise of `example_prompts` runs rendered with another prompt | Fig. 3; Supp. Fig. 11 | `plots/seed_transfer` |
| `model_comparison` | 14 text-to-image models | Supp. Fig. 3A-B | `plots/model_comparison` |
| `fitness_embedding` | novelty in BLIP-2, CLIP, DINOv2 or SigLIP 2 space | Supp. Fig. 3C-D | `plots/fitness_embedding` |
| `fitness_function` | centroid vs. k nearest neighbours | Supp. Fig. 4 | `plots/fitness_function` |
| `long_runs` | 300 generations | Supp. Fig. 1D | `plots/long_runs` |
| `archive_size` | novelty archive of 0, 200 or all candidates | Supp. Fig. 6 | `plots/archive_size` |
| `crossover_function` | 1-, 3-, 7-point, uniform, arithmetic crossover | Supp. Fig. 7 | `plots/crossover_function` |
| `tournament_size` | tournaments of 2, 3, 5, 10 candidates | Supp. Fig. 8 | `plots/tournament_size` |
| `mutation_only` | Gaussian or uniform mutation without crossover | Supp. Fig. 9 | `plots/mutation_only` |
| `operator_ablation` | no crossover / no elitism / no mutation | not in the manuscript | `plots/operator_ablation` |
| `think_then_generate` | prompt rewriting with Think-Then-Generate, then seed optimisation | Supp. Fig. 10 | `plots/think_then_generate` |
| `mutation_grid_search` | every candidate mutated once per mutated fraction and strength | Methods (Mutation) | `plots/mutation_grid_search` |
| `gemma_ratings` | Gemma ratings of distance-optimised images across six perceptual dimensions | Ratings across generations | `plots/gemma_ratings` |
| `gemma_evolution` | Gemma 4, LLaVA-OneVision and Qwen2.5-VL compared with human ratings; evolution with seven Gemma fitness targets | Supp. Fig. 14A-E | `plots/gemma_evolution` |

Panels and files:

| Panel | File (`plots/...`) |
|---|---|
| Fig. 1E | `example_prompts/distance_to_current_mean_a_cat` |
| Fig. 2A, Supp. Fig. 1A-C | `example_prompts/images_<prompt>_detail`, `images_<prompt>_final` |
| Fig. 2C | `example_prompts/distance_to_initial_mean` (+ `_tests.csv`) |
| Fig. 2D, 2G | `parti_prompts/divergence_by_challenge`, `diversity_by_challenge` (+ `_stats.csv`) |
| Fig. 2F | `example_prompts/diversity` (+ `_tests.csv`) |
| Fig. 3A, Supp. Fig. 11A | `seed_transfer/examples_<source>_to_<target>` |
| Fig. 3B-C, Supp. Fig. 11C-D | `seed_transfer/<source>_to_a_cat_distance`, `_diversity` (+ `tests.csv`) |
| Supp. Fig. 1D | `long_runs/long_run_metrics`, `long_run_examples_<prompt>` |
| Supp. Fig. 2 | `parti_prompts/divergence_trajectories_by_challenge`, `diversity_trajectories_by_challenge` |
| Supp. Fig. 3A-B | `model_comparison/model_comp_divergence_curves`, `model_comp_diversity_curves` |
| Supp. Fig. 3C-D | `fitness_embedding/embedding_<divergence,diversity>_curves`, `embedding_heatmap` |
| Supp. Fig. 4, 6, 7 | `fitness_function/runs`, `archive_size/runs`, `crossover_function/runs` (and `overview` with CLIP score) |
| Supp. Fig. 5A, 5B | `example_prompts/clip_scores`, `parti_prompts/fidelity_pooled` |
| Supp. Fig. 8 | `tournament_size/tournament_<metric>_violins`, `_curves` |
| Supp. Fig. 9A, 9B | `mutation_only/examples_generation_8`, `mutation_only/distance_to_initial_mean` |
| Supp. Fig. 10A, 10B | `think_then_generate/t2g_supplement_g30` (tests: `t2g_conditions_g30_tests.csv`, `t2g_pairwise_g30.csv`) |
| Supp. Fig. 12A, 12B | `noise_distribution/example_prompts_noise_hist` (+ `example_prompts_normality.csv`), `example_prompts/noise_pca` |
| Ratings across generations | `gemma_ratings/ratings_across_generations` (+ `_tests.csv`, `_pairs.csv`, `_sensitivity.csv`) |
| Supp. Fig. 14A | `gemma_evolution/evaluation_and_evolution`, panel A (+ `human_vlm_correlations.csv`) |
| Supp. Fig. 14B | `gemma_evolution/evaluation_and_evolution`, panel B (+ `prompt_curves.csv`) |
| Supp. Fig. 14C | `gemma_evolution/evaluation_and_evolution`, panels C-D (+ `prompt_curves.csv`) |
| Supp. Fig. 14D | `gemma_evolution/evaluation_and_evolution`, panel E (images in `data/results/gemma/examples/`) |
| Supp. Fig. 14E | `gemma_evolution/evaluation_and_evolution`, panels F-H (+ `target_curves.csv`) |

Statistics: every figure with significance marks writes its tests next to it (`*_tests.csv`, `tests.csv`):
two-sided Mann-Whitney U tests of generation 0 vs. 30 on all candidates of the runs, Wilcoxon signed-rank
tests on run means where runs are paired, Holm correction as described in the module docstrings.
The Gemma generation comparison uses **one-sided paired prompt-level** Wilcoxon tests for an increase
from generation 0 to 30, with Holm correction across six questions. Tests are saved separately; stars
are not added to the descriptive figure. These are exploratory tests, not human-participant analyses.

### VLM evaluation data

The VLM analyses use three datasets:

| Analysis | Models and images | Included data |
|---|---|---|
| Agreement with human ratings | Gemma 4 26B-A4B, LLaVA-OneVision 7B and Qwen2.5-VL 7B; 36 study images and six rating dimensions | `data/results/gemma/benchmark/`: 648 original model responses, 644 valid image-level pairs, correlations and response completeness |
| Ratings across generations | Gemma 4; 114 source prompts rated on all six dimensions, generations 0, 10, 20 and 30; three runs and 60 images per generation | `data/results/gemma/posthoc/`: individual scores and run, prompt and dimension summaries; `statistics/`: paired tests |
| Rating-guided evolution | Seven Gemma fitness targets; cat, chair and boredom; three seeds and population 100 | `data/results/gemma/evolution/`: population metrics and pooled curves; `examples/`: nine selected original images |

The benchmark contains the original model responses and image-level human averages. The generation
ratings include individual scores and their run, prompt and dimension summaries. Sample sizes and valid
response counts are recorded in the
[generation summary](data/results/gemma/posthoc/question_generation_summary.csv) and shown in the figure.
Invalid responses remain missing values. Model settings, rating questions and aggregation methods are
described in [VLM evaluation and rating-guided evolution](docs/gemma.md).

## Methods of the manuscript in the code

| Method | Code |
|---|---|
| SDXL base 1.0, Euler discrete scheduler (pipeline default), guidance 7.0, 1024 x 1024, 50 / 30 steps | `evolutionary_creativity/models/text_to_image/sdxl.py`; float16 for population 100 (`experiments/common/variants.py`), bfloat16 for population 60 |
| Initial population from a seeded Gaussian, latents 4 x 128 x 128 | `evolutionary_creativity/noise_factory.py` |
| Novelty ½(1 − cos) to the centroid of the current population (Eq. 3-4), optional archive | `evolutionary_creativity/evaluators/divergence.py` |
| k nearest neighbours fitness (Supp. Fig. 4) | `evolutionary_creativity/evaluators/knn.py` |
| BLIP-2 vision encoder (1408), CLIP ViT-B/32, DINOv2 base, SigLIP 2 base | `evolutionary_creativity/models/embeddings/` |
| Tournament selection (k distinct candidates), elitism 1, crossover rate 0.9 (otherwise a copy of the first parent), mutation rate 0.05 | `evolutionary_creativity/operators/selection.py`, `evolutionary_creativity/pipeline.py`, `experiments/common/ga.py` |
| Uniform (p_s = 0.5), n-point (equally spaced cut points over the 128 x 128 positions), arithmetic ((p1 + p2)/√2) crossover | `evolutionary_creativity/operators/crossover.py` |
| Gaussian mutation of a fraction p_m = 0.1 with σ_m = 0.2; grid search of p_m and σ_m | `evolutionary_creativity/operators/mutation.py`, `experiments/mutation_grid_search/` |
| Distance to the mean of generation 0 (Eq. 5), diversity (Eq. 6), CLIP score (Eq. 7, prefix "A photo depicts ", 2.5) | `evolutionary_creativity/evaluators/` |
| CLIP scores and random baselines of runs without stored CLIP scores (Supp. Fig. 5A, 4, 6, 7) | `experiments/common/clip_scores.py` |
| Text-to-image models of the comparison with their settings | `experiments/model_comparison/models.py`, `evolutionary_creativity/models/text_to_image/` |
| Think-Then-Generate prompt rewriting (T2G system prompt, greedy decoding) | `experiments/think_then_generate/rewrite_prompts.py` |
| Noise transfer (generations 0, 10, 20, 30 rendered with another prompt) | `experiments/common/variants.py` (`transfer_populations`) |
| Statistical tests (Mann-Whitney U, Wilcoxon signed-rank, Kolmogorov-Smirnov, Holm) | `analysis/common/statistics.py`, `analysis/parti_prompts.py`, `analysis/noise_distribution.py`, `analysis/think_then_generate.py` |
| PCA of the noise (Supp. Fig. 12B) | `analysis/example_prompts.py` (`noise_projection`) |
| VLM questions, model identities and recorded inference settings for Gemma 4, LLaVA-OneVision and Qwen2.5-VL | `data/results/gemma/methods/study_question_configuration.yaml`, `data/results/gemma/benchmark/latest_responses.csv` |
| Human-VLM agreement (Spearman correlations of image-level ratings, Supp. Fig. 14A) | `analysis/common/gemma/evolution.py` (`validate_benchmark`) |
| Post-hoc rating summaries and generation-0/30 tests | `analysis/common/gemma/aggregation.py`, `analysis/common/gemma/statistics.py` |
| Rating-guided evolution: population metrics and averaging across seeds (Supp. Fig. 14B-E) | `data/results/gemma/evolution/`, `analysis/common/gemma/evolution.py` |

## Setup

```bash
pip install -e ".[analysis]"          # or ".[flux2-dev]" for the GPU environment
cp .env.example .env                  # then adapt paths
```

All settings (result directories, Hugging Face cache, population size) are read from the environment or
from `.env`. All modules are run from the repository root.

## Running experiments

Every experiment package has `config.py`, `create_jobs.py` and `worker.py`. `create_jobs` writes one row per
run into an SQLite job database; `worker <job id>` runs the genetic algorithm of one row on a GPU and marks it
`RUNNING` and then `DONE` or `FAILED`, so the jobs can be started by hand or by any batch system. The
databases in `data/jobs` list the analysed runs, so new jobs go into a new database:

```bash
python -m experiments.archive_size.create_jobs --db new_jobs.db
ARCHIVE_SIZE_DB_FILE=new_jobs.db python -m experiments.archive_size.worker 1      # one call per job id

# model comparison
python -m experiments.model_comparison.create_jobs --db new_jobs.db --prompts "a cat" "a chair" boredom --models all
MODEL_COMPARISON_DB_FILE=new_jobs.db python -m experiments.model_comparison.worker 1

# Think-Then-Generate: rewrite prompts, then seed optimisation with SDXL
python -m experiments.think_then_generate.rewrite_prompts --out plots/think_then_generate/rewrite_test
python -m experiments.think_then_generate.worker --prompt "a cat" --seed 1943746817 --generations 100
```

Two configurations are used:

* **Population 60** (`parti_prompts`, `model_comparison`, `fitness_embedding`, `tournament_size`, `long_runs`,
  `think_then_generate`): 30 generations, tournament size 3, uniform crossover (rate 0.9), Gaussian mutation
  of 10 % of the values with standard deviation 0.2 (probability 0.05 per child), elitism 1, SDXL with 30
  steps, seeds 1943746817, 1394004965 and 3030801653.
* **Population 100** (`experiments.common.variants`: `example_prompts`, `seed_transfer`, `archive_size`,
  `crossover_function`, `fitness_function`, `mutation_only`, `operator_ablation`): the same operators and
  rates unless a variant changes them, SDXL with 50 steps, batch size 5, three seeds per variant.

Every run writes `<run>.json` (configuration), `<run>.csv` (one row per candidate and generation with
lineage, fitness and all metrics) and `<run>.zip` (images, embeddings and noise of every candidate). The GIN
repositories above contain these files of all analysed runs; their READMEs describe how the released files
differ from those written by the code (archives compacted and reduced to selected generations, no absolute
paths in the configurations).

## Analysis

Run with the analysis environment from the repository root; every module writes into `plots/<module>/`
(`--preview DIR` additionally writes PNG previews).

```bash
python -m analysis                          # all modules
python -m analysis.example_prompts          # or a single one
python -m analysis.pca_parti_prompts compute   # PCA projections of the PartiPrompts (about 1.5 h), then:
python -m analysis.pca_parti_prompts plot
```

Expensive intermediate data are cached in `plots/*/data/` (e.g. the candidate table of all
PartiPrompts runs, the random CLIP baselines, the noise histograms, the noise PCA); `--recompute` rebuilds
them from the tables and archives.

### VLM evaluation and rating-guided evolution

The VLM comparison, generation-rating overview and evolution curves use the analysis environment above.
Their inputs are in `data/results/gemma/`. The correlation analysis includes Gemma 4, LLaVA-OneVision
and Qwen2.5-VL; the generation-rating and rating-guided evolution analyses use Gemma 4.

```bash
python -m analysis.gemma_ratings --deep --verify-only
python -m analysis.gemma_evolution --verify-only
python -m analysis --only gemma_ratings gemma_evolution --preview previews
```

PDF/SVG figures and recalculated CSV tables are written under `plots/gemma_ratings/` and
`plots/gemma_evolution/`; PNG previews go to the corresponding subdirectories of `previews/`.
Use `python -m analysis.gemma_ratings --deep` to reconstruct the summaries from all individual
ratings when rendering. These commands use the included numerical inputs and require no GPU inference.
