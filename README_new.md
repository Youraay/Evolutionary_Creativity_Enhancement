# Evolutionary Creativity Enhancement

A genetic algorithm that optimises the **initial noise** of text-to-image diffusion models towards novel
images. A population of noise tensors is rendered by a frozen text-to-image model (SDXL by default), every
image is embedded with BLIP-2, and the fitness of a candidate is its distance to the mean embedding of the
current population. Tournament selection, crossover and mutation act directly on the noise tensors.

This repository contains the method, the Slurm infrastructure of all experiments, the job databases and
candidate tables of all analysed runs, and the code of all figures and tables.

## Repository layout

```
evolutionary_creativity/     the method
├── pipeline.py              GeneticAlgorithmPipeline (evaluate, evolve, save every generation)
├── candidate.py             one individual: noise, image, scores, lineage
├── noise_factory.py         initial population and candidate ids
├── operators/               tournament selection, crossover (uniform, n-point, arithmetic), mutation
├── evaluators/              novelty (distance to mean embedding, k nearest neighbours), diversity, CLIP score
└── models/                  text-to-image models (SDXL, UNet and DiT models, Think-Then-Generate)
                             and image embeddings (BLIP-2, CLIP, DINOv2, SigLIP 2)
experiments/                 one package per experiment: create_jobs -> dispatch -> worker (Slurm)
└── common/                  settings, SQLite job queue, dispatcher, standard GA, configurable experiment
                             families (variants, embedding_spaces), start_experiments.sh, collect_tables
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
| `example_prompts` | 12 prompts x 3 seeds, population 100 | Fig. 1E, 2A-C, 2E-F; Supp. Fig. 1A-C, 5A, 11 | `plots/example_prompts`, `plots/noise_distribution` |
| `parti_prompts` | 114 PartiPrompts x 3 seeds | Fig. 2D, 2G; Supp. Fig. 2, 5B; Supp. Table 1 | `plots/parti_prompts` |
| `seed_transfer` | noise of `example_prompts` runs rendered with another prompt | Fig. 3; Supp. Fig. 10 | `plots/seed_transfer` |
| `model_comparison` | 14 text-to-image models | Supp. Fig. 3A-B | `plots/model_comparison` |
| `fitness_embedding` | novelty in BLIP-2, CLIP, DINOv2 or SigLIP 2 space | Supp. Fig. 3C-D | `plots/fitness_embedding` |
| `fitness_function` | centroid vs. k nearest neighbours | Supp. Fig. 4 | `plots/fitness_function` |
| `long_runs` | 300 generations | Supp. Fig. 1D | `plots/long_runs` |
| `archive_size` | novelty archive of 0, 200 or all candidates | Supp. Fig. 6 | `plots/archive_size` |
| `crossover_function` | 1-, 3-, 7-point, uniform, arithmetic crossover | Supp. Fig. 7 | `plots/crossover_function` |
| `tournament_size` | tournaments of 2, 3, 5, 10 candidates | Supp. Fig. 8 | `plots/tournament_size` |
| `mutation_only` | Gaussian or uniform mutation without crossover | Supp. Fig. 9 | `plots/mutation_only` |
| `operator_ablation` | no crossover / no elitism / no mutation | not in the manuscript | `plots/operator_ablation` |
| `think_then_generate` | prompt rewriting with Think-Then-Generate, then seed optimisation | not in the manuscript | `plots/think_then_generate` |

The human study and the vision-language-model ratings of the manuscript (Fig. 4, Supp. Figs. 12-13) are not
part of this repository.

Panels and files:

| Panel | File (`plots/...`) |
|---|---|
| Fig. 1E | `example_prompts/distance_to_current_mean_a_cat` |
| Fig. 2A, Supp. Fig. 1A-C | `example_prompts/images_<prompt>_detail`, `images_<prompt>_final` |
| Fig. 2C | `example_prompts/distance_to_initial_mean` (+ `_tests.csv`) |
| Fig. 2D, 2G | `parti_prompts/divergence_by_challenge`, `diversity_by_challenge` (+ `_stats.csv`) |
| Fig. 2F | `example_prompts/diversity` (+ `_tests.csv`) |
| Fig. 3A, Supp. Fig. 10A | `seed_transfer/examples_<source>_to_<target>` |
| Fig. 3B-C, Supp. Fig. 10C-D | `seed_transfer/<source>_to_a_cat_distance`, `_diversity` (+ `tests.csv`) |
| Supp. Fig. 1D | `long_runs/long_run_metrics`, `long_run_examples_<prompt>` |
| Supp. Fig. 2 | `parti_prompts/divergence_trajectories_by_challenge`, `diversity_trajectories_by_challenge` |
| Supp. Fig. 3A-B | `model_comparison/model_comp_divergence_curves`, `model_comp_diversity_curves` |
| Supp. Fig. 3C-D | `fitness_embedding/embedding_<divergence,diversity>_curves`, `embedding_heatmap` |
| Supp. Fig. 4, 6, 7 | `fitness_function/runs`, `archive_size/runs`, `crossover_function/runs` (and `overview` with CLIP score) |
| Supp. Fig. 5A, 5B | `example_prompts/clip_scores`, `parti_prompts/fidelity_pooled` |
| Supp. Fig. 8 | `tournament_size/tournament_<metric>_violins`, `_curves` |
| Supp. Fig. 9A, 9B | `mutation_only/examples_generation_8`, `mutation_only/distance_to_initial_mean` |
| Supp. Fig. 11A, 11B | `noise_distribution/example_prompts_noise_hist` (+ `example_prompts_normality.csv`), `example_prompts/noise_pca` |

Statistics: every figure with significance marks writes its tests next to it (`*_tests.csv`, `tests.csv`):
two-sided Mann-Whitney U tests of generation 0 vs. 30 on all candidates of the runs, Wilcoxon signed-rank
tests on run means where runs are paired, Holm correction as described in the module docstrings.

## Data

* **Job databases** (`data/jobs/<experiment>.db`): one row per analysed run with its complete configuration,
  status `DONE`, Slurm job id and result directory. Runs that were started with one script per run (the
  experiments of `experiments.common.variants`) are listed with `create_jobs --register`.
* **Candidate tables** (`data/results/<experiment>/`): copies of all `<run>.csv` and `<run>.json` files of the
  result directories, with the same layout. The analysis reads only these tables, except for figures that need
  images, embeddings or noise.
* **Archives** (`<run>.zip`: images, embeddings and noise of every candidate) stay in the result directories,
  `<RESULTS_ROOT>/<experiment>` or `<EXPERIMENT>_RESULT_DIR` (see `.env.example` for the directories on the
  dldevel cluster). `python -m experiments.common.collect_tables <experiment>` copies new tables into the
  repository (with `--db` also a job database, e.g. that of still running jobs).

Column names: runs of `experiments.common.variants` that were registered from their result directories
store `global_score` (distance to the mean of generation 0), `fitness`, `diversity_score` (the same value for
all candidates of a generation) and `prompt_fidelty` (cosine similarity of the CLIP embeddings of image and
prompt; the figures use the CLIP score instead, `clip_scores/`); all other runs store `model_<metric>`
columns, e.g. `model_LocalMaxMeanDivergence`, `model_DiversityWithinPopulation` and `model_CLIPScore`.
41 of the 87 tables of `seed_transfer` contain every row twice; the analysis drops exact duplicates.

## Setup

Two Python environments were used on the cluster:

* **Experiments** (GPU jobs): Python 3.13 with torch 2.14 (CUDA 13), diffusers 0.40, transformers 5.16,
  accelerate 1.14, bitsandbytes 0.50 (only for FLUX.2 [dev]); the venv
  `/work/dldevel/public/Evolutionary_Diffusion_Enhancement/.venv`.
* **Analysis** (CPU): Python 3.12 with matplotlib 3.10, seaborn 0.13, pandas 2.2, scipy 1.15,
  scikit-learn 1.7, pyarrow 20, torch 2.7 (only for reading result archives); the conda environment
  `genetic`.

```bash
pip install -e ".[analysis]"          # or ".[flux2-dev]" for the GPU environment
cp .env.example .env                  # then adapt paths
```

All settings (result directories, Hugging Face cache, population size, Slurm limits) are read from the
environment or from `.env`. All modules are run from the repository root.

## Running experiments

Every experiment package has `config.py`, `create_jobs.py`, `dispatch.py`, `worker.py` and `submit_job.sh`.
`create_jobs` writes one row per run into an SQLite job database; a dispatcher on the login node (e.g. in
tmux) submits one Slurm job per open row and respects the per-user job limit; the worker runs the genetic
algorithm and marks the row `DONE` or `FAILED`. Rows with another status (e.g. `HOLD`) are ignored by the
dispatcher. The databases in `data/jobs` list the analysed runs, so new jobs go into a new database:

```bash
python -m experiments.archive_size.create_jobs --db new_jobs.db
ARCHIVE_SIZE_DB_FILE=new_jobs.db python -m experiments.archive_size.dispatch

# run an experiment under another Slurm user with its own database, results and logs
experiments/common/start_experiments.sh tournament_size start

# model comparison: single runs, or fixed per-GPU queues in one Slurm job
python -m experiments.model_comparison.create_jobs --db new_jobs.db --prompts "a cat" "a chair" boredom --models all
sbatch --gres=gpu:2 experiments/model_comparison/submit_gpu_queue.sh "125 127" "126 128"

# Think-Then-Generate: rewrite prompts, then seed optimisation with SDXL or the T2G DiT
sbatch experiments/think_then_generate/submit_rewrite.sh plots/think_then_generate/rewrite_test
sbatch experiments/think_then_generate/submit_sdxl.sh "a cat" 1943746817 100
sbatch experiments/think_then_generate/submit_t2g_dit.sh "a cat" 1943746817 t2g
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
lineage, fitness and all metrics) and `<run>.zip` (images, embeddings and noise of every candidate).

## Analysis

Run with the analysis environment from the repository root; every module writes into `plots/<module>/`
(`--preview DIR` additionally writes PNG previews).

```bash
python -m analysis                          # all modules
python -m analysis.example_prompts          # or a single one
sbatch analysis/submit_pca.sh               # PCA projections of the PartiPrompts (CPU node, about 1.5 h), then:
python -m analysis.pca_parti_prompts plot
```

Error bands are the standard error of the mean over runs (every run is first averaged over its
candidates). Expensive intermediate data are cached in `plots/*/data/` (e.g. the candidate table of all
PartiPrompts runs, the random CLIP baselines, the noise histograms, the noise PCA); `--recompute` rebuilds
them from the tables and archives.
