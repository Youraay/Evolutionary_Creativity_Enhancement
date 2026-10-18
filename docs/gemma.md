# Gemma ratings and image evolution

These analyses follow the repository's `python -m analysis.<module>` interface,
shared plotting theme, prompt colours and PDF/SVG exports. Their inputs live in
`data/results/gemma/`, their derived tables and figures in `plots/<module>/`.
Both composites use twice `analysis.common.style.PAGE_WIDTH_IN`; text remains
editable in the SVG. The supplied nine JPEGs are embedded at original resolution.

## Reproduction

Use the existing analysis environment (`pip install -e ".[analysis]"`), then run
these commands from the repository root:

```bash
python -m analysis.gemma_ratings --deep --verify-only
python -m analysis.gemma_evolution --verify-only
python -m analysis --only gemma_ratings gemma_evolution --preview previews
```

`--deep` recomputes the rating summaries from all individual scores. Both modules
always check source hashes and reconcile their calculations with saved results.
`--verify-only` writes nothing. `--output DIR` overrides a module's plot directory;
otherwise the existing `PLOTS_DIR` environment setting is respected. Re-running
overwrites only the named derived outputs, following the other analysis modules.
The original inputs are immutable. `--data DIR` can point to a copy of the same
input folder. No inference, model downloads or archive access is required.

The dependency versions used to check this integration were Python 3.12, NumPy
2.5.2, pandas 2.3.3, SciPy 1.18.1, Matplotlib 3.11.1, Pillow 11.3.0 and seaborn
0.13.2. Original inference environments are separate historical environments.

| Script | Inputs under `data/results/gemma/` | Outputs |
| --- | --- | --- |
| `analysis/gemma_ratings.py` | `posthoc/`, `statistics/` | `plots/gemma_ratings/ratings_across_generations.pdf`, `.svg`, `.csv`, `_tests.csv`, `_pairs.csv`, `_sensitivity.csv` |
| `analysis/gemma_evolution.py` | `benchmark/`, `evolution/`, `examples/` | `plots/gemma_evolution/evaluation_and_evolution.pdf`, `.svg`, `human_vlm_correlations.csv`, `prompt_curves.csv`, `target_curves.csv` |
| `analysis/common/gemma/aggregation.py` | individual ratings | image-to-run, source-prompt and question summaries |
| `analysis/common/gemma/statistics.py` | paired source-prompt means | six directional tests, Holm correction and sensitivity calculations |
| `analysis/common/gemma/evolution.py` | benchmark pairs and population metrics | correlations and balanced seed pooling |
| `analysis/common/gemma/data.py`, `workflow.py` | saved source tables and `manifest.json` | coverage, numerical reconciliation and input/output hash checks |
| `analysis/common/gemma/ratings_plot.py`, `evolution_plot.py`, `style.py` | validated summaries and nine originals | figure composition using the repository's shared theme and save helper |

Each output directory also contains `render_manifest.json` with software versions
and source/code/output hashes. Optional PNG previews follow the common `--preview`
layout. The CSVs next to each figure are regenerated, not manually edited numbers.

Question text and metric definitions are under `data/results/gemma/methods/`.
The YAML is the preserved study-question template, not a new runnable inference
configuration. `source_inventory.json` records historical source paths and hashes;
those historical paths are not runtime dependencies.
The inventory's `package_file` paths are relative to `data/results/gemma/`;
`original_package_file` retains the earlier standalone-package layout.

## Ratings across generations

This is the distance-optimized image dataset subsequently rated by Gemma, **not**
Gemma-in-the-loop evolution. Generations are 0, 10, 20 and 30. Creativity and
unusualness cover 114 source prompts. Suitability, boredom, pleasantness and
excitement retain 108 prompts; the additional six were not rated on those questions.
Each prompt has three runs and 60 image observations per generation. Repeated elite
images remain observations in their populations.

The supplied table has 475,200 expected scores, 475,150 valid scores and 50
parse-invalid scores left missing. Each dot is a source-prompt mean across valid
image observations; faint lines pair the same prompt across generations. Violins
describe these prompt means. Horizontal marks equally weight source prompts.
All prompts use the same circular markers. Do not infer 114-prompt coverage for
all six questions or treat the individual images as independent replications.

The separate exploratory tests ask whether generation 30 is **higher** than
generation 0 for each dimension. They use paired prompt means, one-sided Wilcoxon
signed-rank tests and Holm correction across six tests. Only unusualness passes
the corrected 0.05 threshold. Creativity has raw p=0.0201 but adjusted p=0.1003.
`n.s.` means no demonstrated increase, not equivalence or absence of a decrease.
The signed-rank interpretation assumes reasonably symmetric paired differences;
the source prompts are treated as independent units. These are not preregistered
confirmatory tests or evidence about human ratings. No stars are added to the
figure automatically. Exact sign-test and complete-endpoint sensitivity results
are supplied separately.

## Evaluation and image evolution

**A** compares image-level human aggregates with Gemma 4, LLaVA-OneVision and
Qwen2.5-VL. All 18 Spearman correlations are recalculated from the included paired
values. Most cells have 36 image pairs; LLaVA unusualness has 32 because four
responses were invalid. The table has 644 valid out of 648 expected pairs.
No individual participant records are included. This benchmark is separate from
the 108/114-prompt image dataset and is not enlarged by the six added prompts.

**B–D** show fitness, divergence and diversity for creativity-target evolution
of cat/chair/boredom, averaged across seeds 2025, 2026 and 2027, with sample SD
between seeds. **E** shows the best-rated final-population image for each prompt
and seed. The seed-2025 cat and boredom runs end at generation 29; their examples
are labeled g29, never relabeled g30. At generation 30 those two curves have two
seeds; chair retains three.

**F–H** compare seven separate fitness conditions. At each seed and generation,
cat/chair/boredom are equally averaged, then the seed averages are summarized with
mean and sample SD. The incomplete creativity/seed-2025/generation-30 block is
excluded rather than silently averaged from chair alone. Generation 29 remains 29.
Fitness is each condition's own rating question, not a common creativity score.
SD bands describe between-seed variability, not confidence intervals or significance.

BLIP2 divergence and diversity use half-cosine distances. Exact definitions,
encoder identity and the distinction from the printed manuscript equation are
in [metric definitions](../data/results/gemma/methods/metric_definitions.txt).

## Scope

This package reproduces both figures, their source summaries, benchmark
correlations and the specified post-hoc significance tests. It includes all
individual post-hoc scores, but the evolutionary curves start from the supplied
per-population metric values. Full image archives, raw BLIP2 embeddings, latent
noise tensors, model checkpoints, individual human participant data and cluster
job administration are not needed to redraw these figures and are not included.
Therefore it does not rerun the evolutionary experiments or recompute BLIP2
embeddings from every generated image. Source hashes and the exact nine images
are retained. No existing experiment, repository or result archive is overwritten.
