"""Human/VLM agreement and Gemma-guided image evolution.

Outputs: plots/gemma_evolution/evaluation_and_evolution.pdf|svg, recalculated
correlations and mean/sample-SD curves. The benchmark and evolutionary datasets
are distinct. Actual generation-29 endpoints are retained. Bands show SD,
not the SEM used by most other analysis modules. See docs/gemma.md.

    python -m analysis.gemma_evolution --preview previews
"""

import pandas as pd

from analysis.common.gemma import data, evolution, evolution_plot, workflow


def main():
    """Recalculate benchmark correlations and seed pooling, then render the figure."""
    args = workflow.arguments(__doc__, "gemma_evolution")
    groups = {"benchmark", "evolution", "examples", "methods"}
    before = workflow.check_hashes(args.data, groups)
    prompt_curves, target_curves, examples = data.load_evolution(args.data)
    correlations = evolution.validate_benchmark(
        pd.read_csv(args.data / "benchmark/metrics.csv"),
        pd.read_csv(args.data / "benchmark/headline_analysis_units.csv"),
    )
    if args.verify_only:
        print("[INFO] verified: 18 correlations; 1,951 populations; nine original examples")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    stem = evolution_plot.render(
        correlations, prompt_curves, target_curves, examples, args.data, args.output
    )
    correlations.to_csv(args.output / "human_vlm_correlations.csv", index=False)
    prompt_curves.to_csv(args.output / "prompt_curves.csv", index=False)
    target_curves.to_csv(args.output / "target_curves.csv", index=False)
    workflow.finish(args, "gemma_evolution", stem, before, groups)


if __name__ == "__main__":
    main()
