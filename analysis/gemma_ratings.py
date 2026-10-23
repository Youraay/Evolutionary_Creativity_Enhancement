"""Gemma ratings of distance-optimised images across generations.

Outputs: plots/gemma_ratings/ratings_across_generations.pdf|svg and paired
generation-0/30 statistics. All six dimensions cover the same 114 prompts.
Dots are source-prompt means.
Tests ask whether generation 30 is higher (one-sided Wilcoxon, six-test Holm).

    python -m analysis.gemma_ratings --deep --preview previews
"""

from analysis.common.gemma import data, ratings_plot, workflow


def main():
    """Check numerical inputs, plot the six dimensions and save separate tests."""
    args = workflow.arguments(__doc__, "gemma_ratings")
    groups = {"posthoc", "statistics", "methods"}
    before = workflow.check_hashes(args.data, groups)
    prompt, question, tests, paired, sensitivity = data.load_posthoc(args.data, deep=args.deep)
    if args.verify_only:
        expected = int(question.expected_rating_count.sum())
        valid = int(question.valid_rating_count.sum())
        print(f"[INFO] verified: {expected:,} scores; {valid:,} valid; six paired tests")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    stem = ratings_plot.render(prompt, question, args.output)
    tests.to_csv(args.output / "ratings_across_generations_tests.csv", index=False)
    paired.to_csv(args.output / "ratings_across_generations_pairs.csv", index=False)
    sensitivity.to_csv(args.output / "ratings_across_generations_sensitivity.csv", index=False)
    prompt.to_csv(args.output / "ratings_across_generations.csv", index=False)
    workflow.finish(args, "gemma_ratings", stem, before, groups)


if __name__ == "__main__":
    main()
