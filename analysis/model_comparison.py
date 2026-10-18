"""Figures of the model comparison (14 text-to-image models, 3 prompts, up to 3 seeds).

Output in ``plots/model_comparison/``:

    model_comp_<metric>_curves   one panel per model, mean +- SEM per prompt
    model_comp_<metric>_violins  one panel per model, violins of generations 0/10/20/30 (prompts pooled)
    model_comp_examples_<prompt> random example images of generations 0 and 30, all models except SDXL
    model_comp_tests.csv         Supplementary Fig. 3A-B: Mann-Whitney U tests of generation 0 vs. 30 per model,
                                 prompt and metric (runs pooled), Holm correction over the table

    python -m analysis.model_comparison
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common.paths import PLOTS_DIR
from analysis.common.plotting import (METRIC_LABELS, PROMPTS, SEED_ORDER, gen_violins, prompt_curves,
                                      prompt_legend_handles, set_ticks)
from analysis.common.results import load_images, load_model_comparison, random_candidates
from analysis.common.statistics import generation_test, holm, stars
from analysis.common.style import (ANNOTATE_SIZE, LABEL_SIZE, PAGE_WIDTH_IN, TICK_SIZE, TITLE_SIZE, save_figure,
                                   set_preview_dir)

OUT_DIR = PLOTS_DIR / "model_comparison"

MODELS = {
    "sd15":              ("SD 1.5",             "UNet"),
    "sd_turbo":          ("SD-Turbo",           "UNet"),
    "sdxl":              ("SDXL",               "UNet"),
    "sdxl_turbo":        ("SDXL-Turbo",         "UNet"),
    "ssd_1b":            ("SSD-1B",             "UNet"),
    "segmind_vega":      ("Segmind Vega",       "UNet"),
    "playground_v25":    ("Playground v2.5",    "UNet"),
    "realvisxl_v5":      ("RealVisXL V5",       "UNet"),
    "juggernaut_xl_v9":  ("Juggernaut XL v9",   "UNet"),
    "z_image":           ("Z-Image",            "Transformer"),
    "flux2_klein_4b":    ("Flux2 Klein 4B",     "Transformer"),
    "flux2_klein_9b_kv": ("Flux2 Klein 9B KV",  "Transformer"),
    "krea_2_turbo":      ("Krea 2 Turbo",       "Transformer"),
    "flux2_dev":         ("Flux2 dev",          "Transformer"),
}
"""Model name -> (display name, architecture), in display order."""

MODEL_GRID = [["sd15", "sd_turbo", "sdxl", "sdxl_turbo", "ssd_1b"],
              ["segmind_vega", "playground_v25", "realvisxl_v5", "juggernaut_xl_v9", None],
              ["z_image", "flux2_klein_4b", "flux2_klein_9b_kv", "krea_2_turbo", "flux2_dev"]]
"""Panel layout; the empty slot holds the prompt legend."""


def generation_tests(data: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U tests of generation 0 vs. 30 for every model, prompt and metric (runs pooled).

    Holm correction over all tests of the table.

    Args:
        data: Candidates of all models.

    Returns:
        One row per model, prompt and metric; also written to ``model_comp_tests.csv``.
    """
    rows = []
    for model, (name, architecture) in MODELS.items():
        for prompt in PROMPTS:
            df = data[(data["model"] == model) & (data["prompt"] == prompt)]
            if df.empty:
                continue
            for key in ("divergence", "diversity"):
                rows.append({"model": name, "architecture": architecture, "prompt": prompt,
                             "runs": df["seed"].nunique(), **generation_test(df, key, 0, 30)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"])
    table["sig"] = table["p_holm"].map(stars)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "model_comp_tests.csv", index=False)
    return table


def render_model_panels(data: pd.DataFrame, key: str, kind: str) -> None:
    """One panel per model with curves per prompt or violins per generation.

    Args:
        data: Candidates of all models.
        key: ``"divergence"`` or ``"diversity"``.
        kind: ``"curves"`` or ``"violins"``.
    """
    nrows, ncols = len(MODEL_GRID), len(MODEL_GRID[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_WIDTH_IN, 4.6), sharex=True, sharey=True,
                             layout="constrained")
    for r, row in enumerate(MODEL_GRID):
        for c, model in enumerate(row):
            ax = axes[r, c]
            if model is None:
                ax.axis("off")
                if kind == "curves":
                    ax.legend(handles=prompt_legend_handles(), loc="center", fontsize=TICK_SIZE, frameon=False,
                              handlelength=1.8)
                continue
            subset = data[data["model"] == model]
            if kind == "curves":
                prompt_curves(ax, subset, key, lw=1.2)
                set_ticks(ax, [0, 15, 30])
            else:
                gen_violins(ax, subset, key, dot_size=0.4)
                set_ticks(ax)
            ax.set_title(MODELS[model][0], fontsize=TICK_SIZE - 1.5, fontweight="bold", pad=4)
            if c == 0:
                ax.annotate(MODELS[model][1], xy=(-0.42, 0.5), xycoords="axes fraction", ha="right",
                            va="center", fontsize=ANNOTATE_SIZE, fontweight="bold", rotation=90)
    fig.supxlabel("Generation", fontsize=LABEL_SIZE)
    fig.supylabel(METRIC_LABELS[key].replace("\n", " "), fontsize=LABEL_SIZE)
    save_figure(fig, OUT_DIR / f"model_comp_{key}_{kind}")


def render_model_examples(data: pd.DataFrame, runs: pd.DataFrame, prompt: str, seed: str = SEED_ORDER[0],
                          n: int = 2) -> None:
    """Two blocks (UNet / transformer), one row per model: ``n`` random images of generation 0 and of 30.

    Args:
        data: Candidates of all models.
        runs: Runs of all models.
        prompt: Prompt to show.
        seed: Run seed.
        n: Images per generation.
    """
    blocks = {family: [m for m, (_, f) in MODELS.items() if f == family and m != "sdxl"]
              for family in ("UNet", "Transformer")}
    label_widths = {"UNet": 0.85, "Transformer": 1.1}       # room for the model names left of a block
    gap, top, title_h, pad, gen_gap = 0.05, 0.62, 0.30, 0.03, 0.06
    img = (PAGE_WIDTH_IN - sum(label_widths.values()) - gap - 2 * ((2 * n - 1) * pad + gen_gap)) / (4 * n)
    nrows = max(len(v) for v in blocks.values())
    height = top + nrows * img + (nrows - 1) * pad
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, height))
    rng = np.random.default_rng(0)

    def axes_at(x, y):
        """Image axes at ``(x, y)`` inches from the top left corner."""
        return fig.add_axes([x / PAGE_WIDTH_IN, 1 - (y + img) / height, img / PAGE_WIDTH_IN, img / height])

    x0 = 0
    for family, models in blocks.items():
        label_w = label_widths[family]
        images_w = 2 * n * img + (2 * n - 1) * pad + gen_gap
        fig.text((x0 + label_w + images_w / 2) / PAGE_WIDTH_IN, 1 - title_h / height,
                 "UNet" if family == "UNet" else "Transformer (DiT)", ha="center", va="bottom",
                 fontsize=TITLE_SIZE, fontweight="bold")
        for g, generation in enumerate((0, 30)):
            xc = x0 + label_w + g * (n * (img + pad) + gen_gap) + (n * img + (n - 1) * pad) / 2
            fig.text(xc / PAGE_WIDTH_IN, 1 - (top - 0.05) / height, f"Gen. {generation}", ha="center", va="bottom",
                     fontsize=LABEL_SIZE)
        for r, model in enumerate(models):
            run = runs[(runs["model"] == model) & (runs["prompt"] == prompt) & (runs["seed"] == seed)].iloc[0]
            d = data[(data["model"] == model) & (data["prompt"] == prompt) & (data["seed"] == seed)]
            files = random_candidates(d, 0, n, rng) + random_candidates(d, 30, n, rng)
            y = top + r * (img + pad)
            for k, image in enumerate(load_images(run["zip"], files, size=int(img * 400))):     # ~400 dpi
                ax = axes_at(x0 + label_w + k * (img + pad) + (gen_gap if k >= n else 0), y)
                ax.imshow(image, interpolation="none")
                ax.axis("off")
            fig.text((x0 + label_w - 0.06) / PAGE_WIDTH_IN, 1 - (y + img / 2) / height, MODELS[model][0],
                     ha="right", va="center", fontsize=TICK_SIZE)
        x0 += label_w + images_w + gap
    fig.suptitle(f'"{prompt}"', fontsize=TITLE_SIZE, fontweight="bold", y=1 + 0.05 / height, va="bottom")
    save_figure(fig, OUT_DIR / f"model_comp_examples_{prompt.replace(' ', '_')}")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    set_preview_dir(parser.parse_args().preview)
    data, runs = load_model_comparison()
    generation_tests(data)
    for key in ["diversity", "divergence"]:
        for kind in ("curves", "violins"):
            render_model_panels(data, key, kind)
    for prompt in PROMPTS:
        render_model_examples(data, runs, prompt)


if __name__ == "__main__":
    main()
