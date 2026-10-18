"""Human/VLM benchmark and Gemma-in-loop panels from saved numerical data."""

import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analysis.common.gemma import style
from analysis.common.gemma.evolution import METRICS, PROMPTS, TARGETS

METRIC_LABELS = {
    "fitness_mean": ("Gemma fitness", "Target rating (1–5)"),
    "divergence_mean": ("BLIP2 divergence", "Half-cosine distance"),
    "diversity": ("BLIP2 diversity", "Mean pairwise\nhalf-cosine distance"),
}


def letter(ax, text, x=-0.16, y=1.12):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def metric_limits(prompt_curves, target_curves):
    limits = {"fitness_mean": (0.9, 5.1)}
    for metric in METRICS[1:]:
        all_rows = pd.concat([prompt_curves, target_curves], ignore_index=True)
        low = (all_rows[f"{metric}_mean"] - all_rows[f"{metric}_sd"]).min()
        high = (all_rows[f"{metric}_mean"] + all_rows[f"{metric}_sd"]).max()
        padding = (high - low) * 0.09
        limits[metric] = (max(0.0, low - padding), high + padding)
    return limits


def draw_curves(ax, curves, metric, colors, group, limits, legend=False):
    for name, color in colors.items():
        selected = curves[curves[group].eq(name)].sort_values("generation")
        x = selected.generation.to_numpy(float)
        mean = selected[f"{metric}_mean"].to_numpy(float)
        sd = selected[f"{metric}_sd"].to_numpy(float)
        ax.plot(x, mean, color=color, linewidth=1.6, label=name)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
    style.style_axis(ax)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(*limits[metric])
    ax.set_xlabel("Generation")
    ax.set_ylabel(METRIC_LABELS[metric][1])
    ax.set_title(METRIC_LABELS[metric][0])
    if legend:
        ax.legend(frameon=False, loc="lower right")


def heatmap(fig, ax, correlations):
    models = ("gemma4-26b-a4b", "llava-onevision-7b", "qwen2.5-vl-7b")
    questions = (
        "boredom",
        "creativity",
        "excitement",
        "pleasantness",
        "suitability",
        "unusualness",
    )
    matrix = correlations.pivot(index="model_id", columns="prompt_id", values="spearman_r")
    values = matrix.loc[list(models), [f"study-{q}" for q in questions]].to_numpy(float)
    im = ax.pcolormesh(values, vmin=-1, vmax=1, cmap="coolwarm", rasterized=False)
    ax.grid(False)
    ax.set_ylim(3, 0)
    ax.set_yticks(np.arange(3) + 0.5, ["Gemma 4 26B-A4B", "LLaVA-OneVision 7B", "Qwen2.5-VL 7B"])
    ax.set_xticks(np.arange(6) + 0.5, [q.capitalize() for q in questions], rotation=20, ha="right")
    ax.tick_params(axis="both", length=0)
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            ax.text(x + 0.5, y + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Agreement with human image ratings")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[-1, 0, 1])
    cbar.solids.set_rasterized(False)
    cbar.set_label("Spearman ρ", labelpad=5)
    letter(ax, "A", x=-0.23)


def example_grid(fig, spec, examples, root):
    inner = spec.subgridspec(3, 3, hspace=0.07, wspace=0.04)
    for row, prompt in enumerate(PROMPTS):
        selected = examples[examples.prompt.eq(prompt)].sort_values("seed")
        for column, record in enumerate(selected.itertuples(index=False)):
            ax = fig.add_subplot(inner[row, column])
            with Image.open(Path(root) / record.original_image) as image:
                ax.imshow(np.asarray(image.convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_color(style.PROMPT_COLORS[prompt])
                spine.set_linewidth(1.2)
            if row == 0:
                ax.set_title(str(record.seed), fontsize=8, pad=3)
            ax.text(
                0.5,
                0.02,
                f"g{record.generation:02d}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5},
            )
            if column == 0:
                ax.set_ylabel(
                    prompt,
                    color=style.PROMPT_COLORS[prompt],
                    fontsize=8,
                    labelpad=5,
                    fontweight="bold",
                )
            if row == column == 0:
                letter(ax, "E", x=-0.7, y=1.30)


def render(correlations, prompt_curves, target_curves, examples, root, output):
    style.apply_style()
    fig = plt.figure(figsize=(style.FIGURE_WIDTH, 10.1))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.14,
        right=0.975,
        bottom=0.20,
        top=0.91,
        wspace=0.40,
        hspace=0.70,
        height_ratios=[1, 1.08, 1],
    )
    fig.suptitle("Gemma evaluation and image evolution", fontsize=15, y=0.985)
    heatmap(fig, fig.add_subplot(grid[0, :2]), correlations)
    creativity = prompt_curves[prompt_curves.construct.eq("creativity")]
    limits = metric_limits(creativity, target_curves)
    for metric, position, panel in (
        ("fitness_mean", (0, 2), "B"),
        ("divergence_mean", (1, 0), "C"),
        ("diversity", (1, 1), "D"),
    ):
        ax = fig.add_subplot(grid[position])
        draw_curves(
            ax,
            creativity,
            metric,
            style.PROMPT_COLORS,
            "prompt",
            limits,
            legend=metric == "fitness_mean",
        )
        letter(ax, panel)
    example_grid(fig, grid[1, 2], examples, root)
    for column, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[2, column])
        draw_curves(ax, target_curves, metric, style.TARGET_COLORS, "construct", limits)
        letter(ax, "FGH"[column])
    # One shared target legend avoids covering any of the seven curves.
    from matplotlib.lines import Line2D

    fig.legend(
        [Line2D([], [], color=style.TARGET_COLORS[t], linewidth=2) for t in TARGETS],
        [t.capitalize() for t in TARGETS],
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, 0.095),
        frameon=False,
        title="Fitness target (panels F–H)",
        title_fontsize=8,
    )
    fig.text(
        0.56,
        0.025,
        "B–D: creativity target, mean ± sample SD across seeds. F–H: equal prompt averages within seed, then mean ± sample SD.\n"
        "E: best final image per seed; actual generation 29 is retained where 30 is unavailable. Bands are not confidence intervals.\n"
        "A: 36 paired images per cell, except LLaVA unusualness (32). Each target uses its own rating question.",
        ha="center",
        fontsize=7,
    )
    stem = "evaluation_and_evolution"
    style.save_figure(fig, output, stem)
    # Keep full-resolution, byte-preserved JPEGs in the editable SVG, rather
    # than the smaller raster tiles Matplotlib generates for its preview.
    svg_path = Path(output) / f"{stem}.svg"
    svg = ET.parse(svg_path)
    images = list(svg.getroot().iter("{http://www.w3.org/2000/svg}image"))
    if len(images) != 9:
        raise ValueError("Expected exactly nine SVG image elements")
    records = [
        record
        for prompt in PROMPTS
        for record in examples[examples.prompt.eq(prompt)].sort_values("seed").itertuples(index=False)
    ]
    for element, record in zip(images, records, strict=True):
        payload = (Path(root) / record.original_image).read_bytes()
        element.set(
            "{http://www.w3.org/1999/xlink}href",
            "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"),
        )
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    svg.write(svg_path, encoding="utf-8", xml_declaration=True)
    return stem
