"""Figure layout, colours and helpers shared by all plots (style of the paper figures).

Importing this module applies the seaborn theme of all figures (``whitegrid``, ``talk`` context, thin
axes lines).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis.common.paths import PLOTS_DIR

PAGE_WIDTH = 398.33864
"""Text width of the paper in points."""
PAGE_WIDTH_IN = PAGE_WIDTH / 72.27

MAX_FIG_COLUMNS = 3
LEFT_MARGIN = 1.0
RIGHT_MARGIN = 0.2
TOP_MARGIN = 0.8
BOTTOM_MARGIN = 0.05
FIXED_COL_WIDTH = (PAGE_WIDTH_IN - LEFT_MARGIN - RIGHT_MARGIN) / MAX_FIG_COLUMNS
"""Width of one panel when three panels fill the page width."""

MAX_IMAGE_COLUMNS = 8
IMG_LEFT_MARGIN = 0.4
IMG_RIGHT_MARGIN = 0.0
IMG_TOP_MARGIN = 0.5
IMG_BOTTOM_MARGIN = 0.0
FIXED_IMAGE_WIDTH = (PAGE_WIDTH_IN - IMG_LEFT_MARGIN - IMG_RIGHT_MARGIN) / MAX_IMAGE_COLUMNS

COLORS = ["#A6CEE3", "#1F78B4", "#FDAE6B", "#E6550D", "#BCBDDC", "#54278F"]
ANIMATED_COLORS = COLORS[0:2]
INANIMATED_COLORS = COLORS[2:4]
ABSTRACT_COLORS = COLORS[4:]

COLOR_MAP = {
    "a cat": "#A6CEE3",
    "a dog": "#1F78B4",
    "a chair": "#FDAE6B",
    "a coffee mug": "#E6550D",
    "boredom": "#BCBDDC",
    "creativity": "#54278F",
}
"""Colour of each of the original six prompts in all figures."""

TITLE_SIZE = 9
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 10
ANNOTATE_SIZE = 9
MAX_Y_TICKS = 5

ERRORBAR = "sd"
"""Error bands are the standard deviation over all candidates of a generation, the runs pooled."""

PREVIEW_DIR: Path | None = None
"""If set, :func:`save_figure` also writes a PNG preview into this directory."""


def apply_theme() -> None:
    """Apply the seaborn theme of the figures; the axes line width is set afterwards (the theme resets it)."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.linewidth"] = 0.5


def reset_style(thin_axes: bool = True) -> None:
    """Restore the rc file defaults, apply the seaborn theme and seed NumPy (jitter of the strip plots).

    Every figure group of the experiments with population 100 starts from this state, so the figures do not
    depend on the order in which they are drawn.

    Args:
        thin_axes: Set the axes line width to 0.5 after the theme; the CLIP-score figures keep the theme
            default.
    """
    mpl.rc_file_defaults()
    sns.set_theme(style="whitegrid", context="talk")
    if thin_axes:
        mpl.rcParams["axes.linewidth"] = 0.5
    np.random.seed(0)


def save_figure(fig: plt.Figure, path: Path, redraw: bool = False) -> None:
    """Save a figure as PDF and SVG (300 dpi, tight bounding box) and close it.

    Args:
        fig: Figure to save.
        path: Output path without extension; directories are created.
        redraw: Redraw the canvas after every saved format, as ``pyplot.savefig`` does. The constrained
            layout of the SVG depends slightly on this; the paper figures were created that way.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(path.with_name(f"{path.name}.{ext}"), dpi=300, bbox_inches="tight")
        if redraw:
            fig.canvas.draw_idle()
    if PREVIEW_DIR is not None:
        relative = path.relative_to(PLOTS_DIR) if path.is_relative_to(PLOTS_DIR) else Path(path.name)
        preview = PREVIEW_DIR / relative.parent / f"{relative.name}.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved: {path}.pdf|svg", flush=True)


def set_preview_dir(directory: Path | None) -> None:
    """Enable (or disable with ``None``) PNG previews of all saved figures.

    Args:
        directory: Directory for the previews.
    """
    global PREVIEW_DIR
    PREVIEW_DIR = directory


apply_theme()
