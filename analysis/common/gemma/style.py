"""Gemma figure styling, using the repository's shared theme and prompt colours.

The two multi-panel composites use twice the shared page width. Error bands
remain sample standard deviations; the theme does not change statistical units.
"""

from pathlib import Path

import matplotlib as mpl

from analysis.common import style as common

TITLE_SIZE = common.TITLE_SIZE
FIGURE_WIDTH = 2 * common.PAGE_WIDTH_IN
PROMPT_COLORS = {prompt: common.COLOR_MAP[prompt] for prompt in ("a cat", "a chair", "boredom")}
RATING_QUESTIONS = (
    ("study-suitability", "Suitability", "#69A9C2"),
    ("study-unusualness", "Unusualness", "#60A881"),
    ("study-boredom", "Boredom", "#9A92C2"),
    ("study-pleasantness", "Pleasantness", "#ECA260"),
    ("study-excitement", "Excitement", "#CA80A8"),
    ("study-creativity", "Creativity", "#B88B60"),
)
TARGET_COLORS = {
    "creativity": "#0072B2",
    "unusualness": "#D55E00",
    "uncommonness": "#009E73",
    "innovation": "#CC79A7",
    "novelty": "#E69F00",
    "originality": "#56B4E9",
    "uniqueness": "#6A3D9A",
}


def apply_style():
    """Use the shared theme and sizes, keeping SVG text editable."""
    common.reset_style()
    mpl.rcParams.update(
        {
            "font.size": common.LABEL_SIZE,
            "axes.titlesize": common.TITLE_SIZE,
            "axes.labelsize": common.LABEL_SIZE,
            "xtick.labelsize": common.TICK_SIZE,
            "ytick.labelsize": common.TICK_SIZE,
            "legend.fontsize": common.TICK_SIZE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "gemma-figures",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax):
    """Apply the repository's white grid and thin axes."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.5)
    ax.grid(axis="x", visible=False)


def save_figure(fig, output, stem):
    """Use the common PDF/SVG writer and optional --preview PNG destination."""
    common.save_figure(fig, Path(output) / stem)
