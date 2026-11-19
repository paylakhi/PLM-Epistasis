"""Shared plotting style for Figures 1 and 2."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

FIGSIZE = (7, 5)
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 11
ANNOTATION_SIZE = 10.5
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
INK = "#101010"
AXIS_INK = "#202020"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS_INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.titleweight": "medium",
    "axes.labelweight": "medium",
    "axes.linewidth": 1.25,
    "lines.linewidth": 2.5,
    "lines.markersize": 7,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def polish_panel(ax):
    """Apply consistent, high-contrast finishing to existing axes.

    This function is called after all artists are drawn in the existing scripts,
    so it also improves elements whose styling was hardcoded in those scripts.
    """
    ax.title.set_fontsize(TITLE_SIZE)
    ax.title.set_fontweight("medium")
    ax.title.set_color(INK)

    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontsize(LABEL_SIZE)
        label.set_fontweight("medium")
        label.set_color(INK)

    # Respect deliberate compact tick labels on dense axes (e.g. ClinVar).
    ax.tick_params(axis="both", which="major", width=1.2, length=4.5, colors=INK)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color("#111111")
        label.set_fontweight("semibold")
        # Keep explicit small tick sizes where set in the plotting script.

    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color(AXIS_INK)

    # Improve main plotted lines while keeping dashed reference/guide lines subtle.
    for line in ax.lines:
        if line.get_linestyle() not in ("--", ":", "-."):
            line.set_linewidth(max(line.get_linewidth(), 2.5 if line.get_marker() not in (None, "None", "", " ") else 1.6))
        else:
            line.set_linewidth(max(line.get_linewidth(), 1.15))
            line.set_alpha(max(line.get_alpha() if line.get_alpha() is not None else 1.0, 0.7))

    # Boxplot outlines and whiskers, plus bar-chart error-bar caps/collections.
    for patch in ax.patches:
        if hasattr(patch, "get_linewidth") and hasattr(patch, "set_linewidth"):
            patch.set_linewidth(max(patch.get_linewidth(), 1.4))
    for collection in ax.collections:
        if isinstance(collection, LineCollection):
            widths = collection.get_linewidths()
            if len(widths):
                collection.set_linewidths([max(float(w), 1.6) for w in widths])

    # Values above bars, group descriptions and sample-size labels.
    for annotation in ax.texts:
        annotation.set_fontsize(ANNOTATION_SIZE)
        annotation.set_fontweight("semibold")
        annotation.set_color("#111111")

    legend = ax.get_legend()
    if legend:
        legend.set_frame_on(False)
        for label in legend.get_texts():
            label.set_fontsize(LEGEND_SIZE)
            label.set_fontweight("semibold")
            label.set_color("#111111")


def save_panel(fig, name, output_dir):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    # Preserve exact physical panel size for consistent Illustrator assembly.
    fig.set_size_inches(*FIGSIZE, forward=True)
    fig.savefig(folder / f"{name}.pdf", facecolor="white")
    fig.savefig(folder / f"{name}.svg", facecolor="white")
    fig.savefig(folder / f"{name}.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved {folder / (name + '.pdf')}")
