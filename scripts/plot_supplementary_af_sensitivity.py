#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.environ.get("BASE_DIR", ".")
BASE = os.path.join(BASE_DIR, "gnomad_cis_epistasis_all_genes_results")

ESM_DIR = os.path.join(
    BASE,
    "af_product_sensitivity"
)

VESM_DIR = os.path.join(
    BASE,
    "af_product_sensitivity_VESM35M"
)

OUT_DIR = os.path.join(
    BASE,
    "supplementary_figures"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

# ESM2 analytic dataset used for AF-product distribution
ESM_ANALYTIC = os.path.join(
    ESM_DIR,
    "AF_product_sensitivity_analytic_dataset.tsv"
)

# ESM2 stratified regression results
ESM_RESULTS = os.path.join(
    ESM_DIR,
    "AF_product_stratified_logistic_regression.tsv"
)

# VESM-35M stratified regression results
VESM_RESULTS = os.path.join(
    VESM_DIR,
    "VESM35M_AF_product_stratified_logistic_regression.tsv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

FIG2A_PNG = os.path.join(
    OUT_DIR,
    "Supplementary_Fig2a_AF_product_distribution.png"
)

FIG2A_PDF = os.path.join(
    OUT_DIR,
    "Supplementary_Fig2a_AF_product_distribution.pdf"
)

FIG2B_PNG = os.path.join(
    OUT_DIR,
    "Supplementary_Fig2b_AF_stratified_forest.png"
)

FIG2B_PDF = os.path.join(
    OUT_DIR,
    "Supplementary_Fig2b_AF_stratified_forest.pdf"
)


# ============================================================
# GLOBAL PLOT SETTINGS
# ============================================================

FIGSIZE = (7, 5)

LABEL_SIZE = 12
TITLE_SIZE = 14
TICK_SIZE = 10
LEGEND_SIZE = 10

plt.rcParams.update({

    "font.family": "sans-serif",

    "font.size": 10,

    "axes.labelsize": LABEL_SIZE,

    "axes.titlesize": TITLE_SIZE,

    "xtick.labelsize": TICK_SIZE,

    "ytick.labelsize": TICK_SIZE,

    "legend.fontsize": LEGEND_SIZE,

    "axes.linewidth": 1.0,

    # Keep text editable in Illustrator
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# HELPER FUNCTION
# MAKE ORDINARY TICK LABELS BOLD
# ============================================================

def make_ticks_bold(ax):

    for label in ax.get_xticklabels():

        label.set_fontweight("bold")
        label.set_fontsize(TICK_SIZE)

    for label in ax.get_yticklabels():

        label.set_fontweight("bold")
        label.set_fontsize(TICK_SIZE)


# ============================================================
# CHECK INPUT FILES
# ============================================================

for file_path in [
    ESM_ANALYTIC,
    ESM_RESULTS,
    VESM_RESULTS
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nFile not found:\n{file_path}\n"
        )


# ============================================================
# ============================================================
# SUPPLEMENTARY FIGURE 2A
# AF-PRODUCT DISTRIBUTION
# ============================================================
# ============================================================

print("\n========================================")
print("CREATING SUPPLEMENTARY FIG. 2A")
print("========================================")


# ============================================================
# LOAD AF-PRODUCT DATA
# ============================================================

df = pd.read_csv(
    ESM_ANALYTIC,
    sep="\t",
    usecols=["AF_product"],
    low_memory=False
)


df["AF_product"] = pd.to_numeric(
    df["AF_product"],
    errors="coerce"
)


df = df[
    df["AF_product"].notna()
    &
    np.isfinite(df["AF_product"])
    &
    (df["AF_product"] > 0)
].copy()


df["log10_AF_product"] = np.log10(
    df["AF_product"]
)


print(
    "Number of variant pairs:",
    len(df)
)

print(
    "Minimum log10(AF1 x AF2):",
    df["log10_AF_product"].min()
)

print(
    "Maximum log10(AF1 x AF2):",
    df["log10_AF_product"].max()
)


# ============================================================
# CREATE FIGURE 2A
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)


# ============================================================
# HISTOGRAM
# ============================================================

ax.hist(
    df["log10_AF_product"],
    bins=80,
    edgecolor="none"
)


# ============================================================
# AF-PRODUCT BIN BOUNDARIES
# ============================================================

for boundary in [-8, -7, -6]:

    ax.axvline(
        boundary,
        linestyle="--",
        linewidth=1.4
    )


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    r"log$_{10}$(AF$_1$ × AF$_2$)",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Number of variant pairs",
    fontsize=LABEL_SIZE
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Distribution of joint allele-frequency product",
    fontsize=TITLE_SIZE,
    pad=12
)


# ============================================================
# TICK FORMATTING
# ============================================================

ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE
)


# Scientific notation on y-axis
ax.ticklabel_format(
    axis="y",
    style="sci",
    scilimits=(0, 0)
)


# Make scientific-notation multiplier bold
ax.yaxis.get_offset_text().set_fontsize(
    TICK_SIZE
)

ax.yaxis.get_offset_text().set_fontweight(
    "bold"
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    axis="y",
    alpha=0.20,
    linewidth=0.8
)

ax.set_axisbelow(True)


# ============================================================
# REMOVE TOP / RIGHT SPINES
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# AF-PRODUCT STRATUM LABELS
#
# Entire mathematical expressions are explicitly bold.
# This avoids the problem where only "AF1 x AF2" appears bold.
# ============================================================

ymax = ax.get_ylim()[1]

label_y = ymax * 0.91


stratum_annotations = [

    (
        -9.35,
        r"$\mathbf{<}\mathbf{10}^{\mathbf{-8}}$"
    ),

    (
        -7.50,
        r"$\mathbf{10}^{\mathbf{-8}}"
        r"\mathbf{-}"
        r"\mathbf{10}^{\mathbf{-7}}$"
    ),

    (
        -6.50,
        r"$\mathbf{10}^{\mathbf{-7}}"
        r"\mathbf{-}"
        r"\mathbf{10}^{\mathbf{-6}}$"
    ),

    (
        -4.60,
        r"$\mathbf{\geq}\mathbf{10}^{\mathbf{-6}}$"
    ),
]


for xpos, text in stratum_annotations:

    ax.text(
        xpos,
        label_y,
        text,
        ha="center",
        va="top",
        fontsize=10
    )


# ============================================================
# MAKE X/Y TICKS BOLD
# ============================================================

make_ticks_bold(ax)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE FIGURE 2A
# ============================================================

fig.savefig(
    FIG2A_PNG,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    FIG2A_PDF,
    bbox_inches="tight"
)

plt.close(fig)


print("\nSaved Supplementary Fig. 2a:")
print(FIG2A_PNG)
print(FIG2A_PDF)


# ============================================================
# ============================================================
# SUPPLEMENTARY FIGURE 2B
# ESM2 + VESM FOREST PLOT
# ============================================================
# ============================================================

print("\n========================================")
print("CREATING SUPPLEMENTARY FIG. 2B")
print("========================================")


# ============================================================
# LOAD RESULTS
# ============================================================

esm = pd.read_csv(
    ESM_RESULTS,
    sep="\t"
)

vesm = pd.read_csv(
    VESM_RESULTS,
    sep="\t"
)


# ============================================================
# REMOVE OVERALL RESULT
#
# Overall results are already presented in main Figure 2.
# ============================================================

esm = esm[
    esm["AF_product_bin"].astype(str) != "Overall"
].copy()


vesm = vesm[
    vesm["AF_product_bin"].astype(str) != "Overall"
].copy()


# ============================================================
# NORMALIZE BIN LABELS
# ============================================================

def normalize_bin(x):

    x = str(x).strip()

    mapping = {

        "<1e-8":
            "<1e-8",

        "1e-8–<1e-7":
            "1e-8–<1e-7",

        "1e-8-<1e-7":
            "1e-8–<1e-7",

        "1e-7–<1e-6":
            "1e-7–<1e-6",

        "1e-7-<1e-6":
            "1e-7–<1e-6",

        ">=1e-6":
            ">=1e-6",
    }

    return mapping.get(
        x,
        x
    )


esm["AF_product_bin"] = (
    esm["AF_product_bin"]
    .map(normalize_bin)
)


vesm["AF_product_bin"] = (
    vesm["AF_product_bin"]
    .map(normalize_bin)
)


BIN_ORDER = [

    "<1e-8",

    "1e-8–<1e-7",

    "1e-7–<1e-6",

    ">=1e-6"
]


# ============================================================
# VERIFY THAT ALL FOUR BINS ARE PRESENT
# ============================================================

for model_name, tmp in [

    ("ESM2-650M", esm),

    ("VESM-35M", vesm)

]:

    missing = [

        x
        for x in BIN_ORDER
        if x not in set(
            tmp["AF_product_bin"]
        )
    ]

    if missing:

        raise ValueError(
            f"{model_name} missing bins: {missing}"
        )


# ============================================================
# ORDER BINS
# ============================================================

esm["AF_product_bin"] = pd.Categorical(
    esm["AF_product_bin"],
    categories=BIN_ORDER,
    ordered=True
)


vesm["AF_product_bin"] = pd.Categorical(
    vesm["AF_product_bin"],
    categories=BIN_ORDER,
    ordered=True
)


esm = (
    esm
    .sort_values("AF_product_bin")
    .reset_index(drop=True)
)


vesm = (
    vesm
    .sort_values("AF_product_bin")
    .reset_index(drop=True)
)


# ============================================================
# EXTRACT ESM2 RESULTS
# ============================================================

esm_or = (
    esm["odds_ratio"]
    .astype(float)
    .to_numpy()
)


esm_low = (
    esm["OR_ci_low"]
    .astype(float)
    .to_numpy()
)


esm_high = (
    esm["OR_ci_high"]
    .astype(float)
    .to_numpy()
)


# ============================================================
# EXTRACT VESM RESULTS
# ============================================================

vesm_or = (
    vesm["odds_ratio"]
    .astype(float)
    .to_numpy()
)


vesm_low = (
    vesm["OR_ci_low"]
    .astype(float)
    .to_numpy()
)


vesm_high = (
    vesm["OR_ci_high"]
    .astype(float)
    .to_numpy()
)


# ============================================================
# PRINT RESULTS USED IN FIGURE
# ============================================================

print("\nValues used in Supplementary Fig. 2b:")


for i, bin_name in enumerate(BIN_ORDER):

    print(
        f"\n{bin_name}"
    )

    print(
        "ESM2-650M: "
        f"OR={esm_or[i]:.6f}, "
        f"95% CI="
        f"{esm_low[i]:.6f}–"
        f"{esm_high[i]:.6f}"
    )

    print(
        "VESM-35M: "
        f"OR={vesm_or[i]:.6f}, "
        f"95% CI="
        f"{vesm_low[i]:.6f}–"
        f"{vesm_high[i]:.6f}"
    )


# ============================================================
# CREATE FIGURE 2B
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)


# ============================================================
# Y POSITIONS
# ============================================================

base_y = np.array([
    3,
    2,
    1,
    0
], dtype=float)


# Slight vertical separation between models
offset = 0.11


esm_y = base_y + offset

vesm_y = base_y - offset


# ============================================================
# ESM2-650M
# ============================================================

ax.errorbar(

    esm_or,
    esm_y,

    xerr=[
        esm_or - esm_low,
        esm_high - esm_or
    ],

    fmt="o",

    markersize=7,

    linewidth=1.6,

    capsize=3.5,

    capthick=1.3,

    label="ESM2-650M"
)


# ============================================================
# VESM-35M
# ============================================================

ax.errorbar(

    vesm_or,
    vesm_y,

    xerr=[
        vesm_or - vesm_low,
        vesm_high - vesm_or
    ],

    fmt="s",

    markersize=6.5,

    linewidth=1.6,

    capsize=3.5,

    capthick=1.3,

    label="VESM-35M"
)


# ============================================================
# NULL REFERENCE LINE
# ============================================================

ax.axvline(
    1.0,
    linestyle="--",
    linewidth=1.4
)


# ============================================================
# Y-AXIS TICK LABELS
#
# IMPORTANT:
# Entire expression is placed inside math mode.
# Every component is explicitly bold.
#
# This makes:
# AF
# subscripts 1 and 2
# multiplication sign
# inequalities
# 10
# exponents
#
# all bold.
# ============================================================

pretty_labels = [

    (
        r"$\mathbf{AF}_{\mathbf{1}}\ "
        r"\mathbf{\times}\ "
        r"\mathbf{AF}_{\mathbf{2}}\ "
        r"\mathbf{<}\ "
        r"\mathbf{10}^{\mathbf{-8}}$"
    ),

    (
        r"$\mathbf{10}^{\mathbf{-8}}\ "
        r"\mathbf{\leq}\ "
        r"\mathbf{AF}_{\mathbf{1}}\ "
        r"\mathbf{\times}\ "
        r"\mathbf{AF}_{\mathbf{2}}\ "
        r"\mathbf{<}\ "
        r"\mathbf{10}^{\mathbf{-7}}$"
    ),

    (
        r"$\mathbf{10}^{\mathbf{-7}}\ "
        r"\mathbf{\leq}\ "
        r"\mathbf{AF}_{\mathbf{1}}\ "
        r"\mathbf{\times}\ "
        r"\mathbf{AF}_{\mathbf{2}}\ "
        r"\mathbf{<}\ "
        r"\mathbf{10}^{\mathbf{-6}}$"
    ),

    (
        r"$\mathbf{AF}_{\mathbf{1}}\ "
        r"\mathbf{\times}\ "
        r"\mathbf{AF}_{\mathbf{2}}\ "
        r"\mathbf{\geq}\ "
        r"\mathbf{10}^{\mathbf{-6}}$"
    ),
]


ax.set_yticks(
    base_y
)


ax.set_yticklabels(
    pretty_labels,
    fontsize=TICK_SIZE
)


# ============================================================
# SYMMETRIC X-AXIS AROUND OR = 1
# ============================================================

all_low = np.concatenate([
    esm_low,
    vesm_low
])


all_high = np.concatenate([
    esm_high,
    vesm_high
])


max_distance = max(

    1.0 - all_low.min(),

    all_high.max() - 1.0
)


padding = 0.003


xmin = (
    1.0
    -
    max_distance
    -
    padding
)


xmax = (
    1.0
    +
    max_distance
    +
    padding
)


ax.set_xlim(
    xmin,
    xmax
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Odds ratio for cis observation\n"
    "(per 1 SD increase in epistatic score)",
    fontsize=LABEL_SIZE
)


ax.set_ylabel(
    "Allele-frequency product stratum",
    fontsize=LABEL_SIZE
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Epistatic-score association within AF-product strata",
    fontsize=TITLE_SIZE,
    pad=12
)


# ============================================================
# X-AXIS FORMAT
# ============================================================

ax.xaxis.set_major_formatter(
    FuncFormatter(
        lambda x, pos:
        f"{x:.3f}"
    )
)


ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE
)


# ============================================================
# MAKE ORDINARY TICK TEXT BOLD
#
# For y-axis mathematical labels, boldness is already
# explicitly defined using \mathbf{} above.
# ============================================================

make_ticks_bold(ax)


# ============================================================
# GRID
# ============================================================

ax.grid(
    axis="x",
    alpha=0.20,
    linewidth=0.8
)


ax.set_axisbelow(True)


# ============================================================
# REMOVE TOP / RIGHT SPINES
# ============================================================

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    frameon=False,
    loc="upper right",
    fontsize=LEGEND_SIZE
)


# ============================================================
# Y LIMITS
# ============================================================

ax.set_ylim(
    -0.50,
    3.50
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE FIGURE 2B
# ============================================================

fig.savefig(
    FIG2B_PNG,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    FIG2B_PDF,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# OUTPUT MESSAGE
# ============================================================

print("\nSaved Supplementary Fig. 2b:")

print(FIG2B_PNG)

print(FIG2B_PDF)


print("\n========================================")
print("DONE")
print("========================================")


print("\nSupplementary Fig. 2a:")
print(FIG2A_PDF)


print("\nSupplementary Fig. 2b:")
print(FIG2B_PDF)
