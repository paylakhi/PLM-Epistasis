import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.environ.get("BASE_DIR", ".")
BASE = BASE_DIR

CIS_BASE = os.path.join(
    BASE,
    "gnomad_cis_epistasis_all_genes_results"
)

REG_FILE = os.path.join(
    CIS_BASE,
    "cis_regression",
    "cis_regression_input_merged.tsv"
)

OUTDIR = os.path.join(
    CIS_BASE,
    "revised_figures",
    "Figure2_separate_panels"
)

os.makedirs(
    OUTDIR,
    exist_ok=True
)



from publication_style import (
    FIGSIZE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
    ANNOTATION_SIZE, BLUE, ORANGE, polish_panel,
    save_panel as _save_panel,
)

def save_panel(fig, name):
    _save_panel(fig, name, OUTDIR)

# ============================================================
# LOAD VALIDATED ESM2 CIS SAMPLE
# ============================================================

print("=" * 75)
print("LOADING VALIDATED CIS SAMPLE")
print("=" * 75)

df = pd.read_csv(
    REG_FILE,
    sep="\t",
    usecols=[
        "gene",
        "pair_key",
        "double_mut",
        "observed_in_cis",
        "E_cond"
    ],
    low_memory=False
)

df["E_cond"] = pd.to_numeric(
    df["E_cond"],
    errors="coerce"
)

df["observed_in_cis"] = pd.to_numeric(
    df["observed_in_cis"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "E_cond",
        "observed_in_cis"
    ]
)

df["abs_E"] = np.abs(
    df["E_cond"]
)

print(f"N = {len(df):,}")
print(
    f"Genes = {df['gene'].nunique():,}"
)
print(
    f"Cis observations = "
    f"{int(df['observed_in_cis'].sum()):,}"
)
print(
    f"Cis percentage = "
    f"{100 * df['observed_in_cis'].mean():.3f}%"
)

if len(df) != 1_630_391:
    raise ValueError(
        "Unexpected cis sample size."
    )

if int(df["observed_in_cis"].sum()) != 1_145_027:
    raise ValueError(
        "Unexpected cis count."
    )


# ============================================================
# PANEL A
#
# SIGNED E_cond THRESHOLDS
#
# Negative:
# 1, 2, 3, 4, 5, 10%
#
# Middle:
# 80%
#
# Positive:
# 10, 5, 4, 3, 2, 1%
#
#
# Tail subsets are intentionally nested.
# ============================================================

print("\n" + "=" * 75)
print("PANEL A: CIS ACROSS SIGNED E_cond THRESHOLDS")
print("=" * 75)

thresholds = [
    1,
    2,
    3,
    4,
    5,
    10
]

rows = []


# ------------------------------------------------------------
# Negative tail
# ------------------------------------------------------------

for pct in thresholds:

    cutoff = np.percentile(
        df["E_cond"],
        pct
    )

    sub = df[
        df["E_cond"] <= cutoff
    ]

    rows.append({
        "group": f"Negative {pct}%",
        "side": "negative",
        "threshold_percent": pct,
        "E_cond_cutoff": cutoff,
        "N": len(sub),
        "cis_count": int(
            sub["observed_in_cis"].sum()
        ),
        "cis_rate": (
            sub["observed_in_cis"].mean()
        ),
        "cis_percent": (
            100 *
            sub["observed_in_cis"].mean()
        ),
        "median_E_cond": (
            sub["E_cond"].median()
        )
    })


# ------------------------------------------------------------
# Middle 80%
# ------------------------------------------------------------

q10 = np.percentile(
    df["E_cond"],
    10
)

q90 = np.percentile(
    df["E_cond"],
    90
)

middle = df[
    (df["E_cond"] > q10)
    &
    (df["E_cond"] < q90)
]

rows.append({
    "group": "Middle 80%",
    "side": "middle",
    "threshold_percent": 80,
    "E_cond_cutoff": np.nan,
    "N": len(middle),
    "cis_count": int(
        middle["observed_in_cis"].sum()
    ),
    "cis_rate": (
        middle["observed_in_cis"].mean()
    ),
    "cis_percent": (
        100 *
        middle["observed_in_cis"].mean()
    ),
    "median_E_cond": (
        middle["E_cond"].median()
    )
})


# ------------------------------------------------------------
# Positive tail
# ------------------------------------------------------------

for pct in [
    10,
    5,
    4,
    3,
    2,
    1
]:

    cutoff = np.percentile(
        df["E_cond"],
        100 - pct
    )

    sub = df[
        df["E_cond"] >= cutoff
    ]

    rows.append({
        "group": f"Positive {pct}%",
        "side": "positive",
        "threshold_percent": pct,
        "E_cond_cutoff": cutoff,
        "N": len(sub),
        "cis_count": int(
            sub["observed_in_cis"].sum()
        ),
        "cis_rate": (
            sub["observed_in_cis"].mean()
        ),
        "cis_percent": (
            100 *
            sub["observed_in_cis"].mean()
        ),
        "median_E_cond": (
            sub["E_cond"].median()
        )
    })


panelA = pd.DataFrame(rows)


order_A = (
    [
        f"Negative {x}%"
        for x in [1, 2, 3, 4, 5, 10]
    ]
    +
    ["Middle 80%"]
    +
    [
        f"Positive {x}%"
        for x in [10, 5, 4, 3, 2, 1]
    ]
)

panelA["group"] = pd.Categorical(
    panelA["group"],
    categories=order_A,
    ordered=True
)

panelA = (
    panelA
    .sort_values("group")
    .reset_index(drop=True)
)


panelA.to_csv(
    os.path.join(
        OUTDIR,
        "Figure2A_signed_thresholds.tsv"
    ),
    sep="\t",
    index=False
)


print(
    panelA[
        [
            "group",
            "N",
            "cis_count",
            "cis_percent",
            "median_E_cond"
        ]
    ].to_string(index=False)
)


# ============================================================
# PLOT PANEL A
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)

x = np.arange(
    len(panelA)
)

ax.plot(
    x,
    panelA["cis_percent"],
    marker="o",
    markersize=7.5,
    linewidth=2.5,
    color=BLUE
)

ax.set_xticks(x)

labels_A = [
    "Neg.\n1%",
    "Neg.\n2%",
    "Neg.\n3%",
    "Neg.\n4%",
    "Neg.\n5%",
    "Neg.\n10%",
    "Middle\n80%",
    "Pos.\n10%",
    "Pos.\n5%",
    "Pos.\n4%",
    "Pos.\n3%",
    "Pos.\n2%",
    "Pos.\n1%"
]

ax.set_xticklabels(
    labels_A
)

ax.set_xlabel(
    r"Signed $E_{\mathrm{cond}}$ threshold",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Variant pairs observed in cis (%)",
    fontsize=LABEL_SIZE
)

ax.set_title(
    "Cis observation across signed score thresholds",
)


# Visually separate middle 80%
ax.axvline(
    5.5,
    linestyle="--",
    linewidth=1,
    alpha=0.4
)

ax.axvline(
    6.5,
    linestyle="--",
    linewidth=1,
    alpha=0.4
)


ax.text(
    2.5,
    0.96,
    "Negative tail",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=11
)

ax.text(
    9.5,
    0.96,
    "Positive tail",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=11
)


polish_panel(ax)
fig.tight_layout(pad=1.35)

save_panel(
    fig,
    "Figure2A_cis_signed_thresholds"
)


# ============================================================
# PANEL B
#
# CIS OBSERVATION ACROSS |E_cond| THRESHOLDS
#
# Same thresholds as revised Figure 1B:
#
# Top 10, 5, 4, 3, 2, 1%
# ============================================================

print("\n" + "=" * 75)
print("PANEL B: CIS ACROSS |E_cond| THRESHOLDS")
print("=" * 75)


abs_thresholds = [
    10,
    5,
    4,
    3,
    2,
    1
]

rows = []


for pct in abs_thresholds:

    cutoff = np.percentile(
        df["abs_E"],
        100 - pct
    )

    sub = df[
        df["abs_E"] >= cutoff
    ]

    rows.append({
        "group": f"Top {pct}%",
        "top_percent": pct,
        "abs_E_cutoff": cutoff,
        "N": len(sub),
        "cis_count": int(
            sub["observed_in_cis"].sum()
        ),
        "cis_rate": (
            sub["observed_in_cis"].mean()
        ),
        "cis_percent": (
            100 *
            sub["observed_in_cis"].mean()
        ),
        "median_abs_E": (
            sub["abs_E"].median()
        )
    })


panelB = pd.DataFrame(rows)


panelB.to_csv(
    os.path.join(
        OUTDIR,
        "Figure2B_absE_thresholds.tsv"
    ),
    sep="\t",
    index=False
)


print(
    panelB[
        [
            "group",
            "N",
            "cis_count",
            "cis_percent",
            "median_abs_E"
        ]
    ].to_string(index=False)
)


# ============================================================
# PLOT PANEL B
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)

x = np.arange(
    len(panelB)
)

ax.plot(
    x,
    panelB["cis_percent"],
    marker="o",
    markersize=7.5,
    linewidth=2.5,
    color=BLUE
)

ax.set_xticks(x)

ax.set_xticklabels(
    panelB["group"]
)

ax.set_xlabel(
    r"Most extreme $|E_{\mathrm{cond}}|$ pairs",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Variant pairs observed in cis (%)",
    fontsize=LABEL_SIZE
)

ax.set_title(
    "Cis observation across score-magnitude thresholds",
)

polish_panel(ax)
fig.tight_layout(pad=1.35)

save_panel(
    fig,
    "Figure2B_cis_absE_thresholds"
)


# ============================================================
# PANEL C
#
# ABSOLUTE E_cond ASSOCIATION
#
# Both PLMs:
# unadjusted
# + d1/d2
# + d1/d2 + AF
# ============================================================

print("\n" + "=" * 75)
print("PANEL C: |E_cond| ODDS RATIOS")
print("=" * 75)


panelC = pd.DataFrame({

    "PLM": [
        "ESM2-650M",
        "ESM2-650M",
        "ESM2-650M",
        "VESM-35M",
        "VESM-35M",
        "VESM-35M"
    ],

    "model": [
        "Unadjusted",
        "+ d1/d2",
        "+ d1/d2 + AF",
        "Unadjusted",
        "+ d1/d2",
        "+ d1/d2 + AF"
    ],

    "OR": [
        0.943086,
        0.949998,
        0.975257,
        0.944358,
        0.951585,
        0.980751
    ],

    "CI_low": [
        0.939880,
        0.946797,
        0.972065,
        0.941242,
        0.948454,
        0.977562
    ],

    "CI_high": [
        0.946302,
        0.953210,
        0.978460,
        0.947484,
        0.954726,
        0.983951
    ],

    "P": [
        1.8e-249,
        5.668e-195,
        1.035e-50,
        1.145e-252,
        1.716e-191,
        1.349e-31
    ]
})


panelC.to_csv(
    os.path.join(
        OUTDIR,
        "Figure2C_absE_ORs.tsv"
    ),
    sep="\t",
    index=False
)


print(
    panelC.to_string(index=False)
)


# ============================================================
# PLOT PANEL C
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)


model_order = [
    "Unadjusted",
    "+ d1/d2",
    "+ d1/d2 + AF"
]

y_base = np.arange(
    len(model_order)
)

offset = 0.10


for i, plm in enumerate(
    [
        "ESM2-650M",
        "VESM-35M"
    ]
):

    tmp = (
        panelC[
            panelC["PLM"] == plm
        ]
        .set_index("model")
        .loc[model_order]
        .reset_index()
    )

    y = (
        y_base
        +
        (-offset if i == 0 else offset)
    )

    xerr = np.vstack([
        tmp["OR"] - tmp["CI_low"],
        tmp["CI_high"] - tmp["OR"]
    ])

    ax.errorbar(
        tmp["OR"],
        y,
        xerr=xerr,
        fmt="o",
        markersize=7.5,
        capsize=4,
        linewidth=1.9,
        label=plm,
        color=BLUE if i == 0 else ORANGE
    )


ax.axvline(
    1.0,
    linestyle="--",
    linewidth=1
)


ax.set_yticks(
    y_base
)

ax.set_yticklabels(
    model_order
)

ax.invert_yaxis()


ax.set_xlabel(
    r"OR per 1 SD higher $|E_{\mathrm{cond}}|$",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Regression model",
    fontsize=LABEL_SIZE
)

ax.set_title(
    "Association between score magnitude and cis observation",
)

ax.legend(
    frameon=False
)


polish_panel(ax)
fig.tight_layout(pad=1.35)

save_panel(
    fig,
    "Figure2C_absE_ORs"
)


# ============================================================
# PANEL D
#
# CIS-OBSERVATION PREDICTION
# Five-fold held-out-gene ROC-AUC (mean +/- SD across folds).
# Values are read directly from the completed CV summary.
# ============================================================

print("\n" + "=" * 75)
print("PANEL D: CIS-OBSERVATION PREDICTION (GENE-LEVEL CV)")
print("=" * 75)

CV_SUMMARY = os.path.join(
    CIS_BASE,
    "michel_cis_regression",
    "cis_gene_cv_from_gene_files",
    "cis_gene_cv_summary.tsv"
)

cv = pd.read_csv(CV_SUMMARY, sep="\t")
expected_models = [
    "Singles + AF",
    "Singles + AF + signed E",
    "Singles + AF + abs E"
]
expected_plms = ["ESM2", "VESM"]
cv = cv[cv["plm"].isin(expected_plms) & cv["model"].isin(expected_models)].copy()
if len(cv) != 6 or cv.duplicated(["plm", "model"]).any():
    raise ValueError("Panel D requires exactly six unique PLM/model CV results")
if not (pd.to_numeric(cv["n_folds"], errors="coerce") == 5).all():
    raise ValueError("Panel D requires five CV folds for every model")
for col in ["mean_auc", "sd_auc"]:
    cv[col] = pd.to_numeric(cv[col], errors="raise")
if not np.isfinite(cv[["mean_auc", "sd_auc"]].to_numpy()).all():
    raise ValueError("Nonfinite CV AUC or SD")

cv = cv.set_index(["plm", "model"])
panelD = pd.DataFrame({
    "model": ["d1 + d2 + AF", "+ signed E_cond", "+ |E_cond|"],
    "ESM2-650M": [cv.loc[("ESM2", m), "mean_auc"] for m in expected_models],
    "ESM2-650M_SD": [cv.loc[("ESM2", m), "sd_auc"] for m in expected_models],
    "VESM-35M": [cv.loc[("VESM", m), "mean_auc"] for m in expected_models],
    "VESM-35M_SD": [cv.loc[("VESM", m), "sd_auc"] for m in expected_models]
})

panelD.to_csv(
    os.path.join(OUTDIR, "Figure2D_cis_prediction_AUC.tsv"),
    sep="\t",
    index=False
)
print(panelD.to_string(index=False))

# ============================================================
# PLOT PANEL D
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)

x = np.arange(len(panelD))
width = 0.34

bars_esm2 = ax.bar(
    x - width / 2,
    panelD["ESM2-650M"],
    width=width,
    yerr=panelD["ESM2-650M_SD"],
    error_kw={"ecolor": "black", "capsize": 4, "elinewidth": 1.5, "capthick": 1.5},
    label="ESM2-650M",
    color=BLUE
)

bars_vesm = ax.bar(
    x + width / 2,
    panelD["VESM-35M"],
    width=width,
    yerr=panelD["VESM-35M_SD"],
    error_kw={"ecolor": "black", "capsize": 4, "elinewidth": 1.5, "capthick": 1.5},
    label="VESM-35M",
    color=ORANGE
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1
)

ax.set_xticks(x)

ax.set_xticklabels([
    r"$d_1 + d_2 + \mathrm{AF}$",
    r"+ signed $E_{\mathrm{cond}}$",
    r"+ $|E_{\mathrm{cond}}|$"
])

ax.set_xlabel(
    "Predictor set",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Five-fold gene-level CV ROC-AUC",
    fontsize=LABEL_SIZE
)

ax.set_title(
    "Cis-observation prediction",
)

# Same broad AUC scale as Figure 1C so the structural-vs-cis
# predictive contrast remains visually direct.
ax.set_ylim(
    0.45,
    1.00
)

ax.legend(
    frameon=False
)

# Value labels above the SD error bars, analogous to Figure 1C.
for bars, sd_col in [
    (bars_esm2, "ESM2-650M_SD"),
    (bars_vesm, "VESM-35M_SD")
]:
    for bar, sd in zip(bars, panelD[sd_col]):
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + sd + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE
        )

polish_panel(ax)
fig.tight_layout(pad=1.35)

save_panel(
    fig,
    "Figure2D_cis_prediction_AUC"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)

print(
    "\nAll Figure 2 panels saved separately in:"
)

print(OUTDIR)

print(
    "\nExpected files:"
)

print(
    "Figure2A_cis_signed_thresholds.pdf/png"
)

print(
    "Figure2B_cis_absE_thresholds.pdf/png"
)

print(
    "Figure2C_absE_ORs.pdf/png"
)

print(
    "Figure2D_cis_prediction_AUC.pdf/png"
)
