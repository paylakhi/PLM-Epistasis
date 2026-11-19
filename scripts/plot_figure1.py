import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.environ.get("BASE_DIR", ".")
BASE = BASE_DIR

STRUCT_FILE = os.path.join(
    BASE,
    "3d_linear_contact_all_genes_FINAL",
    "all_usable_pairs.tsv"
)

OUTDIR = os.path.join(
    BASE,
    "3d_linear_contact_all_genes_FINAL",
    "revised_figures",
    "Figure1_separate_panels"
)

os.makedirs(OUTDIR, exist_ok=True)



from publication_style import (
    FIGSIZE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
    ANNOTATION_SIZE, BLUE, ORANGE, polish_panel,
    save_panel as _save_panel,
)

def save_panel(fig, name):
    _save_panel(fig, name, OUTDIR)

# ============================================================
# LOAD STRUCTURAL DATA
# ============================================================

print("=" * 75)
print("LOADING STRUCTURAL DATA")
print("=" * 75)

df = pd.read_csv(
    STRUCT_FILE,
    sep="\t",
    usecols=[
        "gene",
        "E_cond",
        "is_contact"
    ],
    low_memory=False
)

df["E_cond"] = pd.to_numeric(
    df["E_cond"],
    errors="coerce"
)

df["is_contact"] = pd.to_numeric(
    df["is_contact"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "E_cond",
        "is_contact"
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
    f"Contacts = "
    f"{int(df['is_contact'].sum()):,}"
)
print(
    f"Overall contact rate = "
    f"{100 * df['is_contact'].mean():.4f}%"
)


# ============================================================
# PANEL A
#
# Signed epistasis quantile thresholds:
#
# negative:
# 1%, 2%, 3%, 4%, 5%, 10%
#
# middle:
# 80%
#
# positive:
# 10%, 5%, 4%, 3%, 2%, 1%
#
# IMPORTANT:
# Tail categories are intentionally nested.
# ============================================================

print("\n" + "=" * 75)
print("PANEL A: SIGNED E_cond THRESHOLDS")
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
        "contacts": int(
            sub["is_contact"].sum()
        ),
        "contact_rate": (
            sub["is_contact"].mean()
        ),
        "contact_percent": (
            100 *
            sub["is_contact"].mean()
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
    "contacts": int(
        middle["is_contact"].sum()
    ),
    "contact_rate": (
        middle["is_contact"].mean()
    ),
    "contact_percent": (
        100 *
        middle["is_contact"].mean()
    ),
    "median_E_cond": (
        middle["E_cond"].median()
    )
})


# ------------------------------------------------------------
# Positive tail
# ------------------------------------------------------------

# Reverse order deliberately:
# +10%, +5%, +4%, +3%, +2%, +1%
#
# This gives a natural left-to-right progression:
#
# strongest negative
#       ->
# middle
#       ->
# strongest positive

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
        "contacts": int(
            sub["is_contact"].sum()
        ),
        "contact_rate": (
            sub["is_contact"].mean()
        ),
        "contact_percent": (
            100 *
            sub["is_contact"].mean()
        ),
        "median_E_cond": (
            sub["E_cond"].median()
        )
    })


panelA = pd.DataFrame(rows)


# Explicit plotting order
order_A = (
    [f"Negative {x}%" for x in [1, 2, 3, 4, 5, 10]]
    +
    ["Middle 80%"]
    +
    [f"Positive {x}%" for x in [10, 5, 4, 3, 2, 1]]
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
        "Figure1A_signed_thresholds.tsv"
    ),
    sep="\t",
    index=False
)

print(
    panelA[
        [
            "group",
            "N",
            "contact_percent",
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
    panelA["contact_percent"],
    marker="o",
    markersize=7.5,
    linewidth=2.5,
    color=BLUE
)

ax.set_xticks(x)

# Shorter labels for readability
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
    r"Signed $E_{\mathrm{cond}}$ threshold"
)

ax.set_ylabel(
    "Variant pairs in structural contact (%)"
)

ax.set_title(
    "Structural contact across signed score thresholds",
)

# Separate middle visually
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
    fontsize=10
)

ax.text(
    9.5,
    0.96,
    "Positive tail",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=10
)

polish_panel(ax)
fig.tight_layout(pad=1.35)

save_panel(
    fig,
    "Figure1A_structural_signed_thresholds"
)


# ============================================================
# PANEL B
#
# Structural contact according to |E_cond| magnitude
# ============================================================

print("\n" + "=" * 75)
print("PANEL B: |E_cond| MAGNITUDE")
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
        "contacts": int(
            sub["is_contact"].sum()
        ),
        "contact_rate": (
            sub["is_contact"].mean()
        ),
        "contact_percent": (
            100 *
            sub["is_contact"].mean()
        ),
        "median_abs_E": (
            sub["abs_E"].median()
        )
    })


panelB = pd.DataFrame(rows)


panelB.to_csv(
    os.path.join(
        OUTDIR,
        "Figure1B_absE_thresholds.tsv"
    ),
    sep="\t",
    index=False
)


print(
    panelB[
        [
            "group",
            "N",
            "contact_percent",
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
    panelB["contact_percent"],
    marker="o",
    markersize=7.5,
    linewidth=2.5,
    color=BLUE
)

ax.set_xticks(x)

ax.set_xticklabels(
    panelB["group"],
    rotation=35,
    ha="right"
)

ax.set_xlabel(
    r"Most extreme $|E_{\mathrm{cond}}|$ pairs"
)

ax.set_ylabel(
    "Variant pairs in structural contact (%)"
)

ax.set_title(
    "Structural contact increases with score magnitude",
)

polish_panel(ax)
fig.tight_layout(pad=1.35)

save_panel(
    fig,
    "Figure1B_structural_absE_thresholds"
)


# ============================================================
# PANEL C
#
# Exact matched-sample gene-level CV
# ============================================================

print("\n" + "=" * 75)
print("PANEL C: HELD-OUT-GENE STRUCTURAL PREDICTION")
print("=" * 75)


panelC = pd.DataFrame({

    "model": [
        "ESM2-650M",
        "ESM2-650M",
        "ESM2-650M",
        "VESM-35M",
        "VESM-35M",
        "VESM-35M"
    ],

    "predictors": [
        "d1 + d2",
        "+ signed E",
        "+ |E|",
        "d1 + d2",
        "+ signed E",
        "+ |E|"
    ],

    "auc": [
        0.523231,
        0.604980,
        0.928910,
        0.522132,
        0.558243,
        0.912860
    ],

    "sd": [
        0.008596,
        0.005532,
        0.003080,
        0.012170,
        0.007311,
        0.002438
    ]
})


panelC.to_csv(
    os.path.join(
        OUTDIR,
        "Figure1C_gene_level_CV.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# PLOT PANEL C
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)


labels = [
    r"$d_1 + d_2$",
    r"+ signed $E_{\mathrm{cond}}$",
    r"+ $|E_{\mathrm{cond}}|$"
]


esm = (
    panelC[
        panelC["model"] == "ESM2-650M"
    ]
    .reset_index(drop=True)
)

vesm = (
    panelC[
        panelC["model"] == "VESM-35M"
    ]
    .reset_index(drop=True)
)


x = np.arange(3)

width = 0.34


bars1 = ax.bar(
    x - width / 2,
    esm["auc"],
    width,
    yerr=esm["sd"],
    capsize=4,
    label="ESM2-650M",
    color=BLUE
)


bars2 = ax.bar(
    x + width / 2,
    vesm["auc"],
    width,
    yerr=vesm["sd"],
    capsize=4,
    label="VESM-35M",
    color=ORANGE
)


ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1
)


ax.set_xticks(x)

ax.set_xticklabels(
    labels
)


ax.set_ylim(
    0.45,
    1.0
)


ax.set_xlabel(
    "Predictor set"
)

ax.set_ylabel(
    "Held-out-gene ROC-AUC"
)

ax.set_title(
    "Structural-contact prediction",
)


ax.legend(
    frameon=False
)


# Add numerical values
for bar, value in zip(
    bars1,
    esm["auc"]
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.018,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )


for bar, value in zip(
    bars2,
    vesm["auc"]
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.018,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )


polish_panel(ax)
fig.tight_layout(pad=1.35)

save_panel(
    fig,
    "Figure1C_structural_gene_CV"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)

print(
    "\nAll panels saved separately in:"
)

print(OUTDIR)
# ============================================================
# FIGURE 1D-F: Single-mutation validation (same 7 x 5 style)
# Input tables are outputs of the original ClinVar and VEP scripts.
# ============================================================
import argparse
from pathlib import Path


VEP_TABLE_DEFAULT = Path(BASE) / "vep_single_mutation_results_all_genes_FINAL" / "single_mutations_with_vep.tsv"

# The original ClinVar script uses --out-dir, so its location cannot be
# recovered from that script alone. Find the actual saved result automatically.
CLINVAR_FILENAME = "single_mutations_with_clinvar_mutation_level_all.tsv"


def _locate_table(explicit, filename, hint):
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{hint} table does not exist: {p}")
        return p

    if hint == "vep" and VEP_TABLE_DEFAULT.is_file():
        return VEP_TABLE_DEFAULT

    # First check common output folders without traversing all gene directories.
    likely = [
        Path(BASE) / "clinvar_single_mutation_results_all_genes_FINAL" / filename,
        Path(BASE) / "clinvar_single_mutation_results_all_genes" / filename,
        Path(BASE) / "single_mutation_clinvar_results_all_genes_FINAL" / filename,
        Path(BASE) / "single_mutation_clinvar_results_all_genes" / filename,
    ]
    matches = [p for p in likely if p.is_file()]
    if hint == "clinvar" and likely[0].is_file():
        return likely[0]
    # Search only top-level output directories (gene folders contain input files).
    for folder in Path(BASE).iterdir():
        if folder.is_dir() and ("clinvar" in folder.name.lower() or "single" in folder.name.lower()):
            p = folder / filename
            if p.is_file():
                matches.append(p)
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple {hint} results found; choose the final analysis using --{hint}-table:\n" +
            "\n".join(str(x) for x in matches)
        )
    raise FileNotFoundError(
        f"Could not find {filename} under {BASE}. "
        f"The uploaded {hint} analysis script does not establish a different output location. "
        f"Supply --{hint}-table /actual/path/{filename}."
    )


def _single_mutation_box(table, group_col, groups, colors, title, output_name):
    if group_col not in table.columns or 'single_score' not in table.columns:
        raise ValueError(f"Required columns missing for {output_name}: {group_col}, single_score")
    table = table[[group_col, 'single_score']].copy()
    table['single_score'] = pd.to_numeric(table['single_score'], errors='coerce')
    table = table.dropna(subset=[group_col, 'single_score'])
    values = [table.loc[table[group_col].eq(group), 'single_score'].to_numpy() for group in groups]
    if any(len(v) == 0 for v in values):
        raise ValueError(f"Missing one or more categories in {output_name}: {[(g, len(v)) for g, v in zip(groups, values)]}")

    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.65,
                     labels=groups, whis=1.5)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
        patch.set_edgecolor('#242424')
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(1.9)
    for element in box['whiskers'] + box['caps']:
        element.set_color('#222222')
        element.set_linewidth(1.5)

    ax.axhline(0, color='#777777', linestyle='--', linewidth=1.1, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel('Single-mutation score')
    ax.set_xlabel('')
    ax.set_ylim(-15.7, 6.2)
    # Position n labels at a fixed readable height, above the boxes and whiskers.
    for idx, data in enumerate(values, 1):
        ax.text(idx, 5.4, f'n={len(data)}', ha='center', va='bottom',
                fontsize=10, fontweight='medium')
    if len(groups) == 3:
        ax.tick_params(axis='x', labelsize=10)
    polish_panel(ax)
    if len(groups) == 3:
        ax.tick_params(axis='x', labelsize=10)
    fig.tight_layout(pad=1.4)
    save_panel(fig, output_name)
    print(output_name, 'counts:', dict(zip(groups, map(len, values))))


def make_validation_panels():
    parser = argparse.ArgumentParser(description='Figure 1a-f, using precomputed ClinVar and VEP tables')
    parser.add_argument('--clinvar-table', help='ClinVar mutation-level TSV from original analysis')
    parser.add_argument('--vep-table', help='VEP single_mutations_with_vep.tsv from original analysis')
    args = parser.parse_args()

    clinvar_path = _locate_table(args.clinvar_table, 'single_mutations_with_clinvar_mutation_level_all.tsv', 'clinvar')
    vep_path = _locate_table(args.vep_table, 'single_mutations_with_vep.tsv', 'vep')
    print('ClinVar source:', clinvar_path)
    print('VEP source:', vep_path)
    clinvar = pd.read_csv(clinvar_path, sep='\t', usecols=['clinvar_group', 'single_score'])
    vep = pd.read_csv(vep_path, sep='\t', usecols=['sift_group', 'polyphen_group', 'single_score'])

    _single_mutation_box(
        clinvar, 'clinvar_group',
        ['Benign/Likely benign', 'VUS/Conflicting', 'Pathogenic/Likely pathogenic'],
        ['#4C72B0', '#999999', '#C44E52'],
        'Single-mutation scores across ClinVar classes',
        'Figure1D_ClinVar'
    )
    _single_mutation_box(
        vep, 'polyphen_group', ['benign', 'probably_damaging'],
        ['#4C72B0', '#C44E52'],
        'Single-mutation scores across PolyPhen classes',
        'Figure1E_PolyPhen'
    )
    _single_mutation_box(
        vep, 'sift_group', ['tolerated', 'deleterious'],
        ['#4C72B0', '#C44E52'],
        'Single-mutation scores across SIFT classes',
        'Figure1F_SIFT'
    )


make_validation_panels()
