import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.environ.get("BASE_DIR", ".")
BASE = BASE_DIR

CIS_BASE = os.path.join(
    BASE,
    "gnomad_cis_epistasis_all_genes_results"
)

REG_INPUT = os.path.join(
    CIS_BASE,
    "cis_regression",
    "cis_regression_input_merged.tsv"
)

HAP_FILE = os.path.join(
    CIS_BASE,
    "merged_all_pairs_with_qc_REVISED.tsv"
)

OUT_DIR = os.path.join(
    CIS_BASE,
    "cis_regression"
)

OUT_RESULTS = os.path.join(
    OUT_DIR,
    "VESM35M_abs_Econd_cis_FINAL_results.tsv"
)


# ============================================================
# 1. LOAD VALIDATED CIS SAMPLE
# ============================================================

print("=" * 75)
print("LOADING VALIDATED CIS SAMPLE")
print("=" * 75)

reg = pd.read_csv(
    REG_INPUT,
    sep="\t",
    usecols=[
        "gene",
        "pair_key",
        "double_mut",
        "observed_in_cis"
    ],
    low_memory=False
)

reg["observed_in_cis"] = pd.to_numeric(
    reg["observed_in_cis"],
    errors="coerce"
)

print(f"N: {len(reg):,}")
print(f"Genes: {reg['gene'].nunique():,}")
print(
    f"Cis: {int(reg['observed_in_cis'].sum()):,}"
)
print(
    f"Cis %: {100 * reg['observed_in_cis'].mean():.3f}%"
)

if len(reg) != 1_630_391:
    raise ValueError("Validated sample N mismatch.")

if int(reg["observed_in_cis"].sum()) != 1_145_027:
    raise ValueError("Validated cis count mismatch.")

if reg.duplicated(
    ["gene", "pair_key"]
).sum() != 0:
    raise ValueError("Duplicate regression pair_key.")

if reg.duplicated(
    ["gene", "double_mut"]
).sum() != 0:
    raise ValueError("Duplicate regression double_mut.")


# ============================================================
# 2. LOAD VESM SCORES
# ============================================================

print("\n" + "=" * 75)
print("LOADING VESM-35M SCORES")
print("=" * 75)

files = glob.glob(
    os.path.join(
        BASE,
        "*",
        "*_epistasis_VESM_35M.tsv"
    )
)

files = [
    f for f in files
    if "skipped_rows" not in os.path.basename(f)
]

print(f"VESM files: {len(files):,}")

parts = []

for i, f in enumerate(files, 1):

    tmp = pd.read_csv(
        f,
        sep="\t",
        usecols=[
            "gene",
            "double_mut",
            "d1",
            "d2",
            "E_cond"
        ],
        low_memory=False
    )

    parts.append(tmp)

    if i % 1000 == 0:
        print(
            f"Loaded {i:,} / {len(files):,}"
        )

vesm = pd.concat(
    parts,
    ignore_index=True
)

del parts

print(f"VESM rows: {len(vesm):,}")
print(f"VESM genes: {vesm['gene'].nunique():,}")

if vesm.duplicated(
    ["gene", "double_mut"]
).sum() != 0:
    raise ValueError("Duplicate VESM keys.")


# ============================================================
# 3. MERGE VESM INTO VALIDATED SAMPLE
# ============================================================

df = reg.merge(
    vesm,
    on=[
        "gene",
        "double_mut"
    ],
    how="inner",
    validate="one_to_one"
)

print("\n" + "=" * 75)
print("VESM MERGE")
print("=" * 75)

print(f"Merged N: {len(df):,}")
print(f"Genes: {df['gene'].nunique():,}")
print(
    f"Cis: {int(df['observed_in_cis'].sum()):,}"
)
print(
    f"Cis %: "
    f"{100 * df['observed_in_cis'].mean():.3f}%"
)

if len(df) != 1_630_391:
    raise ValueError(
        "STOP: VESM does not cover exact sample."
    )

if int(df["observed_in_cis"].sum()) != 1_145_027:
    raise ValueError(
        "STOP: cis outcome changed."
    )

print(
    "PASS: exact validated cis sample "
    "covered by VESM."
)


# ============================================================
# 4. LOAD HAPLOTYPE INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("LOADING VALIDATED HAPLOTYPE INFORMATION")
print("=" * 75)

hap = pd.read_csv(
    HAP_FILE,
    sep="\t",
    usecols=[
        "gene",
        "pair_key",
        "all_hap_counts_0",
        "all_hap_counts_1",
        "all_hap_counts_2",
        "all_hap_counts_3",
        "total_haplotypes",
        "cis_count",
        "observed_in_cis"
    ],
    low_memory=False
)

print(f"Full hap rows: {len(hap):,}")


# ============================================================
# 5. RESTRICT HAP TABLE TO EXACT VALIDATED KEYS
# ============================================================

keys = df[
    ["gene", "pair_key"]
].drop_duplicates()

hap = hap.merge(
    keys,
    on=[
        "gene",
        "pair_key"
    ],
    how="inner"
)

print(
    f"Hap rows after restriction: {len(hap):,}"
)

dup = hap.duplicated(
    ["gene", "pair_key"]
).sum()

print(
    f"Duplicate hap keys after restriction: {dup:,}"
)

if dup != 0:
    raise ValueError(
        "Duplicate hap keys remain."
    )

if len(hap) != 1_630_391:
    raise ValueError(
        f"Expected 1,630,391 hap rows; "
        f"got {len(hap):,}"
    )


# ============================================================
# 6. VERIFY CIS OUTCOME AGREES
# ============================================================

check = df[
    [
        "gene",
        "pair_key",
        "observed_in_cis"
    ]
].merge(
    hap[
        [
            "gene",
            "pair_key",
            "observed_in_cis"
        ]
    ],
    on=[
        "gene",
        "pair_key"
    ],
    how="inner",
    suffixes=(
        "_reg",
        "_hap"
    ),
    validate="one_to_one"
)

mismatch = (
    check["observed_in_cis_reg"]
    !=
    check["observed_in_cis_hap"]
).sum()

print(
    f"Cis outcome mismatches: {mismatch:,}"
)

if mismatch != 0:
    raise ValueError(
        "Cis outcome mismatch between files."
    )

print(
    "PASS: cis outcome agrees exactly."
)


# ============================================================
# 7. CALCULATE AF PRODUCT
# ============================================================

count_cols = [
    "all_hap_counts_0",
    "all_hap_counts_1",
    "all_hap_counts_2",
    "all_hap_counts_3",
    "total_haplotypes"
]

for c in count_cols:
    hap[c] = pd.to_numeric(
        hap[c],
        errors="coerce"
    )


# Recalculate total as diagnostic
hap["total_calc"] = (
    hap["all_hap_counts_0"]
    + hap["all_hap_counts_1"]
    + hap["all_hap_counts_2"]
    + hap["all_hap_counts_3"]
)

total_diff = np.abs(
    hap["total_calc"]
    -
    hap["total_haplotypes"]
)

print(
    f"Max total-haplotype difference: "
    f"{total_diff.max()}"
)

if total_diff.max() > 1e-6:
    raise ValueError(
        "Stored and calculated haplotype totals differ."
    )

if (hap["total_calc"] <= 0).any():
    raise ValueError(
        "Zero total haplotypes."
    )


# Haplotype coding:
# 0 = ref/ref
# 1 = ref/alt
# 2 = alt/ref
# 3 = alt/alt

hap["AF1"] = (
    hap["all_hap_counts_2"]
    +
    hap["all_hap_counts_3"]
) / hap["total_calc"]

hap["AF2"] = (
    hap["all_hap_counts_1"]
    +
    hap["all_hap_counts_3"]
) / hap["total_calc"]

hap["AF_product"] = (
    hap["AF1"]
    *
    hap["AF2"]
)


# ============================================================
# 8. MERGE AF INTO ANALYSIS DATA
# ============================================================

df = df.merge(
    hap[
        [
            "gene",
            "pair_key",
            "AF1",
            "AF2",
            "AF_product"
        ]
    ],
    on=[
        "gene",
        "pair_key"
    ],
    how="left",
    validate="one_to_one"
)

if df["AF_product"].isna().any():
    raise ValueError(
        "Missing AF-product after merge."
    )


# ============================================================
# 9. AF-PRODUCT STRATA
# ============================================================

df["AF_bin"] = pd.cut(
    df["AF_product"],
    bins=[
        -np.inf,
        1e-8,
        1e-7,
        1e-6,
        np.inf
    ],
    labels=[
        "<1e-8",
        "1e-8-<1e-7",
        "1e-7-<1e-6",
        ">=1e-6"
    ],
    right=False
)

counts = (
    df["AF_bin"]
    .value_counts(sort=False)
)

print("\nAF-product strata:")
print(counts)


expected = {
    "<1e-8": 288212,
    "1e-8-<1e-7": 376794,
    "1e-7-<1e-6": 545383,
    ">=1e-6": 420002
}

for label, n in expected.items():

    observed = int(
        counts.loc[label]
    )

    if observed != n:
        raise ValueError(
            f"AF-bin mismatch: {label}: "
            f"expected {n:,}, got {observed:,}"
        )

print(
    "PASS: AF strata exactly reproduce "
    "ESM2 analysis."
)


# ============================================================
# 10. PREPARE PLM VARIABLES
# ============================================================

for c in [
    "d1",
    "d2",
    "E_cond"
]:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )

df["E_signed"] = df["E_cond"]

df["E_abs"] = np.abs(
    df["E_cond"]
)


required = [
    "observed_in_cis",
    "d1",
    "d2",
    "E_signed",
    "E_abs",
    "AF_bin"
]

missing = df[
    required
].isna().sum()

print("\nMissing values:")
print(missing)

if missing.sum() != 0:
    raise ValueError(
        "Unexpected missing values."
    )


# ============================================================
# 11. FINAL SANITY CHECK
# ============================================================

print("\n" + "=" * 75)
print("FINAL EXACT VALIDATED SAMPLE")
print("=" * 75)

print(f"N: {len(df):,}")
print(f"Genes: {df['gene'].nunique():,}")
print(
    f"Cis: {int(df['observed_in_cis'].sum()):,}"
)
print(
    f"Cis %: "
    f"{100 * df['observed_in_cis'].mean():.3f}%"
)

print("\nVESM score summaries:")

for c in [
    "E_signed",
    "E_abs",
    "d1",
    "d2"
]:

    print(
        f"{c:12s} "
        f"mean={df[c].mean(): .8f}  "
        f"SD={df[c].std(): .8f}  "
        f"median={df[c].median(): .8f}"
    )


# ============================================================
# 12. STANDARDIZE
# ============================================================

for c in [
    "d1",
    "d2",
    "E_signed",
    "E_abs"
]:

    df[c + "_z"] = (
        df[c] - df[c].mean()
    ) / df[c].std()


# ============================================================
# 13. AF DUMMIES
# ============================================================

af = pd.get_dummies(
    df["AF_bin"],
    prefix="AF",
    drop_first=True,
    dtype=float
)

df = pd.concat(
    [df, af],
    axis=1
)

AF_COLS = list(
    af.columns
)


# ============================================================
# 14. MODELS
# ============================================================

models = {

    "M1_signed_E": [
        "E_signed_z"
    ],

    "M2_abs_E": [
        "E_abs_z"
    ],

    "M3_d1_d2_signed_E": [
        "d1_z",
        "d2_z",
        "E_signed_z"
    ],

    "M4_d1_d2_abs_E": [
        "d1_z",
        "d2_z",
        "E_abs_z"
    ],

    "M5_d1_d2_signed_E_AF": [
        "d1_z",
        "d2_z",
        "E_signed_z"
    ] + AF_COLS,

    "M6_d1_d2_abs_E_AF": [
        "d1_z",
        "d2_z",
        "E_abs_z"
    ] + AF_COLS,

    "M7_d1_d2_AF": [
        "d1_z",
        "d2_z"
    ] + AF_COLS
}


# ============================================================
# 15. FIT MODELS
# ============================================================

print("\n" + "=" * 75)
print("FITTING FINAL VESM CIS MODELS")
print("=" * 75)

y = df[
    "observed_in_cis"
].astype(int)

results = []


for name, predictors in models.items():

    print(
        "\n" + name
    )

    X = df[
        predictors
    ].astype(float)

    X = sm.add_constant(
        X,
        has_constant="add"
    )

    fit = sm.GLM(
        y,
        X,
        family=sm.families.Binomial()
    ).fit()

    pred = fit.predict(X)

    auc = roc_auc_score(
        y,
        pred
    )

    print(f"AUC = {auc:.6f}")

    for term in predictors:

        beta = float(
            fit.params[term]
        )

        se = float(
            fit.bse[term]
        )

        p = float(
            fit.pvalues[term]
        )

        OR = np.exp(beta)

        low = np.exp(
            beta - 1.96 * se
        )

        high = np.exp(
            beta + 1.96 * se
        )

        print(
            f"{term:24s} "
            f"beta={beta:+.8f}  "
            f"OR={OR:.6f}  "
            f"95% CI="
            f"{low:.6f}-{high:.6f}  "
            f"P={p:.4g}"
        )

        results.append(
            {
                "model": name,
                "term": term,
                "beta": beta,
                "SE": se,
                "OR": OR,
                "CI_low": low,
                "CI_high": high,
                "P": p,
                "AUC": auc,
                "N": len(df)
            }
        )


results = pd.DataFrame(results)

results.to_csv(
    OUT_RESULTS,
    sep="\t",
    index=False
)


# ============================================================
# 16. EXTRACT MAIN RESULTS
# ============================================================

def get(model, term):

    x = results[
        (results["model"] == model)
        &
        (results["term"] == term)
    ]

    if len(x) != 1:
        raise ValueError(
            f"Could not find {model}/{term}"
        )

    return x.iloc[0]


s1 = get(
    "M1_signed_E",
    "E_signed_z"
)

a1 = get(
    "M2_abs_E",
    "E_abs_z"
)

sd = get(
    "M3_d1_d2_signed_E",
    "E_signed_z"
)

ad = get(
    "M4_d1_d2_abs_E",
    "E_abs_z"
)

saf = get(
    "M5_d1_d2_signed_E_AF",
    "E_signed_z"
)

aaf = get(
    "M6_d1_d2_abs_E_AF",
    "E_abs_z"
)

auc_base = results.loc[
    results["model"] == "M7_d1_d2_AF",
    "AUC"
].iloc[0]


# ============================================================
# 17. FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("FINAL VESM-35M CIS RESULTS")
print("=" * 75)


def report(label, x):

    print(
        f"{label}: "
        f"beta={x['beta']:+.8f}, "
        f"OR={x['OR']:.6f}, "
        f"95% CI="
        f"{x['CI_low']:.6f}-"
        f"{x['CI_high']:.6f}, "
        f"P={x['P']:.4g}, "
        f"AUC={x['AUC']:.6f}"
    )


print("\nSIGNED E_cond")

report(
    "Unadjusted",
    s1
)

report(
    "d1+d2 adjusted",
    sd
)

report(
    "d1+d2+AF adjusted",
    saf
)


print("\nABSOLUTE |E_cond|")

report(
    "Unadjusted",
    a1
)

report(
    "d1+d2 adjusted",
    ad
)

report(
    "d1+d2+AF adjusted",
    aaf
)


# ============================================================
# 18. ATTENUATION
# ============================================================

def attenuation(b0, b1):

    return (
        1 -
        abs(b1) / abs(b0)
    ) * 100


print("\nABSOLUTE-E ATTENUATION")

print(
    "Unadjusted -> d1+d2: "
    f"{attenuation(a1['beta'], ad['beta']):.2f}%"
)

print(
    "Unadjusted -> d1+d2+AF: "
    f"{attenuation(a1['beta'], aaf['beta']):.2f}%"
)


# ============================================================
# 19. AUC CONTRIBUTION
# ============================================================

print("\nAUC CONTRIBUTION OF |E| AFTER AF")

print(
    f"d1+d2+AF:       "
    f"{auc_base:.6f}"
)

print(
    f"d1+d2+|E|+AF:   "
    f"{aaf['AUC']:.6f}"
)

print(
    f"|E| AUC gain:   "
    f"{aaf['AUC'] - auc_base:+.6f}"
)


# ============================================================
# 20. ESM2 REFERENCE
# ============================================================

print("\n" + "=" * 75)
print("ESM2 REFERENCE")
print("=" * 75)

print(
    "ESM2 |E| unadjusted: "
    "OR=0.943086, AUC=0.527869"
)

print(
    "ESM2 d1+d2+|E|: "
    "OR=0.949998, AUC=0.527360"
)

print(
    "ESM2 d1+d2+|E|+AF: "
    "OR=0.975257, "
    "95% CI=0.972065-0.978460, "
    "P=1.035e-50, "
    "AUC=0.629089"
)


print("\n" + "=" * 75)
print("DONE")
print("=" * 75)

print(
    f"Results saved to:\n{OUT_RESULTS}"
)
