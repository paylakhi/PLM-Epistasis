import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.environ.get("BASE_DIR", ".")
CIS_BASE = os.path.join(BASE_DIR, "gnomad_cis_epistasis_all_genes_results")
REG_FILE = os.path.join(CIS_BASE, "cis_regression", "cis_regression_input_merged.tsv")
CIS_FILE = os.path.join(CIS_BASE, "merged_all_pairs_with_qc_REVISED.tsv")
OUT_FILE = os.path.join(CIS_BASE, "cis_regression", "abs_Econd_cis_regression_results.tsv")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)


# ============================================================
# 1. LOAD REGRESSION DATA
# ============================================================

print("Loading regression data...")

reg = pd.read_csv(
    REG_FILE,
    sep="\t",
    low_memory=False
)

print(f"Regression rows: {len(reg):,}")

print("\nRegression columns:")
print(reg.columns.tolist())


# ============================================================
# 2. LOAD HAPLOTYPE DATA
# ============================================================

print("\nLoading haplotype data...")

hap = pd.read_csv(
    CIS_FILE,
    sep="\t",
    usecols=[
        "gene",
        "pair_key",
        "all_hap_counts_0",
        "all_hap_counts_1",
        "all_hap_counts_2",
        "all_hap_counts_3"
    ],
    low_memory=False
)

print(f"Haplotype rows before restriction: {len(hap):,}")


# ============================================================
# 3. CHECK REGRESSION KEYS
# ============================================================

print("\n======================================================")
print("MERGE KEY CHECK")
print("======================================================")

key = [
    "gene",
    "pair_key"
]

reg_dup = reg.duplicated(
    subset=key
).sum()

print(
    f"Duplicate regression keys: "
    f"{reg_dup:,}"
)

if reg_dup > 0:
    raise ValueError(
        "Regression file contains duplicate gene+pair_key."
    )


# ============================================================
# 4. RESTRICT HAPLOTYPE DATA TO REGRESSION PAIRS
# ============================================================

reg_keys = reg[
    key
].drop_duplicates()

hap = hap.merge(
    reg_keys,
    on=key,
    how="inner"
)

print(
    f"Haplotype rows after restricting to regression keys: "
    f"{len(hap):,}"
)

hap_dup = hap.duplicated(
    subset=key
).sum()

print(
    f"Duplicate haplotype keys after restriction: "
    f"{hap_dup:,}"
)

if hap_dup > 0:
    raise ValueError(
        "Unexpected duplicate haplotype keys remain "
        "after restricting to regression pairs."
    )


# ============================================================
# 5. VERIFY EVERY REGRESSION PAIR HAS ONE HAPLOTYPE RECORD
# ============================================================

if len(hap) != len(reg):

    raise ValueError(
        f"Expected {len(reg):,} haplotype records after "
        f"restriction, but found {len(hap):,}."
    )

print(
    "All regression pairs have exactly one "
    "haplotype record: YES"
)


# ============================================================
# 6. MERGE REGRESSION + HAPLOTYPE DATA
# ============================================================

df = reg.merge(
    hap,
    on=key,
    how="inner",
    validate="one_to_one"
)

print("\n======================================================")
print("MERGE RESULTS")
print("======================================================")

print(
    f"Regression rows before merge: "
    f"{len(reg):,}"
)

print(
    f"Merged rows:                 "
    f"{len(df):,}"
)

print(
    f"Merge percentage:            "
    f"{100 * len(df) / len(reg):.4f}%"
)


if len(df) != len(reg):
    raise ValueError(
        "Final merged dataset does not contain "
        "all regression pairs."
    )


# ============================================================
# 7. NUMERIC CONVERSION
# ============================================================

numeric_cols = [
    "E_cond",
    "epistasis_score",
    "d1",
    "d2",
    "observed_in_cis",
    "all_hap_counts_0",
    "all_hap_counts_1",
    "all_hap_counts_2",
    "all_hap_counts_3"
]

for c in numeric_cols:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


# ============================================================
# 8. SANITY CHECK:
# E_cond MUST MATCH epistasis_score
# ============================================================

diff = np.abs(
    df["E_cond"] -
    df["epistasis_score"]
)

print("\n======================================================")
print("E_cond SANITY CHECK")
print("======================================================")

print(
    "Maximum |E_cond - epistasis_score|:",
    diff.max()
)

print(
    "Rows with difference > 1e-8:",
    (diff > 1e-8).sum()
)

if (diff > 1e-8).any():

    raise ValueError(
        "E_cond does not exactly match epistasis_score."
    )


# ============================================================
# 9. CALCULATE ALLELE FREQUENCIES
#
# h0 = neither mutation
# h1 = mutation 1 only
# h2 = mutation 2 only
# h3 = both mutations
#
# AF1 = (h1 + h3) / total
# AF2 = (h2 + h3) / total
#
# AF_product = AF1 * AF2
# ============================================================

hap_cols = [
    "all_hap_counts_0",
    "all_hap_counts_1",
    "all_hap_counts_2",
    "all_hap_counts_3"
]

df["N_hap"] = df[
    hap_cols
].sum(axis=1)


# Remove impossible zero-haplotype rows if any somehow remain

print("\nZero-haplotype rows:")
print(
    (df["N_hap"] <= 0).sum()
)


df.loc[
    df["N_hap"] <= 0,
    "N_hap"
] = np.nan


df["AF1"] = (
    df["all_hap_counts_1"] +
    df["all_hap_counts_3"]
) / df["N_hap"]


df["AF2"] = (
    df["all_hap_counts_2"] +
    df["all_hap_counts_3"]
) / df["N_hap"]


df["AF_product"] = (
    df["AF1"] *
    df["AF2"]
)


# ============================================================
# 10. AF-PRODUCT STRATA
#
# Same strata used in previous sensitivity analysis:
#
# < 1e-8
# 1e-8 to < 1e-7
# 1e-7 to < 1e-6
# >= 1e-6
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


# ============================================================
# 11. CREATE SIGNED AND ABSOLUTE E_cond
# ============================================================

df["E_signed"] = df["E_cond"]

df["E_abs"] = np.abs(
    df["E_cond"]
)


# ============================================================
# 12. CLEAN ANALYSIS SAMPLE
#
# IMPORTANT:
# ALL MODELS USE EXACTLY THE SAME SAMPLE.
# ============================================================

needed = [
    "observed_in_cis",
    "E_signed",
    "E_abs",
    "d1",
    "d2",
    "AF_product",
    "AF_bin"
]

dat = df.dropna(
    subset=needed
).copy()


dat = dat[
    dat["observed_in_cis"].isin(
        [0, 1]
    )
].copy()


print("\n======================================================")
print("FINAL ANALYSIS SAMPLE")
print("======================================================")

print(
    f"N = {len(dat):,}"
)

print(
    f"Observed in cis = "
    f"{int(dat['observed_in_cis'].sum()):,}"
)

print(
    f"Not observed in cis = "
    f"{len(dat) - int(dat['observed_in_cis'].sum()):,}"
)

print(
    f"Cis percentage = "
    f"{100 * dat['observed_in_cis'].mean():.3f}%"
)


print("\nAF-product strata:")

print(
    dat["AF_bin"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 13. BASIC AF PRODUCT SUMMARY
# ============================================================

print("\nAF-product summary:")

print(
    dat["AF_product"].describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


# ============================================================
# 14. STANDARDIZE CONTINUOUS VARIABLES
#
# Standardization is performed ONCE using the common
# analysis sample.
#
# Therefore all ORs are directly comparable and correspond
# to a one-SD increase in the predictor.
# ============================================================

variables = [
    "E_signed",
    "E_abs",
    "d1",
    "d2"
]

print("\n======================================================")
print("STANDARDIZATION")
print("======================================================")

for c in variables:

    mean = dat[c].mean()

    sd = dat[c].std()

    if sd == 0 or np.isnan(sd):
        raise ValueError(
            f"{c} has invalid standard deviation: {sd}"
        )

    dat[c + "_z"] = (
        dat[c] - mean
    ) / sd

    print(
        f"{c:10s} "
        f"mean={mean:.8f} "
        f"SD={sd:.8f}"
    )


# ============================================================
# 15. LOGISTIC REGRESSION FUNCTION
# ============================================================

def fit_model(
    name,
    continuous_cols,
    use_af_bins=False
):

    X = dat[
        continuous_cols
    ].copy()


    # --------------------------------------------------------
    # Add AF-product categorical adjustment if requested
    # --------------------------------------------------------

    if use_af_bins:

        af_dummies = pd.get_dummies(
            dat["AF_bin"],
            prefix="AF",
            drop_first=True,
            dtype=float
        )

        X = pd.concat(
            [
                X.reset_index(
                    drop=True
                ),
                af_dummies.reset_index(
                    drop=True
                )
            ],
            axis=1
        )


    X = X.reset_index(
        drop=True
    ).astype(float)


    y = dat[
        "observed_in_cis"
    ].reset_index(
        drop=True
    ).astype(float)


    X = sm.add_constant(
        X,
        has_constant="add"
    )


    print("\n======================================================")
    print(name)
    print("======================================================")


    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial()
    )


    res = model.fit()


    pred = res.predict(
        X
    )


    auc = roc_auc_score(
        y,
        pred
    )


    print(
        f"AUC = {auc:.6f}"
    )


    rows = []


    for variable in X.columns:

        if variable == "const":
            continue


        beta = res.params[
            variable
        ]

        se = res.bse[
            variable
        ]

        p = res.pvalues[
            variable
        ]


        lo_beta = (
            beta -
            1.96 * se
        )

        hi_beta = (
            beta +
            1.96 * se
        )


        OR = np.exp(
            beta
        )

        lo = np.exp(
            lo_beta
        )

        hi = np.exp(
            hi_beta
        )


        print(
            f"{variable:25s} "
            f"beta={beta: .8f}  "
            f"SE={se:.8f}  "
            f"OR={OR:.6f}  "
            f"95%CI=({lo:.6f}, {hi:.6f})  "
            f"P={p:.4g}"
        )


        rows.append(
            {
                "model": name,
                "variable": variable,
                "beta": beta,
                "SE": se,
                "OR": OR,
                "CI_low": lo,
                "CI_high": hi,
                "P": p,
                "AUC": auc,
                "N": len(dat)
            }
        )


    return rows


# ============================================================
# 16. RUN MODELS
# ============================================================

all_results = []


# ------------------------------------------------------------
# M1
#
# Signed E_cond only
#
# This should approximately reproduce the original
# unadjusted signed-E association.
# ------------------------------------------------------------

all_results += fit_model(
    "M1: signed E",
    [
        "E_signed_z"
    ]
)


# ------------------------------------------------------------
# M2
#
# Absolute E_cond only

# ------------------------------------------------------------

all_results += fit_model(
    "M2: |E|",
    [
        "E_abs_z"
    ]
)


# ------------------------------------------------------------
# M3
#
# Signed E_cond adjusted for constituent single scores

# ------------------------------------------------------------

all_results += fit_model(
    "M3: d1 + d2 + signed E",
    [
        "d1_z",
        "d2_z",
        "E_signed_z"
    ]
)


# ------------------------------------------------------------
# M4
#
# Absolute E_cond adjusted for constituent single scores
#
# ------------------------------------------------------------

all_results += fit_model(
    "M4: d1 + d2 + |E|",
    [
        "d1_z",
        "d2_z",
        "E_abs_z"
    ]
)


# ------------------------------------------------------------
# M5
#
# Signed E_cond
# + constituent single scores
# + AF-product strata
#
# ------------------------------------------------------------

all_results += fit_model(
    "M5: d1 + d2 + signed E + AF strata",
    [
        "d1_z",
        "d2_z",
        "E_signed_z"
    ],
    use_af_bins=True
)


# ------------------------------------------------------------
# M6
#
# Absolute E_cond
# + constituent single scores
# + AF-product strata
#
# ------------------------------------------------------------

all_results += fit_model(
    "M6: d1 + d2 + |E| + AF strata",
    [
        "d1_z",
        "d2_z",
        "E_abs_z"
    ],
    use_af_bins=True
)

# ------------------------------------------------------------
# M7
#
# Constituent single scores
# + AF-product strata
#
# ------------------------------------------------------------

all_results += fit_model(
    "M7: d1 + d2 + AF strata",
    [
        "d1_z",
        "d2_z"
    ],
    use_af_bins=True
)
# ============================================================
# 17. SAVE ALL MODEL RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)


results.to_csv(
    OUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 18. PRINT KEY E_cond RESULTS
# ============================================================

print("\n======================================================")
print("KEY E_cond RESULTS")
print("======================================================")


key_results = results[
    results["variable"].isin(
        [
            "E_signed_z",
            "E_abs_z"
        ]
    )
][
    [
        "model",
        "variable",
        "beta",
        "SE",
        "OR",
        "CI_low",
        "CI_high",
        "P",
        "AUC",
        "N"
    ]
]


print(
    key_results.to_string(
        index=False
    )
)


# ============================================================
# 19. ABSOLUTE-E ATTENUATION AFTER d1/d2 ADJUSTMENT
#
# Compare:
#
# M2: |E|
# vs
# M4: d1 + d2 + |E|
# ============================================================

m2 = key_results[
    (
        key_results["model"] ==
        "M2: |E|"
    )
    &
    (
        key_results["variable"] ==
        "E_abs_z"
    )
]


m4 = key_results[
    (
        key_results["model"] ==
        "M4: d1 + d2 + |E|"
    )
    &
    (
        key_results["variable"] ==
        "E_abs_z"
    )
]


if len(m2) == 1 and len(m4) == 1:

    beta_m2 = float(
        m2["beta"].iloc[0]
    )

    beta_m4 = float(
        m4["beta"].iloc[0]
    )


    attenuation_single = (
        100 *
        (
            1 -
            abs(beta_m4) /
            abs(beta_m2)
        )
    )


    print("\n======================================================")
    print("ABSOLUTE-E ATTENUATION AFTER d1/d2 ADJUSTMENT")
    print("======================================================")


    print(
        f"Beta |E| only:             "
        f"{beta_m2:.8f}"
    )

    print(
        f"Beta after d1/d2:          "
        f"{beta_m4:.8f}"
    )

    print(
        f"Attenuation after d1/d2:   "
        f"{attenuation_single:.2f}%"
    )


# ============================================================
# 20. ABSOLUTE-E ATTENUATION AFTER AF ADJUSTMENT
#
# Compare:
#
# M4: d1 + d2 + |E|
#
# vs
#
# M6: d1 + d2 + |E| + AF strata
# ============================================================

m6 = key_results[
    (
        key_results["model"] ==
        "M6: d1 + d2 + |E| + AF strata"
    )
    &
    (
        key_results["variable"] ==
        "E_abs_z"
    )
]


if len(m4) == 1 and len(m6) == 1:

    beta_before_af = float(
        m4["beta"].iloc[0]
    )

    beta_after_af = float(
        m6["beta"].iloc[0]
    )


    attenuation_af = (
        100 *
        (
            1 -
            abs(beta_after_af) /
            abs(beta_before_af)
        )
    )


    print("\n======================================================")
    print("ABSOLUTE-E ATTENUATION AFTER AF ADJUSTMENT")
    print("======================================================")


    print(
        f"Beta before AF adjustment: "
        f"{beta_before_af:.8f}"
    )

    print(
        f"Beta after AF adjustment:  "
        f"{beta_after_af:.8f}"
    )

    print(
        f"Attenuation after AF:       "
        f"{attenuation_af:.2f}%"
    )


# ============================================================
# 21. FINAL OUTPUT LOCATION
# ============================================================

print("\n======================================================")
print("DONE")
print("======================================================")


print(
    "\nSaved results to:"
)

print(
    OUT_FILE
)
