import os
import glob
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS / SETTINGS
# ============================================================

BASE_DIR = os.environ.get("BASE_DIR", ".")
BASE = BASE_DIR

STRUCT_FILE = os.path.join(
    BASE,
    "3d_linear_contact_all_genes_FINAL",
    "all_usable_pairs.tsv"
)

OUT_DIR = os.path.join(
    BASE,
    "3d_linear_contact_all_genes_FINAL"
)

OUT_FOLDS = os.path.join(
    OUT_DIR,
    "ESM2_VESM_shared_pairs_gene_level_5fold_CV.tsv"
)

OUT_SUMMARY = os.path.join(
    OUT_DIR,
    "ESM2_VESM_shared_pairs_gene_level_5fold_CV_summary.tsv"
)

OUT_SHARED = os.path.join(
    OUT_DIR,
    "ESM2_VESM_shared_pairs_sample_summary.tsv"
)

N_SPLITS = 5


# ============================================================
# HELPER: LOAD MANY PER-GENE FILES
# ============================================================

def load_score_files(files, model_name):

    print("\n" + "=" * 70)
    print(f"LOADING {model_name} SCORES")
    print("=" * 70)

    dfs = []

    for i, f in enumerate(files, start=1):

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

        dfs.append(tmp)

        if i % 1000 == 0:
            print(
                f"Loaded {i:,} / {len(files):,} files"
            )

    dat = pd.concat(
        dfs,
        ignore_index=True
    )

    del dfs

    print(
        f"{model_name} rows: "
        f"{len(dat):,}"
    )

    print(
        f"{model_name} genes: "
        f"{dat['gene'].nunique():,}"
    )

    # Check duplicate pair keys
    key = ["gene", "double_mut"]

    dup = dat.duplicated(
        subset=key
    ).sum()

    print(
        f"{model_name} duplicate keys: "
        f"{dup:,}"
    )

    if dup > 0:

        dup_df = dat[
            dat.duplicated(
                subset=key,
                keep=False
            )
        ].copy()

        disagreement = (
            dup_df
            .groupby(key)[
                ["d1", "d2", "E_cond"]
            ]
            .nunique(dropna=False)
            .max(axis=1)
            .gt(1)
            .sum()
        )

        print(
            f"{model_name} duplicated keys "
            f"with differing scores: "
            f"{disagreement:,}"
        )

        if disagreement > 0:
            raise ValueError(
                f"{model_name}: duplicated keys "
                "contain different scores."
            )

        dat = dat.drop_duplicates(
            subset=key,
            keep="first"
        )

    return dat


# ============================================================
# 1. LOAD STRUCTURAL DATA
# ============================================================

print("=" * 70)
print("LOADING STRUCTURAL DATA")
print("=" * 70)

struct = pd.read_csv(
    STRUCT_FILE,
    sep="\t",
    low_memory=False
)

print(
    f"Structural pairs: "
    f"{len(struct):,}"
)

print(
    f"Structural genes: "
    f"{struct['gene'].nunique():,}"
)

print(
    f"Contacts: "
    f"{int(struct['is_contact'].sum()):,}"
)

print(
    f"Contact percentage: "
    f"{100 * struct['is_contact'].mean():.4f}%"
)


# ============================================================
# 2. FIND ESM2 FILES
# ============================================================

print("\n" + "=" * 70)
print("FINDING ESM2 FILES")
print("=" * 70)

ESM2_PATTERN = os.path.join(
    BASE,
    "*",
    "*_epistasis_esmcnd.tsv"
)

esm_files = glob.glob(
    ESM2_PATTERN
)

print(
    f"Found ESM2 files: "
    f"{len(esm_files):,}"
)

if len(esm_files) == 0:
    raise ValueError(
        "No ESM2 files found."
    )


# ============================================================
# 3. FIND VESM FILES
# ============================================================

print("\n" + "=" * 70)
print("FINDING VESM-35M FILES")
print("=" * 70)

VESM_PATTERN = os.path.join(
    BASE,
    "*",
    "*_epistasis_VESM_35M.tsv"
)

vesm_files = glob.glob(
    VESM_PATTERN
)

vesm_files = [
    f for f in vesm_files
    if "skipped_rows" not in os.path.basename(f)
]

print(
    f"Found VESM files: "
    f"{len(vesm_files):,}"
)

if len(vesm_files) == 0:
    raise ValueError(
        "No VESM files found."
    )


# ============================================================
# 4. LOAD BOTH MODELS
# ============================================================

esm = load_score_files(
    esm_files,
    "ESM2-650M"
)

vesm = load_score_files(
    vesm_files,
    "VESM-35M"
)


# ============================================================
# 5. RENAME SCORE COLUMNS BEFORE MERGING
# ============================================================

esm = esm.rename(
    columns={
        "d1": "ESM2_d1",
        "d2": "ESM2_d2",
        "E_cond": "ESM2_E_cond"
    }
)

vesm = vesm.rename(
    columns={
        "d1": "VESM_d1",
        "d2": "VESM_d2",
        "E_cond": "VESM_E_cond"
    }
)


# ============================================================
# 6. IDENTIFY PAIRS SCORED BY BOTH PLMs
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING SHARED ESM2/VESM PAIRS")
print("=" * 70)

shared_scores = esm.merge(
    vesm,
    on=[
        "gene",
        "double_mut"
    ],
    how="inner",
    validate="one_to_one"
)

print(
    f"Pairs scored by both models: "
    f"{len(shared_scores):,}"
)

print(
    f"Genes represented in both: "
    f"{shared_scores['gene'].nunique():,}"
)


# ============================================================
# 7. MERGE WITH STRUCTURAL DATA
#
# IMPORTANT:
# This produces ONE shared structural sample.
# Every subsequent ESM2/VESM comparison uses these exact rows.
# ============================================================

print("\n" + "=" * 70)
print("MERGING SHARED SCORES WITH STRUCTURAL DATA")
print("=" * 70)

df = struct.merge(
    shared_scores,
    on=[
        "gene",
        "double_mut"
    ],
    how="inner",
    validate="one_to_one"
)

print(
    f"Shared structural pairs: "
    f"{len(df):,}"
)

print(
    f"Shared structural genes: "
    f"{df['gene'].nunique():,}"
)

print(
    f"Percentage of original structural pairs: "
    f"{100 * len(df) / len(struct):.4f}%"
)


# ============================================================
# 8. CLEAN VARIABLES
# ============================================================

score_cols = [
    "ESM2_d1",
    "ESM2_d2",
    "ESM2_E_cond",
    "VESM_d1",
    "VESM_d2",
    "VESM_E_cond",
    "is_contact"
]

for c in score_cols:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


before_clean = len(df)

df = df.dropna(
    subset=[
        "gene",
        "ESM2_d1",
        "ESM2_d2",
        "ESM2_E_cond",
        "VESM_d1",
        "VESM_d2",
        "VESM_E_cond",
        "is_contact"
    ]
).copy()

df = df[
    df["is_contact"].isin(
        [0, 1]
    )
].copy()


# Absolute E
df["ESM2_abs_E_cond"] = np.abs(
    df["ESM2_E_cond"]
)

df["VESM_abs_E_cond"] = np.abs(
    df["VESM_E_cond"]
)


# ============================================================
# 9. FINAL SHARED-SAMPLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL EXACTLY MATCHED SAMPLE")
print("=" * 70)

print(
    f"Pairs before cleaning: "
    f"{before_clean:,}"
)

print(
    f"Final shared pairs: "
    f"{len(df):,}"
)

print(
    f"Final shared genes: "
    f"{df['gene'].nunique():,}"
)

print(
    f"Contacts: "
    f"{int(df['is_contact'].sum()):,}"
)

print(
    f"Non-contacts: "
    f"{int((df['is_contact'] == 0).sum()):,}"
)

print(
    f"Contact percentage: "
    f"{100 * df['is_contact'].mean():.4f}%"
)


# Save sample summary
sample_summary = pd.DataFrame(
    {
        "metric": [
            "n_pairs",
            "n_genes",
            "n_contacts",
            "contact_percentage"
        ],
        "value": [
            len(df),
            df["gene"].nunique(),
            int(df["is_contact"].sum()),
            100 * df["is_contact"].mean()
        ]
    }
)

sample_summary.to_csv(
    OUT_SHARED,
    sep="\t",
    index=False
)


# ============================================================
# 10. SCORE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SCORE SUMMARY ON EXACT SAME PAIRS")
print("=" * 70)

for c in [
    "ESM2_d1",
    "ESM2_d2",
    "ESM2_E_cond",
    "ESM2_abs_E_cond",
    "VESM_d1",
    "VESM_d2",
    "VESM_E_cond",
    "VESM_abs_E_cond"
]:

    print(
        f"{c:22s} "
        f"mean={df[c].mean(): .6f}  "
        f"SD={df[c].std(): .6f}  "
        f"median={df[c].median(): .6f}"
    )


# ============================================================
# 11. DEFINE SIX MODELS
#
# Three ESM2 models + three VESM models.
# ============================================================

models = {

    "ESM2 M0: d1 + d2": [
        "ESM2_d1",
        "ESM2_d2"
    ],

    "ESM2 M1: d1 + d2 + signed E": [
        "ESM2_d1",
        "ESM2_d2",
        "ESM2_E_cond"
    ],

    "ESM2 M2: d1 + d2 + |E|": [
        "ESM2_d1",
        "ESM2_d2",
        "ESM2_abs_E_cond"
    ],

    "VESM M0: d1 + d2": [
        "VESM_d1",
        "VESM_d2"
    ],

    "VESM M1: d1 + d2 + signed E": [
        "VESM_d1",
        "VESM_d2",
        "VESM_E_cond"
    ],

    "VESM M2: d1 + d2 + |E|": [
        "VESM_d1",
        "VESM_d2",
        "VESM_abs_E_cond"
    ]
}


# ============================================================
# 12. GENE-LEVEL GROUPED CV
#
# CRITICAL:
# ONE GroupKFold object / ONE df.
#
# Therefore ESM2 and VESM get exactly the same test genes
# and exactly the same test pairs in every fold.
# ============================================================

groups = df["gene"].values

y = df["is_contact"].astype(
    int
).values

gkf = GroupKFold(
    n_splits=N_SPLITS
)


print("\n" + "=" * 70)
print("SHARED-PAIR GENE-LEVEL 5-FOLD CROSS-VALIDATION")
print("=" * 70)

results = []


for fold, (train_idx, test_idx) in enumerate(
    gkf.split(
        df,
        y,
        groups=groups
    ),
    start=1
):

    train_genes = set(
        df.iloc[
            train_idx
        ]["gene"]
    )

    test_genes = set(
        df.iloc[
            test_idx
        ]["gene"]
    )

    overlap = (
        train_genes
        &
        test_genes
    )

    if len(overlap) != 0:
        raise ValueError(
            f"Gene leakage in fold {fold}"
        )


    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]


    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Fold {fold}: training outcome "
            "contains only one class."
        )

    if len(np.unique(y_test)) < 2:
        raise ValueError(
            f"Fold {fold}: test outcome "
            "contains only one class."
        )


    print("\n" + "-" * 70)
    print(f"FOLD {fold}")
    print("-" * 70)

    print(
        f"Train pairs: "
        f"{len(train_idx):,}"
    )

    print(
        f"Test pairs:  "
        f"{len(test_idx):,}"
    )

    print(
        f"Train genes: "
        f"{len(train_genes):,}"
    )

    print(
        f"Test genes:  "
        f"{len(test_genes):,}"
    )

    print(
        f"Gene overlap: "
        f"{len(overlap)}"
    )

    print(
        f"Train contact %: "
        f"{100 * y_train.mean():.4f}%"
    )

    print(
        f"Test contact %:  "
        f"{100 * y_test.mean():.4f}%"
    )


    # ========================================================
    # FIT ALL SIX MODELS TO IDENTICAL TRAIN/TEST ROWS
    # ========================================================

    for model_name, predictors in models.items():

        X_train = df.iloc[
            train_idx
        ][predictors].values

        X_test = df.iloc[
            test_idx
        ][predictors].values


        # Fit standardization on training genes only
        scaler = StandardScaler()

        X_train_z = scaler.fit_transform(
            X_train
        )

        X_test_z = scaler.transform(
            X_test
        )


        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000
        )

        model.fit(
            X_train_z,
            y_train
        )


        pred = model.predict_proba(
            X_test_z
        )[:, 1]


        auc = roc_auc_score(
            y_test,
            pred
        )


        print(
            f"{model_name:36s} "
            f"AUC = {auc:.6f}"
        )


        row = {
            "fold": fold,
            "model": model_name,
            "auc": auc,
            "n_train_pairs": len(train_idx),
            "n_test_pairs": len(test_idx),
            "n_train_genes": len(train_genes),
            "n_test_genes": len(test_genes),
            "train_contact_rate": y_train.mean(),
            "test_contact_rate": y_test.mean()
        }


        for predictor, coef in zip(
            predictors,
            model.coef_[0]
        ):

            row[
                "beta_" + predictor
            ] = coef


        results.append(
            row
        )


# ============================================================
# 13. SAVE FOLD-LEVEL RESULTS
# ============================================================

results = pd.DataFrame(
    results
)

results.to_csv(
    OUT_FOLDS,
    sep="\t",
    index=False
)


# ============================================================
# 14. SUMMARIZE AUC
# ============================================================

summary = (
    results
    .groupby("model")["auc"]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max"
        ]
    )
    .reset_index()
)

summary.to_csv(
    OUT_SUMMARY,
    sep="\t",
    index=False
)


print("\n" + "=" * 70)
print("SHARED-PAIR GENE-LEVEL CV SUMMARY")
print("=" * 70)

for _, row in summary.iterrows():

    print(
        f"{row['model']:36s} "
        f"Mean AUC = {row['mean']:.6f}  "
        f"SD = {row['std']:.6f}  "
        f"Range = "
        f"{row['min']:.6f}-"
        f"{row['max']:.6f}"
    )


# ============================================================
# 15. WITHIN-MODEL AUC GAINS
# ============================================================

pivot = results.pivot(
    index="fold",
    columns="model",
    values="auc"
)


print("\n" + "=" * 70)
print("WITHIN-MODEL AUC GAINS")
print("=" * 70)


comparisons = [

    (
        "ESM2 signed E gain",
        "ESM2 M1: d1 + d2 + signed E",
        "ESM2 M0: d1 + d2"
    ),

    (
        "ESM2 |E| gain",
        "ESM2 M2: d1 + d2 + |E|",
        "ESM2 M0: d1 + d2"
    ),

    (
        "VESM signed E gain",
        "VESM M1: d1 + d2 + signed E",
        "VESM M0: d1 + d2"
    ),

    (
        "VESM |E| gain",
        "VESM M2: d1 + d2 + |E|",
        "VESM M0: d1 + d2"
    )
]


for label, full_model, base_model in comparisons:

    gain = (
        pivot[full_model]
        -
        pivot[base_model]
    )

    print(
        f"\n{label}"
    )

    print(
        f"Mean gain = "
        f"{gain.mean():+.6f}"
    )

    print(
        f"SD gain   = "
        f"{gain.std():.6f}"
    )

    for fold, value in gain.items():

        print(
            f"  Fold {fold}: "
            f"{value:+.6f}"
        )


# ============================================================
# 16. DIRECT ESM2 VS VESM COMPARISON
#
# Since the same pairs and same folds are used, this is now
# an apples-to-apples descriptive comparison.
# ============================================================

print("\n" + "=" * 70)
print("DIRECT VESM MINUS ESM2 AUC DIFFERENCES")
print("=" * 70)


direct_comparisons = [

    (
        "Baseline d1+d2",
        "VESM M0: d1 + d2",
        "ESM2 M0: d1 + d2"
    ),

    (
        "Signed E model",
        "VESM M1: d1 + d2 + signed E",
        "ESM2 M1: d1 + d2 + signed E"
    ),

    (
        "|E| model",
        "VESM M2: d1 + d2 + |E|",
        "ESM2 M2: d1 + d2 + |E|"
    )
]


for label, vesm_model, esm_model in direct_comparisons:

    difference = (
        pivot[vesm_model]
        -
        pivot[esm_model]
    )

    print(
        f"\n{label}"
    )

    print(
        f"Mean VESM - ESM2 AUC = "
        f"{difference.mean():+.6f}"
    )

    print(
        f"SD difference = "
        f"{difference.std():.6f}"
    )

    print(
        "Fold differences:"
    )

    for fold, value in difference.items():

        print(
            f"  Fold {fold}: "
            f"{value:+.6f}"
        )


# ============================================================
# 17. COMPACT FINAL TABLE
# ============================================================

print("\n" + "=" * 70)
print("COMPACT HEAD-TO-HEAD SUMMARY")
print("=" * 70)

model_order = [
    "ESM2 M0: d1 + d2",
    "ESM2 M1: d1 + d2 + signed E",
    "ESM2 M2: d1 + d2 + |E|",
    "VESM M0: d1 + d2",
    "VESM M1: d1 + d2 + signed E",
    "VESM M2: d1 + d2 + |E|"
]

for model_name in model_order:

    vals = pivot[
        model_name
    ]

    print(
        f"{model_name:36s} "
        f"{vals.mean():.6f} "
        f"+/- {vals.std():.6f}"
    )


# ============================================================
# 18. DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nShared-sample summary:"
)
print(
    OUT_SHARED
)

print(
    "\nFold-level results:"
)
print(
    OUT_FOLDS
)

print(
    "\nCV summary:"
)
print(
    OUT_SUMMARY
)
