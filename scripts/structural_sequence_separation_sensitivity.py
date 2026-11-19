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

OUT_FOLDS = os.path.join(OUT_DIR, "sequence_separation_sensitivity_fold_AUC.tsv")

OUT_SUMMARY = os.path.join(
    OUT_DIR,
    "sequence_separation_sensitivity_summary.tsv"
)

OUT_SHARED = os.path.join(
    OUT_DIR,
    "sequence_separation_sensitivity_sample_summary.tsv"
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
    "is_contact",
    "linear_dist"
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
        "is_contact",
        "linear_dist"
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

# Sequence separation is already present in the structural file.
assert np.allclose(df["linear_dist"], (pd.to_numeric(df["pos1"]) - pd.to_numeric(df["pos2"])).abs()), "linear_dist does not match pos1/pos2"
assert (df["linear_dist"] >= 0).all()
df["log1p_linear_dist"] = np.log1p(df["linear_dist"])



# ============================================================
# SENSITIVITY ANALYSIS: sequence separation and local pairs
# ============================================================

model_specs = {}
for name in ("ESM2", "VESM"):
    base = [f"{name}_d1", f"{name}_d2"]
    model_specs[(name, "single_scores")] = base
    model_specs[(name, "single_scores+distance")] = base + ["log1p_linear_dist"]
    model_specs[(name, "single_scores+absE")] = base + [f"{name}_abs_E_cond"]
    model_specs[(name, "single_scores+distance+absE")] = base + ["log1p_linear_dist", f"{name}_abs_E_cond"]
    model_specs[(name, "single_scores+distance+signedE")] = base + ["log1p_linear_dist", f"{name}_E_cond"]

results = []
samples = []
for subset_name, sub in [
    ("all_pairs", df),
    ("separation_gt_2", df.loc[df["linear_dist"] > 2].copy()),
]:
    print(f"\n{subset_name}: {len(sub):,} pairs, {sub['gene'].nunique():,} genes", flush=True)
    if sub.empty or sub['gene'].nunique() < N_SPLITS:
        raise ValueError(f"Insufficient genes in {subset_name}")
    samples.append({"subset": subset_name, "n_pairs": len(sub), "n_genes": sub['gene'].nunique(), "contact_rate": sub['is_contact'].mean()})
    y = sub["is_contact"].astype(int).to_numpy()
    groups = sub["gene"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=N_SPLITS).split(sub, y, groups), start=1):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            raise ValueError(f"{subset_name}, fold {fold}: only one outcome class")
        assert not (set(groups[train_idx]) & set(groups[test_idx]))
        for (plm, model_name), columns in model_specs.items():
            X_train = sub.iloc[train_idx][columns].to_numpy(dtype=float)
            X_test = sub.iloc[test_idx][columns].to_numpy(dtype=float)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            model.fit(X_train, y_train)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            results.append({"subset": subset_name, "fold": fold, "plm": plm,
                            "model": model_name, "auc": auc,
                            "n_train": len(train_idx), "n_test": len(test_idx),
                            "train_contact_rate": y_train.mean(), "test_contact_rate": y_test.mean()})
            print(f"{subset_name} fold {fold} {plm} {model_name}: {auc:.6f}", flush=True)

res = pd.DataFrame(results)
res.to_csv(OUT_FOLDS, sep="\t", index=False)
pd.DataFrame(samples).to_csv(OUT_SHARED, sep="\t", index=False)
summary = res.groupby(["subset", "plm", "model"], as_index=False).agg(
    mean_auc=("auc", "mean"), sd_auc=("auc", "std"))
summary.to_csv(OUT_SUMMARY, sep="\t", index=False)
print("\nSUMMARY\n", summary.to_string(index=False))

wide = res.pivot(index=["subset", "plm", "fold"], columns="model", values="auc")
wide["delta_absE_over_distance"] = (wide["single_scores+distance+absE"] - wide["single_scores+distance"])
print("\nINCREMENTAL AUC FROM |E| AFTER DISTANCE ADJUSTMENT\n",
      wide.groupby(level=["subset", "plm"])["delta_absE_over_distance"].agg(["mean", "std"]).to_string())
print("\nSaved:", OUT_FOLDS, OUT_SUMMARY, OUT_SHARED, sep="\n")
