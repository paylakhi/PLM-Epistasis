#!/usr/bin/env python3
"""Gene-held-out cis ROC-AUC: ESM2 and VESM-35M from existing gene-level TSVs.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get('BASE_DIR', '.'))
CIS = BASE / 'gnomad_cis_epistasis_all_genes_results'
REG = CIS / 'cis_regression/cis_regression_input_merged.tsv'
HAP = CIS / 'merged_all_pairs_with_qc_REVISED.tsv'
OUT = CIS / 'cis_regression/cis_gene_cv_from_gene_files'
AF_LABELS = ['<1e-8', '1e-8-<1e-7', '1e-7-<1e-6', '>=1e-6']
MODEL_NAMES = ['Singles + AF', 'Singles + AF + signed E', 'Singles + AF + abs E']


def read_reg(path):
    cols = ['gene', 'pair_key', 'observed_in_cis', 'd1', 'd2', 'E_cond']
    reg = pd.read_csv(path, sep='\t', usecols=cols, low_memory=False)
    reg['gene'] = reg['gene'].astype(str)
    reg['pair_key'] = reg['pair_key'].astype(str)
    if reg.duplicated(['gene', 'pair_key']).any():
        raise ValueError('ESM2 cis cohort has duplicate (gene,pair_key) records')
    if len(reg) != 1_630_391 or int(reg.observed_in_cis.sum()) != 1_145_027:
        raise ValueError(f'Unexpected validated ESM2 cohort: N={len(reg)}, observed={reg.observed_in_cis.sum()}')
    return reg


def add_af(reg, path):
    keys = ['gene', 'pair_key']
    hc = [f'all_hap_counts_{i}' for i in range(4)]
    h = pd.read_csv(path, sep='\t', usecols=keys + hc, low_memory=False)
    h['gene'] = h['gene'].astype(str)
    h['pair_key'] = h['pair_key'].astype(str)
    h = h.merge(reg[keys], on=keys, how='inner', validate='many_to_one')
    for c in hc:
        h[c] = pd.to_numeric(h[c], errors='coerce')
    # Check all repeated keys agree, including missingness; don't arbitrarily pick one.
    distinct = h.drop_duplicates(keys + hc)
    bad = distinct.duplicated(keys, keep=False)
    if bad.any():
        raise ValueError('Conflicting haplotype counts for keys:\n' + distinct.loc[bad].head(8).to_string(index=False))
    h = distinct.drop_duplicates(keys)
    if len(h) != len(reg):
        raise ValueError(f'Missing AF records: {len(h):,} / {len(reg):,} keys matched')
    total = h[hc].sum(axis=1, min_count=4)
    af1 = (h['all_hap_counts_1'] + h['all_hap_counts_3']) / total
    af2 = (h['all_hap_counts_2'] + h['all_hap_counts_3']) / total
    h['AF_product'] = af1 * af2
    af_values = h['AF_product'].to_numpy(dtype=float)
    n_missing = int(np.isnan(af_values).sum())
    n_nonfinite = int((~np.isfinite(af_values)).sum())
    n_negative = int((af_values < 0).sum())
    n_zero = int((af_values == 0).sum())
    print(f'AF-product QC: zero={n_zero:,}; missing={n_missing:,}; nonfinite={n_nonfinite:,}; negative={n_negative:,}', flush=True)
    if n_nonfinite or n_negative:
        raise ValueError('Invalid AF products (missing, infinite, or negative); inspect haplotype counts')
    # Zero AF products belong to the lowest AF-product category, as in the original regression.
    h['AF_bin'] = pd.cut(h['AF_product'], [-np.inf, 1e-8, 1e-7, 1e-6, np.inf], labels=AF_LABELS, right=False)
    return reg.merge(h[keys + ['AF_bin']], on=keys, how='left', validate='one_to_one')


def read_vesm_files(base, valid_keys, out):
    paths = sorted(base.glob('*/*_epistasis_VESM_35M.tsv'))
    if not paths:
        raise FileNotFoundError(f'No VESM gene-level files under {base}')
    print(f'Found {len(paths):,} VESM files', flush=True)
    required = ['gene', 'variant_key_1_meta', 'variant_key_2_meta', 'd1', 'd2', 'E_cond']
    pieces, files_with_matches, missing_cols = [], 0, []
    for idx, path in enumerate(paths, 1):
        header = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
        if not set(required).issubset(header):
            missing_cols.append((str(path), sorted(set(required) - set(header))))
            continue
        v = pd.read_csv(path, sep='\t', usecols=required, low_memory=False)
        if v.empty:
            continue
        v['gene'] = v['gene'].astype(str)
        a = v['variant_key_1_meta'].astype('string')
        b = v['variant_key_2_meta'].astype('string')
        forward = (a <= b).fillna(False).to_numpy()
        v['pair_key'] = np.where(forward, a + '__' + b, b + '__' + a)
        # Align d1 and d2 with the canonical order of genomic variant identifiers.
        old_d1 = v['d1'].copy()
        old_d2 = v['d2'].copy()
        v['d1'] = np.where(forward, old_d1, old_d2)
        v['d2'] = np.where(forward, old_d2, old_d1)
        v = v[['gene', 'pair_key', 'd1', 'd2', 'E_cond']]
        # Filter immediately to avoid storing the entire 15k-gene corpus.
        v = v.merge(valid_keys, on=['gene', 'pair_key'], how='inner', validate='many_to_one')
        if not v.empty:
            pieces.append(v)
            files_with_matches += 1
        if idx % 1000 == 0:
            print(f'  scanned {idx:,}/{len(paths):,} files; files with matches={files_with_matches:,}', flush=True)
    if missing_cols:
        print(f'WARNING: {len(missing_cols)} VESM files missing required columns. First examples: {missing_cols[:5]}', flush=True)
    if not pieces:
        raise ValueError('No VESM rows matched the ESM2 cohort; inspect genomic key conventions')
    v = pd.concat(pieces, ignore_index=True)
    print(f'VESM matching rows before deduplication: {len(v):,}', flush=True)
    keys = ['gene', 'pair_key']
    for c in ['d1', 'd2', 'E_cond']:
        v[c] = pd.to_numeric(v[c], errors='coerce')
    # After canonical orientation, all duplicate records must have identical scores.
    distinct = v.drop_duplicates(keys + ['d1', 'd2', 'E_cond'])
    conflict = distinct.duplicated(keys, keep=False)
    if conflict.any():
        conflict_path = out / 'vesm_conflicting_pairs_examples.tsv'
        distinct.loc[conflict].head(100).to_csv(conflict_path, sep='\t', index=False)
        raise ValueError(f'VESM duplicate pairs have conflicting scores. Examples: {conflict_path}')
    v = distinct.drop_duplicates(keys).copy()
    v.rename(columns={'d1': 'd1_VESM', 'd2': 'd2_VESM', 'E_cond': 'E_VESM'}, inplace=True)
    print(f'VESM unique matching pairs: {len(v):,}', flush=True)
    return v


def run_cv(dat, out):
    y = dat['observed_in_cis'].to_numpy(dtype=np.int8)
    genes = dat['gene'].to_numpy()
    af = pd.get_dummies(pd.Categorical(dat['AF_bin'], categories=AF_LABELS), dtype=float).iloc[:, 1:].to_numpy()
    folds, deltas = [], []
    splits = GroupKFold(n_splits=5).split(np.zeros(len(dat)), y, groups=genes)
    for fold, (tr, te) in enumerate(splits, 1):
        if set(genes[tr]) & set(genes[te]):
            raise AssertionError('Gene leakage')
        if np.unique(y[te]).size != 2:
            raise ValueError(f'Fold {fold} has only one outcome class')
        print(f'Fold {fold}: train={len(tr):,}, test={len(te):,}, test genes={len(set(genes[te])):,}', flush=True)
        for model in ('ESM2', 'VESM'):
            raw = dat[[f'd1_{model}', f'd2_{model}', f'E_{model}']].to_numpy(dtype=float)
            features = np.column_stack([raw, np.abs(raw[:, 2])])
            scaler = StandardScaler().fit(features[tr])
            ztr = scaler.transform(features[tr]); zte = scaler.transform(features[te])
            scores = {}
            for name, ix in zip(MODEL_NAMES, ([0,1], [0,1,2], [0,1,3])):
                xtr = np.column_stack([ztr[:, ix], af[tr]])
                xte = np.column_stack([zte[:, ix], af[te]])
                lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, tol=1e-7)
                lr.fit(xtr, y[tr])
                auc = roc_auc_score(y[te], lr.predict_proba(xte)[:, 1])
                scores[name] = auc
                folds.append({'fold':fold, 'plm':model, 'model':name, 'roc_auc':auc, 'n_test':len(te), 'n_test_genes':len(set(genes[te]))})
                print(f'  {model} | {name} | AUC={auc:.7f}', flush=True)
            for name in MODEL_NAMES[1:]:
                deltas.append({'fold':fold, 'plm':model, 'model':name, 'delta_auc':scores[name] - scores[MODEL_NAMES[0]]})
    f = pd.DataFrame(folds); d = pd.DataFrame(deltas)
    s = f.groupby(['plm','model'], sort=False).agg(mean_auc=('roc_auc','mean'), sd_auc=('roc_auc','std'), n_folds=('roc_auc','size')).reset_index()
    ds = d.groupby(['plm','model'], sort=False).agg(mean_delta_auc=('delta_auc','mean'), sd_delta_auc=('delta_auc','std')).reset_index()
    s = s.merge(ds, on=['plm','model'], how='left')
    f.to_csv(out / 'cis_gene_cv_fold_aucs.tsv', sep='\t', index=False)
    d.to_csv(out / 'cis_gene_cv_fold_delta_aucs.tsv', sep='\t', index=False)
    s.to_csv(out / 'cis_gene_cv_summary.tsv', sep='\t', index=False)
    print('\nRESULTS\n' + s.to_string(index=False, float_format=lambda x:f'{x:.7f}'), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--reg', type=Path, default=REG)
    p.add_argument('--hap', type=Path, default=HAP)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--allow-partial', action='store_true', help='Allow matched cohort smaller than 1,630,391; reports sample difference')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reg = read_reg(args.reg)
    reg = add_af(reg, args.hap)
    print(f'Validated ESM2 cohort: {len(reg):,} pairs across {reg.gene.nunique():,} genes', flush=True)
    v = read_vesm_files(args.base, reg[['gene','pair_key']], args.out)
    dat = reg.merge(v, on=['gene','pair_key'], how='inner', validate='one_to_one')
    missing = len(reg) - len(dat)
    print(f'MATCHED cohort: {len(dat):,} pairs; missing VESM scores for {missing:,} ESM2 pairs; genes={dat.gene.nunique():,}', flush=True)
    print(f'Observed cis fraction: {dat.observed_in_cis.mean():.6f}', flush=True)
    if missing and not args.allow_partial:
        raise ValueError('VESM does not cover the full ESM2 cohort. Review missingness and original VESM sample before rerunning with --allow-partial.')
    dat = dat.rename(columns={'d1':'d1_ESM2','d2':'d2_ESM2','E_cond':'E_ESM2'})
    needed = ['observed_in_cis','AF_bin','d1_ESM2','d2_ESM2','E_ESM2','d1_VESM','d2_VESM','E_VESM']
    if dat[needed].isna().any().any():
        raise ValueError('Missing values in matched dataset:\n'+str(dat[needed].isna().sum()))
    if not dat.observed_in_cis.isin([0,1]).all():
        raise ValueError('Outcome is not binary')
    # Retain audit of matched cohort for reproducibility.
    dat[['gene','pair_key','observed_in_cis']].to_csv(args.out/'matched_cohort_keys.tsv',sep='\t',index=False)
    run_cv(dat, args.out)
    print(f'Outputs saved in {args.out}', flush=True)

if __name__ == '__main__':
    main()
