#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from run_cis_gene_cv_from_gene_files_AFfix import BASE, REG, HAP, OUT, AF_LABELS, read_reg, add_af, read_vesm_files

KEY = ['gene', 'pair_key']
HCOUNTS = [f'all_hap_counts_{i}' for i in range(4)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=BASE)
    ap.add_argument('--reg', type=Path, default=REG)
    ap.add_argument('--hap', type=Path, default=HAP)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print('=== COHORT AUDIT ===', flush=True)
    reg = read_reg(args.reg)
    print(f'Validated ESM2 rows: {len(reg):,}; genes: {reg.gene.nunique():,}; cis-positive: {int(reg.observed_in_cis.sum()):,}', flush=True)
    reg = add_af(reg, args.hap)
    print('Reconstructed AF-bin counts:', reg.AF_bin.value_counts(dropna=False).reindex(AF_LABELS).to_dict(), flush=True)
    print('Scanning original VESM gene files', flush=True)
    vesm = read_vesm_files(args.base, reg[KEY], args.out)
    matched = reg.merge(vesm, on=KEY, how='left', validate='one_to_one', indicator=True)
    missing = matched['_merge'].eq('left_only')
    print(f'VESM-matched ESM2 pairs: {(~missing).sum():,} / {len(reg):,} ({100*(~missing).mean():.5f}%)', flush=True)
    print(f'ESM2 pairs missing VESM: {missing.sum():,}', flush=True)
    print(f'Matched genes: {matched.loc[~missing,"gene"].nunique():,}', flush=True)
    print(f'Unmatched genes: {matched.loc[missing,"gene"].nunique():,}', flush=True)
    if missing.any():
        missing_path = args.out / 'audit_missing_vesm_pairs.tsv'
        matched.loc[missing, KEY + ['observed_in_cis']].to_csv(missing_path, sep='\t', index=False)
        print('Missing VESM keys saved:', missing_path, flush=True)
    print('Matched cis-positive fraction:', f'{matched.loc[~missing,"observed_in_cis"].mean():.7f}', flush=True)
    print('Unmatched cis-positive fraction:', f'{matched.loc[missing,"observed_in_cis"].mean():.7f}' if missing.any() else 'N/A', flush=True)

    # Independently recompute the AF product using the original ESM2 formula.
    # Restrict haplotype records to the validated cohort; detect conflicting duplicates.
    print('\n=== AF-PRODUCT AUDIT ===', flush=True)
    h = pd.read_csv(args.hap, sep='\t', usecols=KEY + HCOUNTS, low_memory=False)
    h[KEY] = h[KEY].astype(str)
    h = h.merge(reg[KEY], on=KEY, how='inner', validate='many_to_one')
    for col in HCOUNTS:
        h[col] = pd.to_numeric(h[col], errors='coerce')
    distinct = h.drop_duplicates(KEY + HCOUNTS)
    conflicts = distinct.duplicated(KEY, keep=False)
    print('Conflicting duplicate haplotype keys:', int(distinct.loc[conflicts, KEY].drop_duplicates().shape[0]), flush=True)
    if conflicts.any():
        distinct.loc[conflicts].head(100).to_csv(args.out/'audit_hap_conflicts.tsv', sep='\t', index=False)
        raise ValueError('Conflicting haplotype counts; inspect audit_hap_conflicts.tsv')
    h = distinct.drop_duplicates(KEY)
    n = h[HCOUNTS].sum(axis=1, min_count=4)
    af1 = (h['all_hap_counts_1'] + h['all_hap_counts_3']) / n
    af2 = (h['all_hap_counts_2'] + h['all_hap_counts_3']) / n
    product = af1 * af2
    print('Missing/nonfinite AF products:', int((~np.isfinite(product.to_numpy(dtype=float))).sum()), flush=True)
    print('Zero AF products:', int((product == 0).sum()), flush=True)
    print('Nonpositive haplotype totals:', int((n <= 0).sum()), flush=True)
    # Original scripts: pd.cut(..., bins=[-inf,1e-8,1e-7,1e-6,inf], right=False)
    h['AF_bin_original'] = pd.cut(product, bins=[-np.inf,1e-8,1e-7,1e-6,np.inf], labels=AF_LABELS, right=False)
    comp = reg[KEY + ['AF_bin']].merge(h[KEY + ['AF_bin_original']], on=KEY, how='left', validate='one_to_one')
    mismatch = comp['AF_bin'].astype('string').fillna('<NA>') != comp['AF_bin_original'].astype('string').fillna('<NA>')
    print(f'AF-bin mismatches: {int(mismatch.sum()):,} / {len(comp):,}', flush=True)
    if mismatch.any():
        path = args.out/'audit_af_bin_mismatches.tsv'
        comp.loc[mismatch].head(1000).to_csv(path, sep='\t', index=False)
        print('AF mismatch examples:', path, flush=True)
    print('AF-bin counts (independent reconstruction):')
    print(h['AF_bin_original'].value_counts(dropna=False).reindex(AF_LABELS).to_string(), flush=True)

    # Compare keys persisted by the completed CV run, not merely the prospective cohort.
    saved = args.out / 'matched_cohort_keys.tsv'
    if saved.exists():
        prev = pd.read_csv(saved, sep='\t', dtype={'gene':str,'pair_key':str})
        expected = matched.loc[~missing, KEY + ['observed_in_cis']]
        check = expected.merge(prev, on=KEY, how='outer', suffixes=('_expected','_saved'), indicator=True, validate='one_to_one')
        only_expected = int(check['_merge'].eq('left_only').sum())
        only_saved = int(check['_merge'].eq('right_only').sum())
        both = check['_merge'].eq('both')
        changed_y = int((check.loc[both,'observed_in_cis_expected'] != check.loc[both,'observed_in_cis_saved']).sum())
        print('\n=== COMPLETED RUN AUDIT ===', flush=True)
        print(f'Saved matched cohort: {len(prev):,} rows', flush=True)
        print(f'Expected but not saved: {only_expected:,}; saved but not expected: {only_saved:,}; differing outcomes: {changed_y:,}', flush=True)
        keys_pass = (only_expected == 0 and only_saved == 0 and changed_y == 0)
    else:
        print('\nWARNING: matched_cohort_keys.tsv not found; cannot verify actual CV input keys.', flush=True)
        keys_pass = False
    complete = not missing.any()
    af_pass = not mismatch.any() and np.isfinite(product.to_numpy(dtype=float)).all() and (n > 0).all()
    print('\n=== FINAL CHECKS ===', flush=True)
    print('Complete VESM matching:', 'PASS' if complete else 'REVIEW')
    print('AF definition and bins:', 'PASS' if af_pass else 'REVIEW')
    print('Saved CV input matches reconstruction:', 'PASS' if keys_pass else 'REVIEW')
    if not (complete and af_pass and keys_pass):
        print('Do not finalize figure until REVIEW items are resolved.', flush=True)
    else:
        print('All three audits passed.', flush=True)

if __name__ == '__main__':
    main()
