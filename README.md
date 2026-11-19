**PLM Epistasis Paper Code**

This repository contains the computational pipeline and analysis scripts supporting our investigation of protein language model (PLM) epistasis in human genetic variation.

The primary analyses use ESM2-650M (`Synthyra/ESM2-650M`), with ESM2-35M/VESM-35M included for comparison and supplementary analyses.

**Pipeline**

1. Extract phased variant pairs from gnomAD.
2. Generate per-gene VCF files and annotate variants using VEP.
3. Construct single- and double-mutant protein inputs.
4. Compute PLM single-mutation and conditional epistasis scores.
5. Merge epistasis scores with variant metadata.
6. Validate single-mutation scores using ClinVar, SIFT, and PolyPhen.
7. Evaluate associations between signed and absolute conditional epistasis and AlphaFold-derived structural contacts.
8. Compare ESM2-650M and VESM-35M using matched variant pairs and gene-level cross-validation.
9. Analyze gnomAD cis co-occurrence using signed and absolute epistasis scores, including allele-frequency adjustment.
10. Perform sensitivity analyses and generate manuscript figures and supplementary results.

**Code Organization**

- `scripts/` — Preprocessing, PLM scoring, data preparation, statistical analyses, cross-validation, and figure generation.
- `scripts/legacy/` — Earlier exploratory analyses emphasizing negative conditional epistasis scores. These scripts are retained for transparency but are not the primary analyses supporting the revised manuscript.
- `configs/` — Environment configuration files for PLM scoring.
- `data/` — Information about required input datasets.
- `results/` — Information about analysis outputs and results.

**Data Availability**

Large input datasets and intermediate files are not included in this repository.

The analyses use gnomAD variant and haplotype data, VEP annotations, AlphaFold-derived structural information, and PLM-generated scores. Users must obtain the required datasets separately and prepare the corresponding input files before running the analysis scripts.

**Configuration**

Scripts use the `BASE_DIR` environment variable to specify the location of input data and analysis outputs.

For example:

```bash
export BASE_DIR="/path/to/all_genes"
```

The directory structure must match the input paths expected by each script.

**Example Usage**

To run the ESM2-650M epistasis scoring pipeline:

```bash
source configs/config_650M.env
python scripts/run_all_genes_epistasis_esmcnd_chunked_650m.py
```

To run the cis-observation regression analysis:

```bash
export BASE_DIR="/path/to/all_genes"
python scripts/cis_regression_esm2.py
```

To generate the main manuscript figures:

```bash
python scripts/plot_figure1.py
python scripts/plot_figure2.py
```

The figure-generation scripts require the corresponding intermediate analysis results.

**Reproducibility**

The repository includes scripts for the original PLM scoring pipeline and the revised statistical analyses, including matched-model comparisons, gene-level cross-validation, and allele-frequency sensitivity analyses.

Historical directional analyses are preserved separately under `scripts/legacy/` to distinguish them from the final manuscript analyses.
