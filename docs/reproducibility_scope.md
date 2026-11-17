# Reproducibility scope

## Code coverage

- checkpoint-compatible U-Net architecture;
- reference training and stride/overlap inference entry points;
- segmentation validation, split, seam and overlap assessments;
- census-boundary and population-allocation reconstruction;
- social-canopy fixed-effect, complete-case and cross-scale assessments;
- support-aligned canopy-LST models, spatial cross-validation, block bootstrap
  and nonlinear-form assessment;
- 27-type neighbourhood classification and leave-one-unit-out co-location
  benchmarks;
- equal-budget multiobjective allocation, demographic distribution and
  sensitivity analyses;
- generation paths for five main figures and seventeen Supplementary figures,
  source tables and independent QA checks. The Fig. 2 script also emits
  Supplementary Fig. 17, while the dedicated Supplementary-figure directory
  contains the other sixteen scripts.

## Published supporting dataset

Zenodo record https://doi.org/10.5281/zenodo.21976809 contains:

- versioned block-group and tract analysis-unit ledgers and the fixed
  neighbourhood-context frame;
- source data for five main figures, seventeen Supplementary figures and eight
  Supplementary tables;
- compact final estimates, uncertainty summaries and bootstrap draws for
  Results 2-5;
- 50 licence-permitted neighbourhood GeoJSON layers and reference boundaries;
- random seeds, provenance, source-data maps, restrictions, data dictionaries
  and file-level checksums.

## Access and licensing

Access to raw high-resolution imagery, full-resolution 0.6 m canopy masks,
third-party raster downloads, model-development patches and checkpoints
follows the licences and metadata held by their source providers. The Zenodo
record therefore focuses on author-shareable analysis ledgers, source data,
compact outputs and derived spatial layers.

The 95-patch diagnostic set is author-generated and includes some geographic
overlap with model-development footprints; it characterizes performance within
that support. Public-product comparisons quantify agreement, neighbourhood
ranking differences and social-gradient sensitivity across products.

## Interpretation scope

The code preserves the estimands used in the manuscript: visible crown,
clear-sky Landsat surface temperature, cross-sectional spatial associations and
analytical canopy-deficit capacity. Cast shade, personal exposure, causal
cooling and parcel-level plantability require additional measurements and
designs. Allocation outputs quantify model-implied trade-offs under equal-area
budgets.

The public package supports code inspection, no-data release checks and
verification from archived ledgers, source data and compact analytical
outputs.

