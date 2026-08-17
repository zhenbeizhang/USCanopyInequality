# Reproducibility scope

## Included in the code release

- checkpoint-compatible U-Net architecture;
- reference training and stride/overlap inference entry points;
- segmentation validation, split, seam and overlap audits;
- census-boundary and population-allocation reconstruction;
- social-canopy fixed-effect, complete-case and cross-scale audits;
- support-aligned canopy-LST models, spatial cross-validation, block bootstrap
  and nonlinear-form audit;
- 27-type neighbourhood classification and leave-one-unit-out co-location
  benchmarks;
- equal-budget multiobjective allocation, demographic distribution and
  sensitivity analyses;
- scripts for five main figures, sixteen Supplementary figures, source tables
  and independent QA checks.

## Deferred to the separate data release

- frozen block-group and tract analysis frames;
- figure and table source data;
- 50 city canopy masks where redistribution is permitted;
- model checkpoint, training log and model configuration;
- Landsat, ACS, Census-boundary and public-product derivatives;
- file-level data checksums and data dictionaries.

## Excluded or restricted

Raw high-resolution imagery is not part of this code repository. Its upstream
provider and redistribution licence are not completely recoverable from the
archived map-aggregation cache. No licence for those files is implied here.

The 95-patch diagnostic set is internal and partly overlaps model-development
footprints. It is not an independent 50-city validation set. Product
comparisons quantify disagreement and do not establish that the 0.6 m atlas is
more accurate than every 10-30 m product.

## Interpretation limits

The code preserves the estimands used in the manuscript. It does not convert
visible crown to cast shade, Landsat LST to personal exposure, cross-sectional
associations to causal cooling, or analytical canopy-deficit capacity to
parcel-level plantability. Allocation outputs are model-implied diagnostic
quantities under equal-area budgets.

