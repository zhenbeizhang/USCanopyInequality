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
- generation paths for five main figures and seventeen Supplementary figures,
  source tables and independent QA checks. The Fig. 2 script also emits
  Supplementary Fig. 17, while the dedicated Supplementary-figure directory
  contains the other sixteen scripts.

## Published minimum supporting dataset

Zenodo record https://doi.org/10.5281/zenodo.21976810 contains:

- frozen block-group and tract analysis-unit ledgers and the fixed
  neighbourhood-context frame;
- source data for five main figures, seventeen Supplementary figures and eight
  Supplementary tables;
- compact final estimates, uncertainty summaries and bootstrap draws for
  Results 2-5;
- 50 licence-permitted neighbourhood GeoJSON layers and reference boundaries;
- random seeds, provenance, source-data maps, restrictions, data dictionaries
  and file-level checksums.

## Excluded or restricted

Raw high-resolution imagery, full-resolution 0.6 m canopy masks, third-party
raster downloads, model-development patches and checkpoints, temporary caches
and superseded analyses are not part of this code repository or the Zenodo
record. Redistribution rights and complete upstream acquisition metadata are
not held consistently for the raw imagery. No licence for excluded files is
implied here.

The 95-patch diagnostic set is internal and partly overlaps model-development
footprints. It is not an independent 50-city validation set. Product
comparisons quantify disagreement and do not establish that the 0.6 m atlas is
more accurate than every 10-30 m product.

## Interpretation limits

The code preserves the estimands used in the manuscript. It does not convert
visible crown to cast shade, Landsat LST to personal exposure,
cross-sectional associations to causal cooling, or analytical canopy-deficit
capacity to parcel-level plantability. Allocation outputs are model-implied
diagnostic quantities under equal-area budgets.

The public package supports code inspection, no-data release checks and
verification from archived ledgers, source data and compact analytical outputs.
It does not claim a complete raw-imagery-to-trained-model reconstruction.

