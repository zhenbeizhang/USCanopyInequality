# Data contract

## Public minimum supporting dataset

The published supporting dataset is available at
https://doi.org/10.5281/zenodo.21976809. It is organized for transparent
inspection of the submitted evidence as a structured public record. Its
top-level groups are:

| Group | Public content |
|---|---|
| `analysis_units/` | Versioned block-group and tract ledgers and the fixed neighbourhood-context frame |
| `source_data/` | Panel-level source files for five main figures, seventeen Supplementary figures and eight Supplementary tables |
| `model_outputs/` | Compact final estimates, uncertainty summaries and bootstrap draws for Results 2-5 |
| `derived_spatial_data/` | Fifty neighbourhood GeoJSON files and reference boundaries used by the evidence atlas |
| `metadata/` | File catalogue, dictionaries, provenance, third-party sources, random seeds, restrictions and source-data map |
| `SHA256SUMS.txt` | File-level integrity checks |

The public archive begins from versioned analysis ledgers and supports
verification through source data and compact analytical outputs. Raw
high-resolution imagery, full-resolution canopy masks, third-party raster
downloads, model-development patches and checkpoints remain governed by their
source records and redistribution terms.

## Complete reconstruction environment

The stage scripts preserve the following logical input contract for authorized
users who hold the complete reconstruction environment. Set
`USCANOPY_ANALYSIS_ROOT` to that tree only when these groups are available.
The published Zenodo archive should not be used directly as this root.

| Group | Required content | Supports |
|---|---|---|
| `Result/Result1` | city canopy metrics, display windows and validation ledgers | Result 1 and Fig. 1 |
| `Result/Result2` | boundary-aligned block-group/tract frames, ACS attributes and model tables | Result 2 and Fig. 2 |
| `Result/Result3` | fixed 34,566-unit thermal frame, spatial folds, blocks and model outputs | Result 3 and Fig. 3 |
| `Result/Result4` | G-H-V classification frame and leave-one-unit-out benchmark outputs | Result 4 and Fig. 4 |
| `Result/Result5` | equal-budget allocations, primary trade-off outcomes, structural-decomposition contrasts, cross-city transferability outputs, bootstrap draws and distribution tables | Result 5, Fig. 5 and its Discussion interpretation |
| `Result/Supplementary figure and table` | source data for Supplementary Figs. 1-17 | Supplementary Information |
| `inputs/acs` | city-organized ACS geometry and attributes | geospatial examples and assessments |
| `inputs/census_boundaries` | municipal and census-unit boundaries | mapping and support alignment |
| `inputs/canopy` | source imagery metadata and permitted derived masks | canopy aggregation |
| `inputs/canopy_tiles` | tiled inference inputs used by seam diagnostics | canopy-model QA |
| `inputs/annotations` | train, checkpoint-selection and diagnostic patches | canopy-model QA |
| `inputs/low_resolution_products` | harmonized 10-30 m canopy products | resolution comparison |
| `Top50城市_ACS2024_5yr/_summary` | versioned 50-city registry and state identifiers | cross-city transferability assessment |
| `inputs/external_city_conditions` | archived NASA POWER responses and city-coordinate ledger | cross-city external-predictor reconstruction |

Rerunning segmentation inference additionally requires the project model
record under `USCANOPY_MODEL_ROOT`, including:

```text
save_weights/UNet_20260610_1605/model_best.pth
save_weights/UNet_20260610_1605/UNet.txt
save_weights/UNet_20260610_1605/results20260610_1605.txt
```

The model checkpoint is not part of the public release. The repository provides
the checkpoint-compatible architecture and reference training and inference
code, while the Zenodo record provides final model summaries needed to inspect
the manuscript results.

The Zenodo archive contains a file catalogue, key-variable dictionary,
panel-level source-data map, provenance notes and SHA-256 checksums. Legacy
paths retained by versioned scripts are implementation details of the full
reconstruction tree and are not public-data access routes.
