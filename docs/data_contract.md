# Data contract

The code release contains no full analysis frames or full-resolution research
rasters. It includes only lightweight derived neighbourhood GeoJSON layers
required by the interactive atlas. After the separate dataset record
is extracted, set `USCANOPY_ANALYSIS_ROOT` to its root. The archived workflow
expects the following logical groups; the final data record will include a
machine-readable manifest mapping every file to these groups.

| Group | Required content | Supports |
|---|---|---|
| `Result/Result1` | city canopy metrics, display windows and validation ledgers | Result 1 and Fig. 1 |
| `Result/Result2` | boundary-aligned block-group/tract frames, ACS attributes and model tables | Result 2 and Fig. 2 |
| `Result/Result3` | fixed 34,566-unit thermal frame, spatial folds, blocks and model outputs | Result 3 and Fig. 3 |
| `Result/Result4` | G-H-V classification frame and leave-one-unit-out benchmark outputs | Result 4 and Fig. 4 |
| `Result/Result5` | equal-budget allocations, Step 1 trade-off outcomes, Step 2 structural contrasts, Step 3 transferability outputs, bootstrap draws and distribution tables | Result 5, Fig. 5 and its Discussion boundary |
| `Result/Supplementary figure and table` | source data for Supplementary Figs. 1-16 | Supplementary Information |
| `inputs/acs` | city-organized ACS geometry and attributes | geospatial examples and audits |
| `inputs/census_boundaries` | municipal and census-unit boundaries | mapping and support alignment |
| `inputs/canopy` | source imagery metadata and permitted derived masks | canopy aggregation |
| `inputs/canopy_tiles` | tiled inference inputs used by seam diagnostics | canopy-model QA |
| `inputs/annotations` | train, checkpoint-selection and diagnostic patches | canopy-model QA |
| `inputs/low_resolution_products` | harmonized 10-30 m canopy products | resolution comparison |
| `Top50城市_ACS2024_5yr/_summary` | frozen 50-city registry and state identifiers | cross-city transferability audit |
| `inputs/external_city_conditions` | archived NASA POWER responses and city-coordinate ledger | Step 3 external-predictor reconstruction |

The model record under `USCANOPY_MODEL_ROOT` is expected to include:

```text
save_weights/UNet_20260610_1605/model_best.pth
save_weights/UNet_20260610_1605/UNet.txt
save_weights/UNet_20260610_1605/results20260610_1605.txt
```

Every deposited CSV will have a data dictionary defining columns, units,
missing-value codes, denominator and related figure/table panels. Compressed
analysis frames will include SHA-256 checksums.

The final manifest will also map these semantic groups to any immutable legacy
checkpoint directories referenced by frozen scripts. Users will not need to
rename deposited files manually.
