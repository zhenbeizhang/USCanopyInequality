# Reproduction workflow

This repository separates model development, analytical reconstruction and
manuscript rendering. The code can be inspected and validated without research
data. The public Zenodo record supports verification from versioned
analysis-unit ledgers, source data and compact analytical outputs onward.
Complete reconstruction from raw imagery additionally requires licensed source
inputs and the project model record.

## 1. Prepare the environment

~~~bash
conda env create -f environment.yml
conda activate uscanopy
python -m pip install -e .
python run_pipeline.py --check-only
python -m unittest discover -s tests -v
~~~

Set `USCANOPY_ANALYSIS_ROOT` when the complete reconstruction environment is
available. Set `USCANOPY_MODEL_ROOT` when an authorized local checkpoint is
available for rerunning segmentation inference or model validation. The public
Zenodo archive follows the separate supporting-data contract described in
`docs/data_contract.md`.

## 2. Canopy mapping and aggregation

~~~bash
python scripts/01_canopy_mapping/model/infer_canopy.py --help
python run_pipeline.py --stages 1
~~~

The inference entry point converts three-band imagery to binary 0.6 m canopy
masks using overlap-averaged windows. Stage 1 then aggregates visible woody
crown context to the versioned census supports and runs annotation, geographic
split, seam, overlap and imagery-timing validation. Reference training code is
provided in `scripts/01_canopy_mapping/model/train_model.py`. Reported summaries
and figure source data are available in the Zenodo record; full inference also
requires the licensed raw imagery, full-resolution masks and checkpoint.

## 3. Social-canopy analysis

~~~bash
python run_pipeline.py --stages 2
~~~

Stage 2 reconstructs the municipal-boundary population frame, fits the primary
within-city models, assesses the complete-case suppression result, and evaluates
block-group-to-tract transferability and boundary sensitivity. The primary
social frame contains 34,575 effectively contained block groups; matched
cross-scale analyses use the support-specific versioned ledgers described in
docs/analysis_frames.md.

## 4. Surface-temperature analysis

~~~bash
python run_pipeline.py --stages 3
~~~

Stage 3 fits the support-aligned model sequence, applies spatial block
bootstrap inference, compares paired model specifications and assesses nonlinear
functional forms. Outputs retain city-level estimates and national REML-pooled
summaries. Landsat LST is treated as clear-sky surface temperature, not personal
heat exposure.

## 5. Neighbourhood typology

~~~bash
python run_pipeline.py --stages 4
~~~

Stage 4 classifies each of the 34,566 complete units by city-relative canopy
deficit, positive LST anomaly and the social-condition ordering. It constructs the 27-cell
G-H-V typology, calculates the joint-context benchmark, repeats it after
leaving each component out, and evaluates threshold sensitivity. The benchmark
quantifies descriptive co-location under the stated supports.

## 6. Equal-budget allocation comparison

~~~bash
python run_pipeline.py --stages 5
~~~

Stage 5 specifies and verifies the analytical protocol before comparing fixed
area ranking objectives. All rules receive the same area budget; only ranking
changes. It then derives the submitted overlap and directional opportunity-loss
estimands and runs the controlled capacity-intensity and candidate-screen
decomposition at 5%, 10% and 20% budgets. Outputs quantify model-implied
trade-offs under equal-area analytical budgets.

## 7. Cross-city transferability assessment

~~~bash
python run_pipeline.py --stages transfer
~~~

This stage builds six externally defined city-condition predictors and tests
out-of-city prediction of the versioned heat-context opportunity loss. It uses
nested leave-one-city-out and Census-region holdouts, outcome-bootstrap
uncertainty and a permutation test. The prespecified evaluation found limited
predictive skill, supporting the manuscript's city-heterogeneity interpretation.

## 8. Figures, tables and QA

~~~bash
python run_pipeline.py --stages figures qa
~~~

This regenerates five main figures, seventeen Supplementary figures and the
source-table ledgers, then runs figure-specific and reporting-consistency
checks when the complete analytical-output tree is available. The
Fig. 2 script also emits Supplementary Fig. 17; the dedicated Supplementary
directory contains the other sixteen scripts. The Zenodo record already maps
every submitted panel to its archived source files.

## Complete run

~~~bash
python run_pipeline.py --stages 1 2 3 4 5 transfer figures qa
~~~

Use `--dry-run` to print the ordered commands without executing them, or
`--continue-on-error` only for diagnostic batch runs. A complete run is
computationally substantial and depends on third-party geospatial inputs whose
licences are documented in the data record. The minimum public archive supports
inspection from versioned analysis ledgers onward.

## Determinism and expected variation

Fixed random seeds are retained where they were used. Minor floating-point
differences may arise across BLAS, GDAL, CPU and GPU builds. Reproduced claims
should be checked against archived source tables and uncertainty intervals,
not against identical raster bytes or last-decimal equality.
