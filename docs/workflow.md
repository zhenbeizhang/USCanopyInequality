# Reproduction workflow

This repository separates model development, analytical reconstruction and
manuscript rendering. The code can be inspected and audited without research
data. Numerical reproduction requires the separately archived data record and,
for canopy inference, the separately archived model checkpoint.

## 1. Prepare the environment

~~~bash
conda env create -f environment.yml
conda activate uscanopy
python -m pip install -e .
python run_pipeline.py --check-only
python -m unittest discover -s tests -v
~~~

Set USCANOPY_ANALYSIS_ROOT to the extracted analysis-data record. Set
USCANOPY_MODEL_ROOT only when rerunning segmentation inference or its model
audits. The exact logical inputs are listed in docs/data_contract.md.

## 2. Canopy mapping and aggregation

~~~bash
python scripts/01_canopy_mapping/model/infer_canopy.py --help
python run_pipeline.py --stages 1
~~~

The inference entry point converts three-band imagery to binary 0.6 m canopy
masks using overlap-averaged windows. Stage 1 then aggregates visible woody
crown context to the frozen census supports and runs annotation, geographic
split, seam, overlap and imagery-timing audits. Reference training code is
provided in scripts/01_canopy_mapping/model/train_model.py; rerunning training
is not required to reproduce the manuscript analyses from archived masks.

## 3. Social-canopy analysis

~~~bash
python run_pipeline.py --stages 2
~~~

Stage 2 reconstructs the municipal-boundary population frame, fits the primary
within-city models, audits the complete-case suppression result, and evaluates
block-group-to-tract transferability and boundary sensitivity. The primary
social frame contains 34,575 effectively contained block groups; matched
cross-scale analyses use the support-specific frozen ledgers described in
docs/analysis_frames.md.

## 4. Surface-temperature analysis

~~~bash
python run_pipeline.py --stages 3
~~~

Stage 3 fits the support-aligned model sequence, applies spatial block
bootstrap inference, compares paired model specifications and audits nonlinear
functional forms. Outputs retain city-level estimates and national REML-pooled
summaries. Landsat LST is treated as clear-sky surface temperature, not personal
heat exposure.

## 5. Neighbourhood typology

~~~bash
python run_pipeline.py --stages 4
~~~

Stage 4 classifies each of the 34,566 complete units by city-relative canopy
deficit, positive LST anomaly and social condition. It constructs the 27-cell
G-H-V typology, calculates the joint-context benchmark, repeats it after
leaving each component out, and evaluates threshold sensitivity. The benchmark
is descriptive co-location evidence, not a causal risk score.

## 6. Equal-budget allocation comparison

~~~bash
python run_pipeline.py --stages 5
~~~

Stage 5 freezes and verifies the analytical protocol before comparing fixed
area ranking objectives. All rules receive the same area budget; only ranking
changes. It then derives the submitted overlap and directional opportunity-loss
estimands and runs the controlled capacity-intensity and candidate-screen
decomposition at 5%, 10% and 20% budgets. Outputs are model-implied diagnostic
quantities, not parcel-level planting plans or realised cooling benefits.

## 7. Cross-city transferability boundary

~~~bash
python run_pipeline.py --stages transfer
~~~

This stage builds six externally defined city-condition predictors and tests
whether they predict the frozen heat-context opportunity loss outside the
fitted city. It uses nested leave-one-city-out and Census-region holdouts,
outcome-bootstrap uncertainty and a permutation test. Its prespecified skill
gate failed, so the result is used only to delimit transferability rather than
to propose a city taxonomy or causal explanation.

## 8. Figures, tables and QA

~~~bash
python run_pipeline.py --stages figures qa
~~~

This regenerates five main figures, sixteen Supplementary figures and the
source-table ledgers, then runs figure-specific and reporting-consistency
checks. Figure scripts consume frozen analytical outputs rather than refitting
models. The separate data record will map every panel to its source files.

## Complete run

~~~bash
python run_pipeline.py --stages 1 2 3 4 5 transfer figures qa
~~~

Use --dry-run to print the ordered commands without executing them, or
--continue-on-error only for diagnostic batch runs. A complete run is
computationally substantial and depends on third-party geospatial inputs whose
licences are documented in the data record.

## Determinism and expected variation

Frozen random seeds are retained where they were used. Minor floating-point
differences may arise across BLAS, GDAL, CPU and GPU builds. Reproduced claims
should be checked against archived source tables and uncertainty intervals,
not against identical raster bytes or last-decimal equality.
