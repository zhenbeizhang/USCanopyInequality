# USCanopyInequality

Reproducible code for the analysis of urban tree-canopy inequality, summer
land-surface-temperature associations and equal-budget allocation trade-offs
across 50 US cities.

[![static-audit](https://github.com/zhenbeizhang/USCanopyInequality/actions/workflows/static-audit.yml/badge.svg)](https://github.com/zhenbeizhang/USCanopyInequality/actions/workflows/static-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21976809.svg)](https://doi.org/10.5281/zenodo.21976809)
[![Interactive atlas](https://img.shields.io/badge/live_atlas-open_map-176b55.svg)](https://atlas.8-217-164-241.sslip.io/atlas/)

## Interactive evidence atlas

### [Open the live US Urban Canopy Evidence Atlas](https://atlas.8-217-164-241.sslip.io/atlas/)

**Public viewer:** [https://atlas.8-217-164-241.sslip.io/atlas/](https://atlas.8-217-164-241.sslip.io/atlas/)

The companion **US Urban Canopy Evidence Atlas** turns the analytical outputs
into a city-first map. It opens with the 50 study cities, then allows users to
select a city and inspect:

- the native 0.6 m visible-crown mask over a public basemap;
- neighbourhood canopy deficit, social-condition ordering and positive summer
  surface-heat anomaly;
- six interpretable neighbourhood contexts derived from the 27 nominal G/H/V
  combinations; and
- equal-area analytical allocations under canopy-deficit, surface-heat,
  social-condition and illustrative equal-weight ordering rules.

The viewer is an evidence browser, not a parcel-level planting plan. Its
display-only six-pixel component filter improves map legibility without
changing the archived masks or any reported statistic.

- **[Open the live atlas](https://atlas.8-217-164-241.sslip.io/atlas/)**
- **[Read the atlas documentation](atlas/README.md)**
- **[View the full-size interface preview](atlas/docs/assets/atlas-preview.png)**
- **[Read the deployment design](atlas/docs/deployment.md)**

![City-scale canopy and neighbourhood evidence viewer](atlas/docs/assets/atlas-preview.png)

The public deployment serves the web client, its whitelisted query and tile
API, and the city catalogue. Full-resolution Cloud-Optimized GeoTIFFs are
deployment assets rather than source code and are not included in Git or the
minimum Zenodo dataset.

## Release status

This repository is the **code component** of the public reproducibility
package. It contains the analysis, figure and quality-control code, together
with lightweight derived neighbourhood layers needed by the interactive atlas.
The minimum supporting dataset is available from
[Zenodo](https://doi.org/10.5281/zenodo.21976809). It contains frozen
analysis-unit ledgers, source data for all five main figures, seventeen
Supplementary figures and eight Supplementary tables, compact model and
bootstrap outputs, random seeds, checksums and licence-permitted derived
spatial data.

The Zenodo record is not a raw-input or end-to-end reconstruction archive. Raw
high-resolution imagery, full-resolution canopy masks and third-party raster
downloads are excluded where redistribution rights or complete upstream
metadata are unavailable. Model-development patches and checkpoints are also
outside the scope of this minimum supporting dataset.

Release `0.3.1` preserves the submitted analysis freeze and updates the public
archive description, licensing and release checks without changing analytical
frames, estimands or numerical outputs. It includes the frozen equal-city
allocation overlap, algebraic shared/redirected-area translation, directional
opportunity losses, controlled structural decomposition and the cross-city
transferability audit used to delimit the Discussion claim. The neighbourhood
web layer retains its independent `GHV27-2026-08-09` data-freeze identifier for
the 34,566 neighbourhood classifications used by the typology and allocation
modules.

The code snapshot corresponds to the manuscript analysis with:

- 38,618 municipal block-group intersections for the descriptive canopy atlas;
- 34,575 effectively contained block groups for the primary social analysis;
- 34,566 complete block groups for the surface-heat, neighbourhood-typology and
  allocation analyses;
- 50 cities in every national analysis.

## What the repository reproduces

| Module | Question | Main entry points |
|---|---|---|
| 1. Canopy mapping | How do citywide canopy supply and resident context differ? | `scripts/01_canopy_mapping/` |
| 2. Social inequality | Which within-city social gradients recur across supports and scales? | `scripts/02_social_inequality/` |
| 3. Surface heat | How is canopy associated with summer Landsat LST after environmental and spatial adjustment? | `scripts/03_surface_heat/` |
| 4. Neighbourhood typology | Where do canopy deficit, positive surface heat and social conditions co-occur? | `scripts/04_neighbourhood_typology/` |
| 5. Allocation | How do equal-area objectives trade off, what shared constraints alter their divergence, and does that divergence transfer among cities? | `scripts/05_allocation/` |

Main-figure, Supplementary-figure and independent audit scripts are under
`scripts/06_main_figures/`, `scripts/07_supplementary_figures/` and
`scripts/08_qa/`.

## Installation

The frozen analysis used Python 3.12 on Windows 11. A cross-platform Conda
environment is provided:

```bash
conda env create -f environment.yml
conda activate uscanopy
python -m pip install -e .
```

For a pip-only installation:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

GPU-specific PyTorch builds may require installation from the official PyTorch
index before running canopy inference or training.

## Configure paths

The public scripts contain no workstation-specific drive paths. They read four
optional environment variables:

```text
USCANOPY_ANALYSIS_ROOT   extracted analysis-data record
USCANOPY_MODEL_ROOT      checkpoint and model-development record
USCANOPY_MANUSCRIPT_ROOT optional manuscript files used by reporting audits
USCANOPY_EXTERNAL_ROOT   third-party inputs not included in either record
```

These variables describe the complete internal reconstruction tree. The public
Zenodo record is organized for inspection of frozen ledgers, source data and
compact outputs and should not be used directly as
`USCANOPY_ANALYSIS_ROOT` or `USCANOPY_MODEL_ROOT`. Authorized users with the
full inputs may copy `configs/environment.example` to a local shell
configuration and set the paths. If variables are omitted, the code uses
`work/`, `model_work/`, `manuscript/` and `external/` below the repository
root.

Check the code release without data:

```bash
python run_pipeline.py --check-only
python -m unittest discover -s tests -v
```

Preview the execution order:

```bash
python run_pipeline.py --dry-run --stages 1 2 3 4 5 transfer figures qa
```

See `docs/data_contract.md` before running an analysis stage,
`docs/workflow.md` for stage-level commands and outputs, and
`docs/terminology.md` for the controlled public vocabulary.

## Release checks

The repository-level audit verifies syntax, expected entry points, absence of
workstation-specific absolute path literals and exclusion of private binary
artifacts. The public U-Net definition was also checked against the frozen
checkpoint: strict loading succeeded at epoch 495 with no missing or unexpected
keys (17,554,457 parameters; checkpoint SHA-256 recorded in
`manifests/checkpoint_compatibility.json`). The checkpoint itself is not
included in this code-only release.

## Scientific boundaries

- Canopy denotes mapped visible woody-crown context, not cast shade or
  individual access.
- Landsat LST denotes clear-sky surface temperature, not air temperature,
  personal exposure or health risk.
- The canopy-LST models are cross-sectional associations, not causal estimates
  of cooling from added canopy.
- Analytical canopy-deficit capacity is not verified plantable land.
- Objective-specific allocation results are diagnostic mathematical quantities,
  not planting plans or realized benefits.
- Raw commercial imagery is excluded until redistribution rights are resolved.

Further limitations are documented in `docs/reproducibility_scope.md`.

## Citation and licence

For the code, cite this repository using `CITATION.cff`. The supporting data
record is available from [Zenodo](https://doi.org/10.5281/zenodo.21976809).
Code is released under the MIT License. Author-generated derived data tracked
in this repository and the Zenodo record are released under CC BY 4.0 to the
extent that the authors hold rights in them; see [DATA_LICENSE.md](DATA_LICENSE.md).
Neither licence applies to excluded third-party imagery or datasets.
