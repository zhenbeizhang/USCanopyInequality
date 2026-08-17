# USCanopyInequality

Reproducible code for the analysis of urban tree-canopy inequality, summer
land-surface-temperature associations and equal-budget allocation trade-offs
across 50 US cities.

[![static-audit](https://github.com/zhenbeizhang/USCanopyInequality/actions/workflows/static-audit.yml/badge.svg)](https://github.com/zhenbeizhang/USCanopyInequality/actions/workflows/static-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Interactive atlas](https://img.shields.io/badge/live_atlas-open_map-176b55.svg)](http://8.217.164.241/atlas/)

## Interactive evidence atlas

### [Open the live US Urban Canopy Evidence Atlas](http://8.217.164.241/atlas/)

**Public viewer:** [http://8.217.164.241/atlas/](http://8.217.164.241/atlas/)

The companion **US Urban Canopy Evidence Atlas** turns the analytical outputs
into a city-first map. It opens with the 50 study cities, then allows users to
select a city and inspect:

- the native 0.6 m visible-crown mask over a public basemap;
- neighbourhood canopy deficit, social-condition ordering and positive summer
  surface-heat anomaly;
- six interpretable neighbourhood contexts derived from the 27 nominal G/H/V
  combinations; and
- equal-area analytical allocations under canopy-gap, surface-heat,
  social-context and balanced ordering rules.

The viewer is an evidence browser, not a parcel-level planting plan. Its
display-only six-pixel component filter improves map legibility without
changing the archived masks or any reported statistic.

**[Open the live atlas](http://8.217.164.241/atlas/)** ·
**[Read the atlas documentation](atlas/README.md)** ·
**[View the full-size interface preview](atlas/docs/assets/atlas-preview.png)** ·
**[Read the deployment design](atlas/docs/deployment.md)**

![City-scale canopy and neighbourhood evidence viewer](atlas/docs/assets/atlas-preview.png)

The public deployment serves the web client, its whitelisted query and tile
API, and the city catalogue. Full-resolution Cloud-Optimized GeoTIFFs remain
outside Git because they are data-release assets rather than source code.

## Release status

This repository is the **code component** of a two-record reproducibility
release. It contains the analysis, figure and quality-control code, together
with lightweight derived neighbourhood layers needed by the interactive atlas.
The frozen analysis frames, source data, full-resolution canopy masks, model
checkpoint and figure source data will be deposited as a separate dataset and
linked here by a persistent identifier. Restricted third-party imagery and
full-resolution research rasters are not stored in this repository.

Release `0.3.0` is synchronized to the submitted analysis freeze. It includes
the frozen equal-city allocation overlap, algebraic shared/redirected-area
translation, directional opportunity losses, controlled structural
decomposition and the cross-city transferability audit used to delimit the
Discussion claim. The neighbourhood web layer retains its independent
`GHV27-2026-08-09` data-freeze identifier for the 34,566 neighbourhood
classifications used by the typology and allocation modules.

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

Copy `configs/environment.example` to a local shell configuration and set the
paths after the data record is released. If variables are omitted, the code
uses `work/`, `model_work/`, `manuscript/` and `external/` below the repository
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

See `docs/data_contract.md` before running an analysis stage and
`docs/workflow.md` for stage-level commands and outputs.

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

Until the article and archived software release receive persistent identifiers,
cite this repository using `CITATION.cff`. Code is released under the MIT
License. Data will receive a separate licence compatible with source-data
rights; this code licence does not apply to third-party imagery or datasets.
