# Step 3 frozen protocol: cross-city transferability of allocation divergence

**Status:** frozen before predictor-outcome association, model fitting or outcome-tuned variable selection  
**Date:** 12 August 2026  
**Parent modules:** Step 1 frozen city outcomes and Step 2 controlled structural decomposition  
**Scope:** out-of-city prediction and descriptive transportability, not causal attribution

All stochastic operations use the fixed seed `20260812`. Bootstrap and
permutation replicate identifiers are retained in the output tables.

## Question

Can a small, independently defined set of climate, built-form, water-context
and terrain conditions predict how much surface-heat-context quantity is
forgone when a city follows the social-context allocation rule?

The analysis tests transferability across the 50 sampled cities. It does not
estimate why any city acquired its observed social or thermal geography and it
does not claim transportability to cities outside the sampled size and
geographic frame unless the regional holdout test also succeeds.

## Frozen outcome

The sole primary outcome is the Step 1 quantity

    L_Hc = 1 - Q_Hc(x_S) / Q_Hc(x_H)

under `observed_gap_capacity` at the 10% equal-area analytical budget. Larger
values indicate that the social-context ordering retains less of the
surface-heat-context rule's mathematical upper bound. The 5% and 20% outcomes
are frozen budget sensitivities. Allocation overlap and social-context loss
under the heat allocation are not alternative Step 3 targets.

All outcomes are model-defined ratios. They are not realised cooling, personal
heat exposure, health effects or welfare.

## Prohibited proximal predictors

No variable used to define G, H, V, analytical capacity, either ranking score
or Step 2 structural contrasts may enter a predictive model. The following are
therefore excluded: canopy cover or canopy deficit; LST or H summaries;
social-condition indicators or V summaries; population or density; capacity,
budget or concentration; G-H-V type shares; H-V or score correlations; and
allocation outputs. These quantities may not be introduced after results are
seen.

## Six frozen external predictors

All predictors are city-level and defined without reference to the outcome.

### Long-term climate domain

1. `jja_air_temperature_C`: day-weighted mean of June, July and August 2-m air
   temperature for 1991-2020 at the municipal centroid from NASA POWER `T2M`.
2. `jja_precipitation_mm`: June-August precipitation total for 1991-2020,
   computed from NASA POWER `PRECTOTCORR` daily rates and calendar-month days.

These are external MERRA-2-based climate conditions, not Landsat LST inputs.

### Urban-environment domain

3. `built_fraction_area_weighted_mean_pct`: city land-area-weighted mean of
   census-block-group ESA WorldCover 2020 built-up percentage.
4. `built_fraction_area_weighted_sd_pct`: corresponding land-area-weighted
   standard deviation, representing within-city built-form heterogeneity.
5. `log_distance_to_water_weighted_median`: city land-area-weighted median of
   log(1 + distance in metres) to ESA WorldCover mapped water.
6. `log_terrain_relief_weighted_median`: city land-area-weighted median of
   log(1 + 1-km terrain relief in metres) from Copernicus DEM GLO-30.

Absolute elevation, municipal compactness and geographic coordinates are not
predictors. Compactness is excluded because coastlines, water exclusions,
holes and administrative annexation can dominate it. Coordinates are used only
to retrieve climate and display geography.

## Missingness and preprocessing

- All 50 cities must have all six predictors. If an external source fails for a
  city, the analysis stops; no outcome-informed imputation is allowed.
- Within each training split, predictors are centred and scaled using training
  means and standard deviations. Any zero-variance predictor in a training
  split stops the analysis.
- No transformations beyond the frozen definitions above are selected after
  outcome inspection.

## Frozen models

The primary model is ridge regression with all six predictors. Two diagnostic
models are fixed in advance: climate-only (predictors 1-2) and
urban-environment-only (predictors 3-6). They cannot replace the full model as
the primary model after results are observed.

### Nested leave-one-city-out prediction

- Each city is held out once as the outer test set.
- Within the remaining 49 cities, ridge alpha is selected using five fixed,
  shuffled inner folds and the grid 10^-3 to 10^3 in 25 log-spaced steps.
- The one-standard-error rule selects the largest alpha whose inner mean
  squared error is within one standard error of the minimum.
- Scaling is fitted separately within every inner and outer training split.
- The null prediction for each held-out city is the mean outcome of its 49
  training cities.

The complete procedure is repeated without changing predictors at 5% and 20%.

### Regional extrapolation

The full model is also evaluated by holding out each US Census region
(Northeast, Midwest, South and West) in turn. Alpha selection and scaling use
only cities outside the held-out region. This is a harder geographic
extrapolation audit, not a claim about all US cities.

## Metrics

For each model and budget, report:

- test RMSE and MAE;
- RMSE and MAE relative to the training-mean null;
- out-of-sample R2, defined as 1 - SSE_model / SSE_null;
- Spearman correlation between held-out observations and predictions;
- city-level held-out predictions and residuals.

No in-sample R2 or coefficient P value can support the primary claim.
Standardised coefficients are reported only as descriptive orientation and
must not be labelled mechanisms or drivers.

## Uncertainty and null tests

1. **Outcome uncertainty:** 1,000 draws select one of the existing paired
   spatial-block bootstrap outcomes independently within each city and rerun
   the complete nested leave-one-city-out full model. Report percentile
   intervals for RMSE ratio and out-of-sample R2.
2. **Permutation test:** 999 random permutations of the primary 10% outcome
   across cities rerun the complete nested procedure. The one-sided P value is
   `(1 + number of permuted RMSE advantages at least as large as observed) /
   1,000`.
3. **Influence audit:** recompute the mean squared-error advantage after
   deleting each held-out prediction from the scoring set. A result dominated
   by one city fails the gate.
4. **Regularisation sensitivity:** alpha grids shifted one decade lower and
   higher are descriptive sensitivities; they cannot rescue a failed primary
   model.

## Prespecified main-text gate

A cross-city predictive statement is eligible for the main text only if all
conditions hold:

1. at the 10% budget, the full-model nested-LOOCV RMSE is at least 10% lower
   than the training-mean null (`RMSE ratio <= 0.90`);
2. the complete-procedure permutation P value is below 0.05;
3. the full-model RMSE ratio is below 1.00 at both 5% and 20% budgets;
4. the pooled leave-one-Census-region-out RMSE ratio is below 1.00;
5. the 95% outcome-uncertainty interval for the 10% RMSE ratio lies wholly
   below 1.00; and
6. the leave-one-city scoring audit retains a positive mean squared-error
   advantage for every omitted city.

If any condition fails, the confirmatory conclusion is that these six generic
city conditions do not reliably transfer the allocation-divergence outcome
across the sampled cities. Domain-model or coefficient patterns may be shown
only as exploratory diagnostics and cannot be used to claim explanation.

## Permitted language

Permitted terms are `predict`, `out-of-city prediction`, `transferability`,
`associated city condition`, `predictive limitation` and `regional holdout`.
The module must not use `cause`, `drive`, `mechanism`, `determinant`, `explains
why`, `percentage explained`, `optimal policy`, `realised cooling` or
`implementation effect`.

This is an internal frozen protocol and not a public preregistration.
