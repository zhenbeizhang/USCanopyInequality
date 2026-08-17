# Result 2 matched transferability audit: frozen analysis contract

## Scientific question

Do the headline social-canopy gradients show less cross-city heterogeneity than
the canopy-LST associations when both are evaluated with matched city-specific
slopes, REML partial pooling, physical spatial-block resampling and the same
transferability diagnostics?

## Primary estimand

For each social attribute, the primary estimand is the city-specific
population-weighted percentage-point difference in neighbourhood canopy per one
unweighted within-city standard deviation of that attribute. It is an ecological
gradient, not an individual effect or causal coefficient.

## Frozen variables

1. Hispanic/Latino share.
2. Renter-occupied household share.
3. Household share without a vehicle.
4. Median age, retained as an observed opposite-direction comparison.

No variable will be added or removed after inspecting city heterogeneity.

## Frozen spatial supports

- Primary block-group support: the 34,566 Result 3 support-aligned block groups,
  which are an exact subset of the 34,575 Result 2 fully contained frame.
- The four-variable common block-group frame is fixed before fitting.
- Secondary tract support: the Result 2 fully contained tract frame; the same
  four-variable common-case rule is applied.
- Predictors are restandardised separately within city and geography using the
  unweighted population standard deviation with `ddof=0`.

## Frozen model sequence

1. Fit a separate population-weighted WLS slope in each city, with an intercept
   and HC3 standard error.
2. Pool the 50 city slopes by the same REML random-effects implementation used
   in Result 3.
3. Report raw and partially pooled city slopes, the national REML mean, between-
   city tau, I2 and the random-effects prediction interval.
4. Retain city-equal and city-population-weighted means as alternative estimands.

The two-stage city-specific WLS plus REML design is chosen because it preserves
the Result 2 population weights and matches the Result 3 heterogeneity analysis.
An unweighted one-stage mixed model is not substituted for the weighted
estimand.

## Frozen spatial inference

- Use the Result 3 M2/M3 shared physical block widths and the same four grid
  origins.
- Use 500 replicates at the primary block width and 250 at half and double
  widths.
- Within every replicate, resample occupied blocks within each city, refit all
  city slopes and repeat REML pooling.
- Add a second city-resampling level for national intervals.
- Derive tract coordinates from the land-area-weighted centroids of their
  constituent support-aligned block groups. Use the same physical block widths;
  only shrink a width if fewer than eight tract blocks would be occupied.

## Transferability diagnostics

No single arbitrary threshold defines transferability. Each construct reports:

- national mean and 95% interval;
- between-city tau and tau divided by the absolute national mean;
- random-effects prediction interval;
- share of raw city point estimates in the national direction;
- number of main-scale city intervals entirely in that direction;
- number of cities whose intervals remain in that direction at all three block
  widths;
- rank and sign agreement between block-group and tract city slopes;
- rank stability across the three block widths.

The same metrics are extracted for Result 3 M2 city and M3.

## Claim decision rule

The manuscript may not state that social inequality is universally more
transferable than thermal priority. It may state that a named social gradient is
more cross-city consistent only when the matched diagnostics support that
gradient-specific comparison. If the social gradients are themselves strongly
heterogeneous, the conclusion must become: social and thermal relations both
vary among cities, but their heterogeneity profiles differ by construct.

## Mandatory validation checks

1. The block-group primary sample must be a strict common support with Result 3.
2. The four predictors must use identical rows within each geography.
3. Population weights, within-city scaling and canopy outcome must be identical
   across the four social models.
4. Every city must contribute at both spatial supports.
5. The central WLS and REML results must be reproduced by an independent script.
6. No city or variable may be omitted because it weakens the headline.

