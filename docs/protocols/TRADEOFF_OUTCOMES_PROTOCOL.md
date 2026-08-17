# Step 1 frozen city trade-off outcomes

## Status

This protocol freezes the three city-level outcomes before any explanatory or
predictive modelling. The frozen NS directory is read-only input and is not
overwritten by this module.

## Spatial frame and allocation experiment

- Analysis frame: 34,566 block groups in 50 cities.
- Eligibility: positive density-peer q75 canopy deficit, G > 0.
- Budget: 10% of each city's analytical canopy-deficit capacity.
- Constraint: both rules receive and exhaust the same city-specific area budget.
- Surface-heat rule: maximises the population-weighted positive LST-context quantity.
- Social-context rule: maximises the population-weighted prespecified social-condition quantity.
- National estimand: equal-city mean; every city contributes total weight 1/50.

Analytical capacity is not parcel-level plantable land. The objectives do not
estimate realised cooling, air-temperature exposure, health benefit or welfare.

## Frozen outcomes

For city c, let x_H and x_S be the surface-heat and social-context upper-bound
allocations under the common budget B.

1. Continuous allocated-area overlap

   O_c = sum_i min(x_Hic, x_Sic) / sum_i max(x_Hic, x_Sic).

   This is a continuous Jaccard measure. Because both rules receive the same
   budget, the fraction of either budget placed in common is 2O_c/(1+O_c), and
   the redirected fraction is 1-2O_c/(1+O_c). These are algebraic translations,
   not additional outcomes.

2. Surface-heat-context opportunity loss under social allocation

   L_Hc = 1 - Q_Hc(x_S) / Q_Hc(x_H).

3. Social-context opportunity loss under heat allocation

   L_Sc = 1 - Q_Sc(x_H) / Q_Sc(x_S).

The directional difference L_Hc - L_Sc is retained only as a diagnostic. It
does not make the two objectives normatively equivalent and does not identify a
preferred allocation rule.

## Uncertainty

Each city has 1,000 spatial-block bootstrap replicates. G, H, V, analytical
capacity and the 10% city budget are recomputed within each replicate. National intervals use
5,000 two-level draws that resample cities and select a spatial-block replicate
within each sampled city. Intervals are percentile intervals.

## Primary headline values

- Allocation overlap: 0.355
  (0.322-0.390).
- Equivalent shared fraction of each equal-area budget:
  0.523
  (0.487-
  0.561); the complementary
  redirected fraction is 0.477.
- Heat-context opportunity loss under social allocation:
  0.288
  (0.255-
  0.331).
- Social-context opportunity loss under heat allocation:
  0.229
  (0.199-
  0.262).

These values describe model-defined opportunity losses. They are not realised
intervention effects.
