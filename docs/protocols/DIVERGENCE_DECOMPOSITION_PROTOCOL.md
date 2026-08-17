# Step 2 frozen protocol: structural decomposition of allocation divergence

**Status:** frozen before the Step 2 confirmatory run  
**Date:** 12 August 2026  
**Baseline:** Step 1 city trade-off outcomes  
**Scope:** structural counterfactual decomposition, not causal attribution

## Question

Does the divergence between surface-heat-context and social-context allocation
arise primarily from unequal analytical capacity among canopy-deficit
neighbourhoods, from restricting allocation to neighbourhoods with a positive
canopy deficit, or from the remaining difference in the spatial ordering of the
two objectives?

The analysis does not estimate why social and thermal conditions came to occupy
their observed locations. It tests whether two features of the analytical
allocation design alter the already frozen divergence outcomes.

## Frozen analysis frame

- 34,566 block groups in 50 US municipal geographies.
- Positive represented population and positive city-clipped land area.
- Density-peer q75 canopy deficit, positive population-relative LST anomaly and
  the prespecified three-domain social-condition ordering are reconstructed as
  in the Step 1 confirmatory allocation analysis.
- Race and ethnicity are not allocation inputs.

## Frozen objectives

For block group i in city c, the two marginal objective scores are

    s_Hic = 100 P_ic H_ic / L_ic

and

    s_Sic = 100 P_ic V_ic / L_ic.

The surface-heat-context rule allocates in descending s_H, and the
social-context rule allocates in descending s_S. These are mathematical upper
bounds for their respective linear objectives under each counterfactual
capacity structure. Their superiority on their own objective is definitional.
If multiple units share the score at which the budget is exhausted, the
remaining budget is allocated in proportion to their available capacity. This
neutral tie convention avoids GEOID-dependent allocation. For each
city-structure-budget cell we record the fraction of the surface-heat
allocation assigned where H = 0. When positive-H capacity is below the budget,
the heat upper bound is spatially non-unique; overlap and the social quantity
retained by that heat allocation are treated as convention-dependent
sensitivities rather than unique optima.

## Three prespecified capacity structures

Let A_obs,ic = L_ic G_ic / 100 and A_c = sum_i A_obs,ic.

1. `observed_gap_capacity`

   A_ic = A_obs,ic. Only G > 0 units have positive capacity. Both candidate
   membership and capacity intensity retain the observed density-peer canopy
   deficit.

2. `neutral_capacity_same_candidates`

   Candidate membership remains G > 0, but every eligible unit receives the
   same capacity per unit land area:

       A_ic = L_ic A_c / sum_{j:G_jc>0} L_jc  if G_ic > 0,
              0 otherwise.

   Total city capacity remains A_c. Comparing this structure with
   `observed_gap_capacity` tests sensitivity to unequal gap intensity among the
   same candidates.

3. `neutral_capacity_all_units`

   Every analysed unit receives the same capacity per unit land area:

       A_ic = L_ic A_c / sum_j L_jc.

   Total city capacity remains A_c. Comparing this structure with
   `neutral_capacity_same_candidates` tests sensitivity to the G > 0 candidate
   gate after capacity intensity has been neutralised. This is a null
   comparator, not a proposal to allocate canopy to neighbourhoods without a
   measured deficit.

### Candidate-gate isolation sensitivity

The total-matched all-unit structure changes the constant capacity rate because
the denominator expands from eligible land to all analysed land. We therefore
run a shadow specification, `neutral_capacity_all_units_rate_matched`, in which
all units receive the same capacity per unit land as
`neutral_capacity_same_candidates`, while the budget remains anchored to 5%,
10% or 20% of A_c. Its total notional capacity is larger than A_c. This
specification holds the capacity constraint on the original candidates fixed
and changes only candidate availability. A candidate-gate interpretation is
considered robust only when the total-matched and rate-matched contrasts have
the same direction across all three budgets. The shadow specification is a
sensitivity analysis and not a fourth primary structure. A candidate-gate
result is eligible for main-text interpretation only when both contrasts
independently satisfy the three main-text gate criteria below and do so in the
same direction.

These sequential contrasts are path-dependent and must not be added or
described as unique percentages of divergence explained.

## Budgets

Each structure is evaluated at 5%, 10% and 20% of A_c. Within every city,
structure and budget, both rules receive and exhaust exactly the same area
budget. Total capacity and nominal budget are identical across the three
structures.

## Frozen outcomes

The three Step 1 outcomes remain unchanged:

1. continuous Jaccard overlap O_c;
2. heat-context opportunity loss under social allocation L_Hc;
3. social-context opportunity loss under heat allocation L_Sc.

The primary controlled contrasts are

    D_capacity = neutral_capacity_same_candidates - observed_gap_capacity

and

    D_gate = neutral_capacity_all_units - neutral_capacity_same_candidates.

A positive overlap contrast means that removing the tested structure increases
alignment. A positive opportunity-loss contrast means that removing the tested
structure increases the corresponding loss.

## Uncertainty

- 1,000 paired spatial-block bootstrap replicates per city.
- In each replicate, density strata and q75 canopy deficits are recomputed,
  the population-weighted city LST reference is recomputed, and the
  social-condition composite and its missing-value medians are reconstructed.
- All three structures and all three budgets are evaluated on the same
  resampled units, so controlled contrasts are paired.
- National estimates use equal-city means and 5,000 two-level draws that
  resample cities and select one spatial replicate within each sampled city.
- Percentile 95% intervals are reported.

## Prespecified main-text gate

A controlled contrast is eligible for main-text interpretation only if, for at
least one of the three frozen outcomes:

1. its deterministic point estimate has the same non-zero direction at 5%, 10%
   and 20% budgets;
2. its paired 95% two-level interval excludes zero in that same direction at
   all three budgets; and
3. the absolute deterministic change is at least 0.02 at all three budgets.

The 0.02 threshold is an interpretive materiality rule on the 0-1 outcome
scale, not a significance threshold. Failure to pass this gate is itself a
valid result: the tested structural feature is not a stable explanation of
allocation divergence over the prespecified budget range.

The heat objective remains uniquely valued when a budget extends into H = 0
units, because every allocation among those units adds zero to the heat
quantity. Its geography is nevertheless non-unique. Consequently,
`allocation_overlap` and
`social_context_opportunity_loss_under_heat` cannot pass the main-text gate
for a contrast if either compared structure allocates any deterministic
budget at H = 0 in any city at any of the three budgets. Those outcomes remain
reported as convention-dependent sensitivities. The
`heat_context_opportunity_loss_under_social` outcome remains eligible because its heat
upper bound is invariant to how the H = 0 tie is resolved. This exclusion rule
is frozen before inspecting the controlled-contrast results.

## Forbidden interpretations

This module must not use `cause`, `drive`, `mechanism`, `percentage explained`,
`realised cooling`, `health benefit`, `plantable land`, `optimal policy` or
`implementation plan` for its outputs. Permitted terms include `controlled
structural contrast`, `allocation geometry`, `candidate gate`, `capacity
intensity`, `structural sensitivity` and `model-defined opportunity loss`.
