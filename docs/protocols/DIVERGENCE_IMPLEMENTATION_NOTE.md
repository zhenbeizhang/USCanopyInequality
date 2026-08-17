# Divergence decomposition implementation safeguards

This note records two safeguards enforced by the released Step 2 implementation.
It supplements the frozen analytical definitions in
`DIVERGENCE_DECOMPOSITION_PROTOCOL.md`; it is not a claim of public
preregistration.

## Outcome-specific spatial uniqueness

The spatial-non-uniqueness restriction does not apply to
`heat_context_opportunity_loss_under_social`. Allocation among units with
H = 0 cannot alter the surface-heat-context upper bound used in that outcome.
The implementation therefore applies the restriction by the complete,
registered outcome key rather than by a shortened alias.

## Candidate-gate reporting rule

`passes_individual_gate` records whether one candidate-gate specification
meets its own direction, magnitude and interval criteria.
`passes_main_text_gate` is stricter: it is true only when the total-matched and
rate-matched candidate-gate specifications both pass and agree in direction.
Neither individual specification is sufficient for a headline statement.

The released audit verifies these safeguards across all 50 city analyses,
1,000 city spatial-block replicates and 5,000 two-level draws.
