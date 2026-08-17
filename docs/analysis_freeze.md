# Analysis freeze

Software release 0.3.0 freezes the submitted analysis as of 17 August 2026.
The immutable analytical frames and their intended roles
are listed in docs/analysis_frames.md. Key row counts, allocation trade-offs,
structural contrasts and the transferability decision gate are recorded in
manifests/analysis_targets.csv.

Analytical products use role-based directory names. The atlas layer identifier
`GHV27-2026-08-09` records the scientific content and freeze date, whereas
`0.3.0` identifies the software release. Derived allocation outputs are written
to dedicated `step01_city_tradeoff_outcomes`,
`step02_divergence_decomposition`, `step03_cross_city_transferability` and
`main_figure` directories.

The target ledger is a compact consistency aid, not a substitute for archived
source tables. Values with deliberately limited precision, such as population
reported in millions, must not be treated as bitwise test targets. The future
data release will add machine-readable panel-level values and tolerances.
