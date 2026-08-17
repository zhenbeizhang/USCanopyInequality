from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


WORK = Path(__file__).resolve().parent
BASE = Path(
    str(analysis_store_root().joinpath('Result', 'Result2', 'Result2_city_slope_transferability_audit'))
)

SOCIAL_PREDICTORS = ("hispanic", "renter", "no_vehicle", "median_age")
NEGATIVE_PREDICTORS = {"hispanic", "renter", "no_vehicle"}


def interval_has_expected_direction(
    predictor: str, low: float, high: float
) -> bool:
    if predictor in NEGATIVE_PREDICTORS:
        return high < 0
    return low > 0


def interval_crosses_zero(low: float, high: float) -> bool:
    return low < 0 < high


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    pre = json.loads(
        (BASE / "qa/pre_independent_audit.json").read_text(encoding="utf-8")
    )
    independent = json.loads(
        (BASE / "qa/independent_implementation_check.json").read_text(
            encoding="utf-8"
        )
    )
    summary = pd.read_csv(BASE / "tables/05_social_transferability_summary.csv")
    national_bootstrap = pd.read_csv(
        BASE / "tables/03_national_social_spatial_bootstrap.csv"
    )
    cross_support = pd.read_csv(
        BASE / "tables/08_block_group_tract_city_slope_agreement.csv"
    )
    social_block = pd.read_csv(
        BASE / "tables/07_social_block_scale_stability.csv"
    )
    thermal = pd.read_csv(
        BASE / "tables/09_result3_matched_transferability_metrics.csv"
    )
    thermal_block = pd.read_csv(
        BASE / "tables/10_result3_block_scale_stability.csv"
    )

    main_bootstrap = national_bootstrap.loc[
        national_bootstrap["block_scale"].eq("main")
    ].copy()

    national_social_direction = all(
        interval_has_expected_direction(
            row.predictor,
            float(row.main_two_level_reml_ci_low),
            float(row.main_two_level_reml_ci_high),
        )
        for row in summary.itertuples(index=False)
    )
    alternative_pooling_direction = all(
        interval_has_expected_direction(
            row.predictor,
            float(row.two_level_city_equal_ci_low),
            float(row.two_level_city_equal_ci_high),
        )
        and interval_has_expected_direction(
            row.predictor,
            float(row.two_level_city_population_weighted_ci_low),
            float(row.two_level_city_population_weighted_ci_high),
        )
        for row in main_bootstrap.itertuples(index=False)
    )
    social_prediction_intervals_all_cross_zero = all(
        interval_crosses_zero(
            float(row.prediction_interval_low),
            float(row.prediction_interval_high),
        )
        for row in summary.itertuples(index=False)
    )
    thermal_prediction_intervals_all_cross_zero = all(
        interval_crosses_zero(
            float(row.prediction_interval_low),
            float(row.prediction_interval_high),
        )
        for row in thermal.itertuples(index=False)
    )

    primary_cross_support = cross_support.loc[
        cross_support["slope_type"].eq("main_spatial_bootstrap_median")
    ].copy()
    minimum_bg_tract_rho = float(primary_cross_support["spearman_rho"].min())
    minimum_bg_tract_sign_agreement = float(
        primary_cross_support["city_sign_agreement"].min()
    )
    minimum_social_block_rho = float(social_block["spearman_rho"].min())
    minimum_thermal_block_rho = float(thermal_block["spearman_rho"].min())

    bg = summary.loc[summary["geography"].eq("block_group")].copy()
    social_all_scale_counts = {
        row.predictor: int(row.n_all_scale_intervals_in_national_direction)
        for row in bg.itertuples(index=False)
    }
    thermal_all_scale_counts = {
        row.construct: int(row.n_all_scale_intervals_in_national_direction)
        for row in thermal.itertuples(index=False)
    }
    three_disadvantage_social_counts = np.array(
        [
            social_all_scale_counts["hispanic"],
            social_all_scale_counts["renter"],
            social_all_scale_counts["no_vehicle"],
        ],
        dtype=int,
    )
    thermal_counts = np.array(list(thermal_all_scale_counts.values()), dtype=int)
    universal_social_superiority_supported = bool(
        three_disadvantage_social_counts.min() > thermal_counts.max()
    )

    checks = {
        "all_preanalysis_checks_pass": all(pre["checks"].values()),
        "independent_implementation_passes": bool(independent["pass"]),
        "all_50_cities_represented_at_both_scales": (
            summary.groupby("geography")["n_cities"].min().eq(50).all()
            and summary.groupby("geography")["n_cities"].max().eq(50).all()
        ),
        "common_case_sample_counts_match_contract": (
            set(
                summary.loc[
                    summary["geography"].eq("block_group"), "n_units"
                ].astype(int)
            )
            == {34434}
            and set(
                summary.loc[
                    summary["geography"].eq("tract"), "n_units"
                ].astype(int)
            )
            == {11932}
        ),
        "national_social_intervals_have_expected_direction": (
            national_social_direction
        ),
        "equal_city_and_population_weighted_summaries_have_expected_direction": (
            alternative_pooling_direction
        ),
        "all_social_city_prediction_intervals_cross_zero": (
            social_prediction_intervals_all_cross_zero
        ),
        "all_thermal_city_prediction_intervals_cross_zero": (
            thermal_prediction_intervals_all_cross_zero
        ),
        "minimum_bg_tract_city_slope_spearman_at_least_0_90": (
            minimum_bg_tract_rho >= 0.90
        ),
        "minimum_bg_tract_city_sign_agreement_at_least_0_95": (
            minimum_bg_tract_sign_agreement >= 0.95
        ),
        "minimum_social_block_scale_spearman_at_least_0_98": (
            minimum_social_block_rho >= 0.98
        ),
        "minimum_thermal_block_scale_spearman_at_least_0_98": (
            minimum_thermal_block_rho >= 0.98
        ),
        "universal_social_superiority_is_not_asserted": (
            not universal_social_superiority_supported
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise AssertionError(f"Final audit failed: {failed}")

    claim_rows = [
        {
            "claim": (
                "The national mean social-canopy gradients retain their expected "
                "directions under spatial block bootstrap."
            ),
            "decision": "SUPPORTED",
            "evidence": (
                "All four block-group and tract two-level bootstrap intervals "
                "exclude zero in the expected direction."
            ),
            "permitted_wording": (
                "Across the 50-city sample, canopy exposure was lower in "
                "neighbourhoods with higher Hispanic/Latino, renter and "
                "no-vehicle shares, and higher in older neighbourhoods on average."
            ),
        },
        {
            "claim": (
                "The social gradients are robust to changing census support from "
                "block groups to tracts."
            ),
            "decision": "SUPPORTED_WITH_SCOPE",
            "evidence": (
                f"City-slope Spearman rho ranges from "
                f"{minimum_bg_tract_rho:.2f} to "
                f"{primary_cross_support['spearman_rho'].max():.2f}; sign "
                f"agreement is "
                f"{minimum_bg_tract_sign_agreement:.0%}-"
                f"{primary_cross_support['city_sign_agreement'].max():.0%}."
            ),
            "permitted_wording": (
                "City rankings and slope directions were highly concordant between "
                "block groups and tracts; this supports scale robustness, not "
                "individual-level inference."
            ),
        },
        {
            "claim": "The same social-canopy direction applies to every city.",
            "decision": "NOT_SUPPORTED",
            "evidence": (
                "Every random-effects city prediction interval crosses zero; only "
                "25-36 of 50 tract intervals and 26-35 of 50 block-group intervals "
                "retain the national direction at all three block widths."
            ),
            "permitted_wording": (
                "The national gradients coexist with substantial between-city "
                "heterogeneity and should not be treated as universal city rules."
            ),
        },
        {
            "claim": (
                "Social-canopy gradients are generally more transferable across "
                "cities than canopy-LST associations."
            ),
            "decision": "NOT_SUPPORTED",
            "evidence": (
                "All-block-width same-direction counts are 26, 32 and 28 for the "
                "three disadvantage indicators, compared with 33 for M2 and 30 "
                "for M3 canopy-LST associations; all prediction intervals cross zero."
            ),
            "permitted_wording": (
                "Both social-canopy and canopy-LST associations have stable national "
                "average directions but material city heterogeneity."
            ),
        },
    ]
    claim_table = pd.DataFrame(claim_rows)
    claim_table.to_csv(
        BASE / "tables/12_claim_level_transferability_decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final = {
        "audit_status": "PASS_WITH_INTERPRETIVE_BOUNDARY",
        "checks": checks,
        "key_diagnostics": {
            "minimum_bg_tract_city_slope_spearman": minimum_bg_tract_rho,
            "minimum_bg_tract_city_sign_agreement": (
                minimum_bg_tract_sign_agreement
            ),
            "minimum_social_block_scale_spearman": minimum_social_block_rho,
            "minimum_thermal_block_scale_spearman": minimum_thermal_block_rho,
            "block_group_all_scale_same_direction_counts": (
                social_all_scale_counts
            ),
            "thermal_all_scale_same_direction_counts": thermal_all_scale_counts,
            "all_social_prediction_intervals_cross_zero": (
                social_prediction_intervals_all_cross_zero
            ),
            "all_thermal_prediction_intervals_cross_zero": (
                thermal_prediction_intervals_all_cross_zero
            ),
        },
        "claim_decision": {
            "national_social_gradient": "SUPPORTED",
            "block_group_to_tract_concordance": "SUPPORTED_WITH_SCOPE",
            "universal_city_direction": "NOT_SUPPORTED",
            "social_more_transferable_than_thermal": "NOT_SUPPORTED",
        },
        "reporting_boundary": (
            "National average directions coexist with substantial city "
            "heterogeneity; neither universal city direction nor comparative "
            "transferability is supported."
        ),
        "remaining_boundary": (
            "The thermal comparison is matched at the block-group support and block "
            "bootstrap design. A tract-scale thermal refit would be required before "
            "claiming fully symmetric cross-support transferability for Result 3."
        ),
    }
    (BASE / "qa/final_audit.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )

    table_lines = []
    for row in bg.itertuples(index=False):
        tract_row = summary.loc[
            summary["geography"].eq("tract")
            & summary["predictor"].eq(row.predictor)
        ].iloc[0]
        agreement = primary_cross_support.loc[
            primary_cross_support["predictor"].eq(row.predictor)
        ].iloc[0]
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.predictor_label),
                    (
                        f"{fmt(row.main_two_level_reml_median)} "
                        f"[{fmt(row.main_two_level_reml_ci_low)}, "
                        f"{fmt(row.main_two_level_reml_ci_high)}]"
                    ),
                    f"{int(row.n_all_scale_intervals_in_national_direction)}/50",
                    (
                        f"{fmt(row.prediction_interval_low)} to "
                        f"{fmt(row.prediction_interval_high)}"
                    ),
                    (
                        f"{fmt(tract_row.main_two_level_reml_median)} "
                        f"[{fmt(tract_row.main_two_level_reml_ci_low)}, "
                        f"{fmt(tract_row.main_two_level_reml_ci_high)}]"
                    ),
                    fmt(float(agreement["spearman_rho"])),
                ]
            )
            + " |"
        )

    english_report = f"""# Result 2 city heterogeneity and transferability audit

## Frozen estimand and sample

The matched analysis uses 34,434 block groups and 11,932 tracts in all 50
cities. The four variables share one complete-case mask within each geography.
Each coefficient is the population-weighted ecological difference in canopy
percentage points per one unweighted within-city standard deviation of the
named social attribute. City-specific WLS-HC3 slopes are partially pooled by
REML. Spatial uncertainty uses 500 primary-width and 250 half- and double-width
block-bootstrap replicates, followed by city-level resampling.

## Results

| Social attribute | Block-group slope, 95% two-level interval | Cities retaining direction at all widths | City prediction interval | Tract slope, 95% interval | BG-tract rho |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

All four national average directions remain after changing the pooling summary
from REML to equal-city or city-population weighting. City slope rankings are
also highly concordant across block groups and tracts (minimum Spearman rho
{minimum_bg_tract_rho:.2f}). However, all random-effects city prediction
intervals cross zero, demonstrating that a national average direction is not a
universal city-level rule.

## Symmetric comparison with Result 3

The all-block-width same-direction counts are 26, 32 and 28 cities for
Hispanic/Latino share, renter share and no-vehicle share, respectively. The
corresponding counts are 33 cities for the M2 canopy-LST model and 30 for M3.
Relative between-city heterogeneity is also of similar magnitude. The proposed
claim that social gradients are generally more transferable than thermal
associations therefore fails the matched test.

## Technical decision

`PASS_WITH_INTERPRETIVE_BOUNDARY`. The computation and the national social
gradients are reproducible. The supported interpretation reports city
heterogeneity explicitly: both social-canopy and canopy-LST
associations have stable national average directions while varying materially
among cities. A tract-scale thermal refit remains necessary before making a
fully symmetric cross-support claim for Result 3.
"""
    (BASE / "text/Result2_transferability_audit_report_EN.md").write_text(
        english_report, encoding="utf-8"
    )

    scripts_dir = BASE / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    public_sources = {
        repo_root() / "docs" / "result2_transferability_contract.md": "analysis_contract.md",
        Path(__file__).with_name("run_cross_scale_transferability.py"): "run_cross_scale_transferability.py",
        Path(__file__).with_name("audit_cross_scale_transferability.py"): "audit_cross_scale_transferability.py",
        Path(__file__): "summarize_cross_scale_transferability.py",
    }
    for source, filename in public_sources.items():
        shutil.copy2(source, scripts_dir / filename)

    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
