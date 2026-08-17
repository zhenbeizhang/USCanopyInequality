from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


OLD = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'step2_multicriteria_pareto')))
AUDIT = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'weighting_heterogeneity_audit')))
TRADEOFF = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'step01_city_tradeoff_outcomes')))
STRUCTURAL = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'step02_divergence_decomposition')))
FIG = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'main_figure')))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    city = pd.read_csv(OLD / "tables" / "01_city_rule_outcomes.csv")
    intervals = pd.read_csv(OLD / "tables" / "04_two_level_rule_intervals.csv")
    groups_city = pd.read_csv(OLD / "tables" / "02_city_group_distribution_ratios.csv")
    groups = pd.read_csv(OLD / "tables" / "05_two_level_group_intervals.csv")
    sensitivity = pd.read_csv(AUDIT / "tables" / "result5_weighting_two_level_intervals.csv")
    sensitivity_qa = json.loads(
        (AUDIT / "qa" / "result5_weighting_heterogeneity_audit.json").read_text(encoding="utf-8")
    )
    headline = pd.read_csv(
        TRADEOFF / "tables" / "national_primary_tradeoff_outcomes.csv"
    ).set_index("metric")
    structural = pd.read_csv(
        STRUCTURAL / "tables" / "structural_finding_evidence_ledger.csv"
    )

    require(city["city_key"].nunique() == 50, "Result 5 does not contain 50 cities")
    require(len(city) == 50 * 28, "Unexpected city-by-rule table size")
    require(np.allclose(city["budget_fraction"], 0.10), "Primary budget is not uniformly 10%")
    budget_error = float(np.max(np.abs(city["allocated_m2"] - city["budget_m2"])))
    require(budget_error < 1e-5, f"Allocation budget mismatch: {budget_error}")

    metric_columns = {
        "heat_context_efficiency": "heat_context_efficiency",
        "social_condition_efficiency": "social_condition_efficiency",
        "population_efficiency": "population_efficiency",
    }
    central_error = 0.0
    for _, row in intervals.iterrows():
        values = city.loc[city["rule"].eq(row["rule"]), metric_columns[row["metric"]]]
        require(len(values) == 50, f"Missing cities for {row['rule']}, {row['metric']}")
        central_error = max(central_error, abs(float(values.mean()) - float(row["equal_city_central"])))
    require(central_error < 1e-12, f"Figure point estimates do not reproduce city means: {central_error}")

    group_error = 0.0
    for _, row in groups.iterrows():
        values = groups_city.loc[
            groups_city["rule"].eq(row["rule"]) & groups_city["group"].eq(row["group"]),
            "allocation_share_to_population_share",
        ]
        require(len(values) == 50, f"Missing group-city rows for {row['rule']}, {row['group']}")
        group_error = max(group_error, abs(float(values.mean()) - float(row["equal_city_central"])))
    require(group_error < 1e-12, f"Group ratio points do not reproduce city means: {group_error}")

    diff = sensitivity.loc[
        sensitivity["metric"].eq("overlap_difference_equal_minus_population")
    ].iloc[0]
    require(sensitivity_qa["status"] == "PASS", "Weighting audit did not pass")
    require(sensitivity_qa["cities"] == 50, "Weighting audit is not a 50-city analysis")
    require(sensitivity_qa["city_spatial_block_replicates"] == 500, "Wrong city bootstrap count")
    require(sensitivity_qa["two_level_outer_draws"] == 5000, "Wrong outer bootstrap count")
    require(float(diff["two_level_ci_low"]) < 0 < float(diff["two_level_ci_high"]), "Weighting difference interval should span zero")

    expected_headlines = {
        "allocation_overlap": 0.3545374061181079,
        "shared_budget_fraction": 0.5234811597180717,
        "redirected_budget_fraction": 0.47651884028192826,
        "heat_context_opportunity_loss_under_social": 0.2881230116651707,
        "social_context_opportunity_loss_under_heat": 0.22905911497378814,
    }
    for metric, expected in expected_headlines.items():
        observed = float(headline.loc[metric, "equal_city_deterministic_point"])
        require(
            np.isclose(observed, expected, atol=5e-4),
            f"Frozen headline mismatch for {metric}: {observed}",
        )
    require(len(structural) == 9, "Structural evidence ledger must contain nine controlled contrasts")
    require(
        set(structural["outcome"]) == {"heat_context_opportunity_loss_under_social"},
        "Structural ledger contains an unintended outcome",
    )

    png = FIG / "figures" / "Fig5.png"
    with Image.open(png) as image:
        width, height = image.size
        require(width >= 4200 and height >= 3000, "Raster export is below expected submission resolution")
        rgb = np.asarray(image.convert("RGB"))
    require(np.any(rgb < 245), "Rendered figure is blank")
    for suffix in (".pdf", ".svg", ".tif"):
        path = png.with_suffix(suffix)
        require(path.exists() and path.stat().st_size > 10_000, f"Missing or empty {suffix} export")

    caption = (FIG / "text" / "Fig5_caption.txt").read_text(encoding="utf-8")
    required_phrases = (
        "diagnostic allocation",
        "illustrative rather than a recommended optimum",
        "not parcel-level planting plans",
        "47.7% redirected area",
        "28.8% for the surface-heat objective",
        "22.9% for the social objective",
        "or estimates of realised cooling",
    )
    for phrase in required_phrases:
        require(phrase in caption, f"Caption lacks boundary statement: {phrase}")

    report = {
        "status": "PASS",
        "quality_gate_status": "PASS",
        "cities": 50,
        "analysis_units": 34566,
        "budget_fraction": 0.10,
        "max_budget_error_m2": budget_error,
        "central_point_reproduction_error": central_error,
        "group_point_reproduction_error": group_error,
        "weighting_sensitivity_difference": float(diff["equal_city_deterministic_point"]),
        "weighting_sensitivity_95_interval": [
            float(diff["two_level_ci_low"]),
            float(diff["two_level_ci_high"]),
        ],
        "headline_results": {
            metric: float(headline.loc[metric, "equal_city_deterministic_point"])
            for metric in expected_headlines
        },
        "structural_contrast_rows": int(len(structural)),
        "figure_pixels": [width, height],
        "technical_judgement": (
            "The figure supports a diagnostic trade-off claim. It does not establish an optimal policy, "
            "realized cooling, parcel feasibility or causal intervention effects."
        ),
    }
    out = FIG / "qa" / "Fig5_independent_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
