from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULT = Path(str(analysis_store_root().joinpath('Result')))
OUT = (
    RESULT
    / "Supplementary figure and table"
    / "Supplementary Tables"
    / "versions"
    / "current_release"
)
R3 = RESULT / "Result3" / "versions"
R4 = RESULT / "Result4" / "versions"
R5 = RESULT / "Result5" / "versions"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"{name}.csv")


def main() -> None:
    checks: dict[str, object] = {}

    t5a = read("Table5a")
    checks["table5_has_four_prespecified_models"] = len(t5a) == 4
    checks["table5_roles_are_distinct"] = (
        t5a.loc[t5a["Model"].str.startswith("M2 city"), "Prespecified role"]
        .str.contains("Primary")
        .all()
        and t5a.loc[t5a["Model"].str.startswith("M3"), "Prespecified role"]
        .str.contains("Spatial-sensitivity")
        .all()
    )

    t5g = read("Table5g")
    rcs4 = t5g[t5g["Canopy functional form"].str.contains("4 knots")].iloc[0]
    checks["nonlinearity_p_matches"] = np.isclose(
        float(rcs4["City-clustered nonlinearity P"]), 0.092, atol=5e-4
    )
    checks["nonlinearity_delta_rmse_matches"] = np.isclose(
        float(rcs4["ΔRMSE versus linear (°C)"]), -0.0014, atol=5e-5
    )

    t6d = read("Table6d")
    checks["table6d_has_two_events_four_benchmarks"] = (
        len(t6d) == 8
        and t6d["Joint event"].nunique() == 2
        and t6d["Expected-prevalence benchmark"].nunique() == 4
    )
    checks["table6d_all_intervals_exceed_one"] = all(
        float(value.split("[")[1].split(",")[0]) > 1
        for value in t6d["Observed / expected [95% interval]"]
    )

    t6f = read("Table6f")
    checks["threshold_grid_complete"] = (
        len(t6f) == 12
        and t6f["Positive G and H quantile"].nunique() == 4
        and t6f["V quantile"].nunique() == 3
    )
    checks["threshold_loo_intervals_all_exceed_one"] = all(
        float(value.split("[")[1].split(",")[0]) > 1
        for value in t6f[
            "Primary-block leave-one-unit-out ratio [95% interval]"
        ]
    )

    t7b = read("Table7b")
    source = pd.read_csv(
        R5
        / "step2_multicriteria_pareto"
        / "tables"
        / "04_two_level_rule_intervals.csv"
    )
    primary = source[
        (source["rule"] == "pareto_0.50")
        & (source["metric"] == "heat_context_efficiency")
    ].iloc[0]
    printed = t7b.loc[
        t7b["Rule"] == "Equal-weight rule (λ=0.50)",
        "Surface-heat objective fraction [95% interval]",
    ].iloc[0]
    checks["table7_uses_deterministic_point"] = np.isclose(
        float(printed.split()[0]), primary["equal_city_central"], atol=5e-4
    )
    checks["table7_does_not_use_bootstrap_median_as_point"] = not np.isclose(
        float(printed.split()[0]), primary["two_level_median"], atol=5e-4
    )

    t7f = read("Table7f")
    difference = t7f[
        t7f["Diagnostic"].str.startswith("Overlap difference")
    ].iloc[0]
    checks["weighting_difference_interval_crosses_zero"] = (
        float(difference["95% two-level interval"].split(" to ")[0]) < 0
        < float(difference["95% two-level interval"].split(" to ")[1])
    )
    expected_tradeoff = {
        "Shared budget fraction": 0.5235,
        "Redirected budget fraction": 0.4765,
        "Heat-context opportunity loss": 0.2881,
        "Social-context opportunity loss": 0.2291,
    }
    for label, expected in expected_tradeoff.items():
        row = t7f[t7f["Diagnostic"].str.startswith(label)].iloc[0]
        checks[f"table7f_{label.lower().replace(' ', '_')}_matches"] = np.isclose(
            float(row["Deterministic equal-city point"]), expected, atol=5e-4
        )

    t7g = read("Table7g")
    checks["table7g_has_three_structures_and_three_budgets"] = (
        len(t7g) == 9
        and t7g["Controlled change"].str.contains("5% budget").sum() == 3
        and t7g["Controlled change"].str.contains("10% budget").sum() == 3
        and t7g["Controlled change"].str.contains("20% budget").sum() == 3
    )
    checks["table7g_intervals_are_positive"] = all(
        float(value.split(" to ")[0]) > 0
        for value in t7g["95% two-level interval"]
    )

    t8 = read("Table8")
    checks["table8_has_all_50_cities"] = len(t8) == 50 and t8["City"].nunique() == 50
    checks["all_city_budgets_equal_ten_percent_capacity"] = np.allclose(
        pd.to_numeric(t8["10% budget (km²)"]),
        0.1 * pd.to_numeric(t8["Analytical capacity (km²)"]),
        atol=0.011,
    )

    concatenated = "\n".join(
        path.read_text(encoding="utf-8-sig") for path in OUT.glob("Table*.csv")
    )
    forbidden = [
        "add_public_transit",
        "add_income_and_transit",
        "complete_four_indicator",
        "leave_one_city_out_threshold_transfer",
        "nan%",
        "population efficiency",
        "independent objective",
    ]
    checks["forbidden_raw_labels_absent"] = not any(term in concatenated for term in forbidden)

    checks = {key: bool(value) for key, value in checks.items()}
    failures = [key for key, value in checks.items() if not value]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "technical_interpretation": {
            "result3": "Linear canopy term retained because spline improvement is negligible and the city-clustered nonlinearity test is inconclusive.",
            "result4": "Co-location exceeds leave-one-unit-out marginal expectations at all prespecified spatial scales, but weaker ratios at smaller blocks remain descriptive rather than causal.",
            "result5": "Endpoint rules redirect 47.7% of equal-area allocation and incur 22.9-28.8% cross-objective losses; shared canopy-deficit constraints partly attenuate this divergence.",
        },
    }
    (OUT / "table_source_independent_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
