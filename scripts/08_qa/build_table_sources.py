from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document


BASE = Path(str(manuscript_root()))
RESULT = Path(str(analysis_store_root().joinpath('Result')))
SOURCE_DOC = BASE / "supplementary_information.docx"
OUT = (
    RESULT
    / "Supplementary figure and table"
    / "Supplementary Tables"
    / "versions"
    / "current_release"
)
OUT.mkdir(parents=True, exist_ok=True)

R3 = RESULT / "Result3" / "versions"
R4 = RESULT / "Result4" / "versions"
R5 = RESULT / "Result5" / "versions"


def clean(value: object) -> str:
    return " ".join(str(value).split())


def extract_table(index: int) -> pd.DataFrame:
    doc = Document(SOURCE_DOC)
    table = doc.tables[index]
    rows = [[clean(cell.text) for cell in row.cells] for row in table.rows]
    return pd.DataFrame(rows[1:], columns=rows[0])


def write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def fmt_interval(point: float, low: float, high: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def build_table5a() -> pd.DataFrame:
    old = extract_table(12)
    model = pd.read_csv(
        R3
        / "main_figure"
        / "source_data"
        / "Fig3b_model_estimates.csv"
    ).set_index("model_id")
    diag = pd.read_csv(
        R3
        / "main_figure"
        / "source_data"
        / "Fig3c_model_diagnostics.csv"
    ).set_index("model_id")
    roles = pd.read_csv(
        R3
        / "nonlinearity_role_audit"
        / "tables"
        / "result3_m2_m3_complementary_roles.csv"
    ).set_index("model_id")

    ids = [
        "M1_common_FE_density",
        "M2_common_FE_environment",
        "M2_city_specific_REML",
        "M3_spatial_GAM_REML",
    ]
    labels = [
        "M1: city FE + density",
        "M2 common: city FE + environment",
        "M2 city: city slopes + REML",
        "M3: spatial GAM + REML",
    ]
    held_moran = dict(zip(labels, pd.to_numeric(old["Held-block Moran's I"])))
    held_r2 = dict(zip(labels, pd.to_numeric(old["Held-block R²"])))
    rows = []
    for model_id, label in zip(ids, labels):
        row = model.loc[model_id]
        point = float(row["coefficient_C_per_10pp_canopy"])
        analytic = fmt_interval(point, float(row["ci_low"]), float(row["ci_high"]), 2)
        if model_id in roles.index:
            boot = fmt_interval(
                point,
                float(row["plot_ci_low"]),
                float(row["plot_ci_high"]),
                2,
            )
            tau = f"{float(row['tau_between_city']):.3f}"
            pred = (
                f"{float(row['prediction_interval_low']):.2f} to "
                f"{float(row['prediction_interval_high']):.2f}"
            )
            role = str(roles.loc[model_id, "inferential_role"])
        else:
            boot, tau, pred = "—", "—", "—"
            role = (
                "Common-slope adjustment stage"
                if model_id == "M2_common_FE_environment"
                else "Baseline adjustment stage"
            )
        rows.append(
            {
                "Model": label,
                "Deterministic slope [95% analytic interval] (°C per 10 pp)": analytic,
                "Deterministic slope [95% spatial-bootstrap interval]": boot,
                "Held-block RMSE (°C)": f"{float(diag.loc[model_id, 'weighted_rmse_C']):.3f}",
                "Held-block R²": f"{held_r2[label]:.3f}",
                "In-sample pooled Moran's I": f"{float(diag.loc[model_id, 'pooled_block_diagonal_moran_i_8nn']):.3f}",
                "Held-block Moran's I": f"{held_moran[label]:.3f}",
                "Between-city τ": tau,
                "City-slope prediction interval": pred,
                "Prespecified role": role,
            }
        )
    return pd.DataFrame(rows)


def build_table5g() -> pd.DataFrame:
    cv = pd.read_csv(
        R3
        / "nonlinearity_role_audit"
        / "tables"
        / "result3_nonlinearity_cv_summary.csv"
    )
    full = pd.read_csv(
        R3
        / "nonlinearity_role_audit"
        / "tables"
        / "result3_nonlinearity_full_sample.csv"
    )
    merged = cv.merge(
        full[["model_id", "knots_canopy_pct", "nonlinearity_city_cluster_p"]],
        on="model_id",
        how="left",
    )
    labels = {
        "linear": "Linear canopy term",
        "rcs3": "Restricted cubic spline, 3 knots",
        "rcs4": "Restricted cubic spline, 4 knots",
        "rcs5": "Restricted cubic spline, 5 knots",
    }
    rows = []
    for row in merged.itertuples(index=False):
        p = getattr(row, "nonlinearity_city_cluster_p")
        rows.append(
            {
                "Canopy functional form": labels[row.model_id],
                "Canopy knots (%)": "—" if row.model_id == "linear" else row.knots_canopy_pct,
                "Spatial-CV RMSE (°C)": f"{row.weighted_rmse_C:.4f}",
                "ΔRMSE versus linear (°C)": f"{row.rmse_change_vs_linear_C:+.4f}",
                "City-clustered nonlinearity P": "—" if pd.isna(p) else f"{p:.3f}",
            }
        )
    return pd.DataFrame(rows)


def build_table6d() -> pd.DataFrame:
    source = pd.read_csv(
        R4
        / "leave_one_unit_out_benchmark"
        / "tables"
        / "result4_leave_one_unit_out_benchmarks.csv"
    )
    source = source[
        (source["grid_origin"].isin(["not_applicable", "origin_mean"]))
        & source["event"].isin(["joint_context", "strict_G3H3V3"])
    ].copy()
    event_labels = {
        "joint_context": "Context A (G>0, H3, V3)",
        "strict_G3H3V3": "Strict G3H3V3",
    }
    benchmark_labels = {
        "city_marginal_leave_one_unit_out": "Within-city leave-one-unit-out marginals",
        "small_block_leave_one_unit_out": "Small spatial blocks, leave-one-unit-out",
        "main_block_leave_one_unit_out": "Primary spatial blocks, leave-one-unit-out",
        "large_block_leave_one_unit_out": "Large spatial blocks, leave-one-unit-out",
    }
    order = {
        "city_marginal_leave_one_unit_out": 0,
        "large_block_leave_one_unit_out": 1,
        "main_block_leave_one_unit_out": 2,
        "small_block_leave_one_unit_out": 3,
    }
    source["_order"] = source["benchmark"].map(order)
    source = source.sort_values(["event", "_order"])
    rows = []
    for row in source.itertuples(index=False):
        rows.append(
            {
                "Joint event": event_labels[row.event],
                "Expected-prevalence benchmark": benchmark_labels[row.benchmark],
                "Observed equal-city prevalence": f"{100 * row.observed_equal_city_prevalence_supported_units:.2f}%",
                "Expected equal-city prevalence": f"{100 * row.expected_equal_city_prevalence:.2f}%",
                "Observed / expected [95% interval]": fmt_interval(
                    row.observed_expected_ratio,
                    row.bootstrap_p025,
                    row.bootstrap_p975,
                    3,
                ),
                "Supported-unit fraction": f"{100 * row.equal_city_supported_unit_fraction:.1f}%",
            }
        )
    return pd.DataFrame(rows)


def build_table6e() -> pd.DataFrame:
    # The released table consolidates construct and boundary-support sensitivities into one
    # Supplementary table. Reading that table directly avoids silently
    # relabelling the preceding leave-one-unit-out benchmark table.
    combined = extract_table(23)
    expected = [
        "Specification",
        "Units",
        "Six-context agreement",
        "Exact-type agreement",
        "Context A Jaccard",
    ]
    missing = set(expected).difference(combined.columns)
    if missing:
        raise KeyError(f"Table 6e is missing columns: {sorted(missing)}")
    combined = combined[
        [
            "Specification",
            "Units",
            "Six-context agreement",
            "Exact-type agreement",
            "Context A Jaccard",
        ]
    ].copy()
    return combined.replace({"nan%": "—", "nan": "—", "": "—"})


def build_table6f() -> pd.DataFrame:
    source = pd.read_csv(
        R4
        / "threshold_leave_one_unit_out"
        / "tables"
        / "result4_threshold_leave_one_unit_out.csv"
    ).sort_values(["positive_G_H_quantile", "V_quantile"])
    rows = []
    for row in source.itertuples(index=False):
        rows.append(
            {
                "Positive G and H quantile": f"q{int(round(row.positive_G_H_quantile * 100))}",
                "V quantile": f"q{int(round(row.V_quantile * 100))}",
                "G threshold (pp)": f"{row.G_cut_pp:.2f}",
                "H threshold (°C)": f"{row.H_cut_C:.2f}",
                "V threshold": f"{row.V_cut:.3f}",
                "City leave-one-unit-out ratio [95% interval]": fmt_interval(
                    row.city_loo_ratio,
                    row.city_loo_ratio_bootstrap_ci_low,
                    row.city_loo_ratio_bootstrap_ci_high,
                    3,
                ),
                "Primary-block leave-one-unit-out ratio [95% interval]": fmt_interval(
                    row.main_block_loo_ratio,
                    row.main_block_loo_ratio_bootstrap_ci_low,
                    row.main_block_loo_ratio_bootstrap_ci_high,
                    3,
                ),
                "Supported units": f"{100 * row.main_block_supported_fraction:.1f}%",
            }
        )
    return pd.DataFrame(rows)


RULE_LABELS = {
    "uniform": "Uniform fraction",
    "canopy_deficit_first": "Canopy-deficit first",
    "surface_heat_only": "Surface-heat score",
    "social_condition_only": "Social-condition score",
    "population_oracle": "Resident-context upper bound",
    "heat_context_oracle": "Surface-heat upper bound",
    "social_condition_oracle": "Social-context upper bound",
    "pareto_0.50": "Equal-weight rule (λ=0.50)",
}

METRIC_LABELS = {
    "heat_context_efficiency": "Surface-heat objective fraction",
    "social_condition_efficiency": "Social-context objective fraction",
    "population_efficiency": "Resident-context objective fraction",
}


def result5_rule_intervals() -> pd.DataFrame:
    return pd.read_csv(
        R5
        / "step2_multicriteria_pareto"
        / "tables"
        / "04_two_level_rule_intervals.csv"
    )


def build_table7a() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Capacity A_i", "L_i G_i / 100", "m²", "Area that closes the analytical density-peer canopy deficit; not verified plantable land"],
            ["City budget B_c", "0.10 Σ_i A_ic", "m²", "Same city-specific area for every rule"],
            ["Canopy-context increment ΔC_i", "100 x_i / L_i", "canopy percentage points", "Definitional mapped-canopy increment"],
            ["Surface-heat quantity Q_H", "Σ_i P_i H_i ΔC_i", "person × °C × canopy pp", "Alignment with residents and positive surface-temperature context"],
            ["Social-context quantity Q_S", "Σ_i P_i V_i ΔC_i", "person × score × canopy pp", "Alignment with residents and the frozen social-condition ordering"],
            ["Resident-context quantity Q_P", "Σ_i P_i ΔC_i", "person × canopy pp", "Population-weighted canopy-context increment"],
            ["Objective fraction E", "Q(rule) / Q(objective-specific upper bound)", "0–1", "City-normalized fraction of a mathematical upper bound"],
            ["Allocation overlap", "Σ_i min(x_ai,x_bi) / Σ_i max(x_ai,x_bi)", "0–1", "Continuous area Jaccard overlap"],
            ["Weighted rule", "arg max [λE_H + (1−λ)E_S]", "0–1 λ", "Explicit location on the surface-heat–social-context frontier"],
        ],
        columns=["Quantity", "Definition", "Unit", "Interpretation"],
    )


def build_table7b() -> pd.DataFrame:
    source = result5_rule_intervals()
    keep = list(RULE_LABELS)
    source = source[source["rule"].isin(keep)]
    rows = []
    for rule in keep:
        subset = source[source["rule"] == rule].set_index("metric")
        row = {"Rule": RULE_LABELS[rule]}
        for metric, label in METRIC_LABELS.items():
            x = subset.loc[metric]
            row[f"{label} [95% interval]"] = fmt_interval(
                x["equal_city_central"],
                x["two_level_ci_low"],
                x["two_level_ci_high"],
                3,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_table7c() -> pd.DataFrame:
    source = result5_rule_intervals()
    source = source[source["rule"].str.startswith("pareto_")].copy()
    source["lambda"] = source["rule"].str.replace("pareto_", "", regex=False).astype(float)
    source = source[np.isclose((source["lambda"] * 10) % 1, 0.0)]
    rows = []
    for lam in sorted(source["lambda"].unique()):
        subset = source[np.isclose(source["lambda"], lam)].set_index("metric")
        row = {"Surface-heat weight λ": f"{lam:.2f}"}
        for metric, label in METRIC_LABELS.items():
            x = subset.loc[metric]
            row[f"{label} [95% interval]"] = fmt_interval(
                x["equal_city_central"],
                x["two_level_ci_low"],
                x["two_level_ci_high"],
                3,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_table7d() -> pd.DataFrame:
    source = pd.read_csv(
        R5
        / "step2_multicriteria_pareto"
        / "tables"
        / "05_two_level_group_intervals.csv"
    )
    keep = ["uniform", "heat_context_oracle", "social_condition_oracle", "pareto_0.50"]
    source = source[source["rule"].isin(keep)]
    group_labels = {
        "people_of_color_pct": "People of colour",
        "hispanic_or_latino_pct": "Hispanic/Latino",
        "black_alone_pct": "Black",
        "asian_alone_pct": "Asian",
        "white_alone_pct": "White alone",
        "age_65_plus_pct": "Age 65 years or older",
    }
    rows = []
    for rule in keep:
        for row in source[source["rule"] == rule].itertuples(index=False):
            point = row.equal_city_central
            low = row.two_level_ci_low
            high = row.two_level_ci_high
            if low > 1:
                relation = "Above parity"
            elif high < 1:
                relation = "Below parity"
            else:
                relation = "Includes parity"
            rows.append(
                {
                    "Rule": RULE_LABELS[rule],
                    "Population group": group_labels[row.group],
                    "Allocation-share / population-share ratio [95% interval]": fmt_interval(point, low, high, 2),
                    "Interval relative to 1": relation,
                }
            )
    return pd.DataFrame(rows)


def build_table7e() -> pd.DataFrame:
    old = extract_table(29).rename(
        columns={
            "Heat efficiency": "Surface-heat objective fraction",
            "Social efficiency": "Social-context objective fraction",
            "Population efficiency": "Resident-context objective fraction",
            "Median heat-social oracle overlap": "Median endpoint-allocation overlap",
        }
    )
    return old


def build_table7f() -> pd.DataFrame:
    source = pd.read_csv(
        R5
        / "weighting_heterogeneity_audit"
        / "tables"
        / "result5_weighting_two_level_intervals.csv"
    )
    labels = {
        "population_weighted_overlap": "Endpoint overlap, primary population weighting",
        "equal_neighbourhood_overlap": "Endpoint overlap, equal-neighbourhood weighting",
        "overlap_difference_equal_minus_population": "Overlap difference: equal-neighbourhood minus population weighting",
        "population_heat_retained_by_social": "Surface-heat fraction retained by social upper bound, population weighting",
        "population_social_retained_by_heat": "Social-context fraction retained by heat upper bound, population weighting",
        "equal_heat_retained_by_social": "Surface-heat fraction retained by social upper bound, equal-neighbourhood weighting",
        "equal_social_retained_by_heat": "Social-context fraction retained by heat upper bound, equal-neighbourhood weighting",
        "heat_selection_overlap_between_weightings": "Surface-heat allocation overlap between weighting schemes",
        "social_selection_overlap_between_weightings": "Social-context allocation overlap between weighting schemes",
    }
    rows = []
    for row in source.itertuples(index=False):
        rows.append(
            {
                "Diagnostic": labels[row.metric],
                "Deterministic equal-city point": f"{row.equal_city_deterministic_point:.3f}",
                "95% two-level interval": f"{row.two_level_ci_low:.3f} to {row.two_level_ci_high:.3f}",
            }
        )
    primary = pd.read_csv(
        R5
        / "step01_city_tradeoff_outcomes"
        / "tables"
        / "national_primary_tradeoff_outcomes.csv"
    ).set_index("metric")
    primary_labels = {
        "shared_budget_fraction": "Shared budget fraction (algebraic transform)",
        "redirected_budget_fraction": "Redirected budget fraction (algebraic transform)",
        "heat_context_opportunity_loss_under_social": "Heat-context opportunity loss under social allocation",
        "social_context_opportunity_loss_under_heat": "Social-context opportunity loss under heat allocation",
    }
    for metric, label in primary_labels.items():
        row = primary.loc[metric]
        rows.append(
            {
                "Diagnostic": label,
                "Deterministic equal-city point": f"{row.equal_city_deterministic_point:.4f}",
                "95% two-level interval": f"{row.two_level_ci_low:.4f} to {row.two_level_ci_high:.4f}",
            }
        )
    return pd.DataFrame(rows)


def build_table7g() -> pd.DataFrame:
    source = pd.read_csv(
        R5
        / "step02_divergence_decomposition"
        / "tables"
        / "structural_finding_evidence_ledger.csv"
    )
    source = source.loc[
        source["outcome"].eq("heat_context_opportunity_loss_under_social")
    ].copy()
    contrast_labels = {
        "capacity_intensity_neutral_minus_observed": "Neutral capacity, same candidates",
        "candidate_gate_total_matched_all_minus_same": "All units, total-matched capacity",
        "candidate_gate_rate_matched_all_minus_same": "All units, rate-matched capacity",
    }
    contrast_order = list(contrast_labels)
    source["contrast"] = pd.Categorical(
        source["contrast"], categories=contrast_order, ordered=True
    )
    source = source.sort_values(["contrast", "budget_fraction"])
    rows = []
    for row in source.itertuples(index=False):
        rows.append(
            {
                "Controlled change": (
                    f"{contrast_labels[str(row.contrast)]}; "
                    f"{100 * row.budget_fraction:.0f}% budget "
                    f"({int(row.cities_positive_audit)}/50 positive)"
                ),
                "Delta heat-context loss (percentage points)": (
                    f"{100 * row.equal_city_deterministic_difference:+.2f}"
                ),
                "95% two-level interval": (
                    f"{100 * row.ci_low:+.2f} to {100 * row.ci_high:+.2f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_table8() -> pd.DataFrame:
    source = pd.read_csv(
        R5
        / "weighting_heterogeneity_audit"
        / "tables"
        / "result5_weighting_city_diagnostics.csv"
    ).sort_values("city_display")
    return pd.DataFrame(
        {
            "City": source["city_display"],
            "Analytical capacity (km²)": source["capacity_km2"].map(lambda x: f"{x:.2f}"),
            "10% budget (km²)": source["budget_km2"].map(lambda x: f"{x:.2f}"),
            "Primary endpoint overlap": source["population_weighted_overlap"].map(lambda x: f"{x:.3f}"),
            "Equal-neighbourhood endpoint overlap": source["equal_neighbourhood_overlap"].map(lambda x: f"{x:.3f}"),
            "Heat fraction retained by social upper bound": source["population_heat_retained_by_social"].map(lambda x: f"{x:.3f}"),
            "Social fraction retained by heat upper bound": source["population_social_retained_by_heat"].map(lambda x: f"{x:.3f}"),
            "Capacity-weighted H–V rank correlation": source["capacity_weighted_heat_social_spearman"].map(lambda x: f"{x:.3f}"),
            "Capacity concentration (HHI)": source["capacity_hhi"].map(lambda x: f"{x:.3f}"),
        }
    )


def main() -> None:
    outputs = {
        "Table5a": build_table5a(),
        "Table5g": build_table5g(),
        "Table6d": build_table6d(),
        "Table6e": build_table6e(),
        "Table6f": build_table6f(),
        "Table7a": build_table7a(),
        "Table7b": build_table7b(),
        "Table7c": build_table7c(),
        "Table7d": build_table7d(),
        "Table7e": build_table7e(),
        "Table7f": build_table7f(),
        "Table7g": build_table7g(),
        "Table8": build_table8(),
    }
    for name, frame in outputs.items():
        write(frame, f"{name}.csv")

    checks = {
        "files": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in outputs.items()},
        "reporting_convention": "deterministic equal-city full-data points with two-level bootstrap 2.5th and 97.5th percentiles",
        "table5_primary_role": "M2 city for adjusted association and held-block prediction; M3 for spatial sensitivity and residual structure",
        "nonlinearity": {
            "rcs4_city_cluster_p": 0.09179642681212946,
            "rcs4_delta_cv_rmse_C": -0.001351807945033201,
        },
        "leave_one_unit_out": {
            "context_A_city_ratio": 1.426752,
            "context_A_primary_block_ratio": 1.141060,
        },
        "weighting_sensitivity": {
            "primary_overlap": 0.3545374061181079,
            "equal_neighbourhood_overlap": 0.34795201821154115,
            "difference_ci": [-0.034552528988569975, 0.02054760535247624],
        },
        "tradeoff": {
            "shared_budget_fraction": 0.5235,
            "redirected_budget_fraction": 0.4765,
            "heat_context_opportunity_loss_under_social": 0.2881,
            "social_context_opportunity_loss_under_heat": 0.2291,
        },
        "structural_contrasts": {
            "rows": len(outputs["Table7g"]),
            "budgets": [0.05, 0.10, 0.20],
        },
    }
    (OUT / "table_source_audit.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
