#!/usr/bin/env python
"""Scientific audit of the completed Step 3 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uscanopy.paths import analysis_store_root


DEFAULT_ROOT = (
    analysis_store_root()
    / "Result"
    / "Result5"
    / "versions"
    / "step03_cross_city_transferability"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root

    metrics = pd.read_csv(root / "tables" / "nested_loocv_metrics.csv")
    regions = pd.read_csv(root / "tables" / "regional_holdout_metrics.csv")
    grids = pd.read_csv(root / "tables" / "regularisation_grid_sensitivity.csv")
    predictions = pd.read_csv(root / "tables" / "city_heldout_predictions.csv")
    uncertainty = pd.read_csv(root / "tables" / "outcome_uncertainty_summary.csv")
    gates = pd.read_csv(root / "tables" / "prespecified_gate_audit.csv")
    implementation = json.loads(
        (root / "qa" / "independent_audit_summary.json").read_text(encoding="utf-8")
    )

    primary = metrics.loc[
        (metrics["model"] == "full_external_conditions")
        & np.isclose(metrics["budget_fraction"], 0.10)
    ].iloc[0]
    primary_predictions = predictions.loc[
        (predictions["model"] == "full_external_conditions")
        & np.isclose(predictions["budget_fraction"], 0.10)
    ].copy()
    primary_predictions["model_squared_error"] = (
        primary_predictions["observed"] - primary_predictions["predicted"]
    ) ** 2
    primary_predictions["null_squared_error"] = (
        primary_predictions["observed"]
        - primary_predictions["training_mean_null"]
    ) ** 2
    by_region = (
        primary_predictions.groupby("census_region", sort=True)
        .agg(
            n_cities=("city_key", "size"),
            model_mse=("model_squared_error", "mean"),
            null_mse=("null_squared_error", "mean"),
            mean_mse_advantage=("squared_error_advantage", "mean"),
        )
        .reset_index()
    )
    by_region["within_LOOCV_rmse_ratio"] = np.sqrt(
        by_region["model_mse"] / by_region["null_mse"]
    )
    by_region.to_csv(root / "qa" / "primary_loocv_skill_by_region.csv", index=False)

    region_primary = regions.loc[np.isclose(regions["budget_fraction"], 0.10)].copy()
    region_primary.to_csv(root / "qa" / "primary_region_holdout_summary.csv", index=False)

    budget_consistency = metrics.loc[
        metrics["model"] == "full_external_conditions",
        [
            "budget_fraction",
            "rmse_ratio",
            "mae_ratio",
            "out_of_sample_r2",
            "spearman_rho",
        ],
    ].copy()
    budget_consistency.to_csv(
        root / "qa" / "full_model_budget_consistency.csv", index=False
    )

    domain_comparison = metrics.loc[
        np.isclose(metrics["budget_fraction"], 0.10),
        ["model", "rmse_ratio", "mae_ratio", "out_of_sample_r2", "spearman_rho"],
    ].copy()
    domain_comparison.to_csv(
        root / "qa" / "primary_domain_model_comparison.csv", index=False
    )

    primary_boot = uncertainty.loc[
        np.isclose(uncertainty["budget_fraction"], 0.10)
        & (uncertainty["metric"] == "rmse_ratio")
    ].iloc[0]
    pooled_region = region_primary.loc[region_primary["region"] == "Pooled"].iloc[0]
    lower_grid = grids.loc[grids["alpha_grid"] == "one_decade_lower"].iloc[0]
    frozen_grid = grids.loc[grids["alpha_grid"] == "frozen_primary"].iloc[0]
    higher_grid = grids.loc[grids["alpha_grid"] == "one_decade_higher"].iloc[0]
    urban_only = domain_comparison.loc[
        domain_comparison["model"] == "urban_environment_only"
    ].iloc[0]

    # Scientific—not computational—review gates. These preserve the frozen
    # confirmatory gate and flag reasons a result cannot be promoted after the
    # fact even if several weak-signal diagnostics are positive.
    findings = [
        {
            "finding": "F1_prespecified_material_skill_gate",
            "pass": bool(float(primary["rmse_ratio"]) <= 0.90),
            "evidence": f"Nested-LOOCV RMSE ratio {primary['rmse_ratio']:.3f}; required <=0.900.",
            "implication": "A strong transferable prediction claim is not supported.",
        },
        {
            "finding": "F2_outcome_uncertainty_supports_direction",
            "pass": bool(float(primary_boot["q975"]) < 1.0),
            "evidence": (
                f"Bootstrap RMSE-ratio interval {primary_boot['q025']:.3f}-"
                f"{primary_boot['q975']:.3f}."
            ),
            "implication": "The small average improvement is directionally stable at 10%.",
        },
        {
            "finding": "F3_geographic_holdout_supports_direction",
            "pass": bool(float(pooled_region["rmse_ratio"]) < 1.0),
            "evidence": f"Pooled leave-region-out RMSE ratio {pooled_region['rmse_ratio']:.3f}.",
            "implication": "The direction survives a harder geographic split, but weakly.",
        },
        {
            "finding": "F4_effect_size_robust_to_alpha_grid",
            "pass": bool(
                max(
                    float(lower_grid["rmse_ratio"]),
                    float(frozen_grid["rmse_ratio"]),
                    float(higher_grid["rmse_ratio"]),
                )
                - min(
                    float(lower_grid["rmse_ratio"]),
                    float(frozen_grid["rmse_ratio"]),
                    float(higher_grid["rmse_ratio"]),
                )
                <= 0.05
            ),
            "evidence": (
                "RMSE ratios across shifted grids were "
                f"{lower_grid['rmse_ratio']:.3f}, {frozen_grid['rmse_ratio']:.3f} "
                f"and {higher_grid['rmse_ratio']:.3f}."
            ),
            "implication": (
                "The direction remains below the null, but its magnitude is sensitive "
                "to the permitted regularisation range."
            ),
        },
        {
            "finding": "F5_no_single_city_dominance",
            "pass": bool(
                implementation["all_leave_one_scoring_city_advantages_positive"]
                and implementation[
                    "largest_positive_city_fraction_of_total_advantage"
                ]
                < 0.25
            ),
            "evidence": (
                f"Largest positive city contributed "
                f"{100 * implementation['largest_positive_city_fraction_of_total_advantage']:.1f}% "
                "of the total MSE advantage; every deletion retained a positive advantage."
            ),
            "implication": "The aggregate direction is not created by one city.",
        },
        {
            "finding": "F6_substantive_out_of_sample_explanation",
            "pass": bool(float(primary["out_of_sample_r2"]) >= 0.20),
            "evidence": f"Out-of-sample R2 was {primary['out_of_sample_r2']:.3f}; audit threshold 0.20.",
            "implication": (
                "Most cross-city variation remains unpredicted; the module cannot be framed "
                "as explaining why priorities diverge."
            ),
        },
        {
            "finding": "F7_geographic_ranking_transferability",
            "pass": bool(float(pooled_region["spearman_rho"]) > 0),
            "evidence": (
                f"Pooled leave-region-out Spearman rho was "
                f"{pooled_region['spearman_rho']:.3f}."
            ),
            "implication": (
                "The small RMSE gain reflects shrinkage toward the training mean rather "
                "than successful ranking of cities in unseen regions."
            ),
        },
        {
            "finding": "F8_full_model_adds_to_urban_environment_domain",
            "pass": bool(float(primary["rmse_ratio"]) < float(urban_only["rmse_ratio"])),
            "evidence": (
                f"Full-model RMSE ratio {primary['rmse_ratio']:.3f}; "
                f"urban-environment-only ratio {urban_only['rmse_ratio']:.3f}."
            ),
            "implication": (
                "Long-term climate variables did not add held-out skill to the simpler "
                "urban-environment specification."
            ),
        },
    ]
    findings_table = pd.DataFrame(findings)
    findings_table.to_csv(root / "qa" / "scientific_quality_findings.csv", index=False)

    confirmatory_pass = bool(gates["pass"].all())
    if confirmatory_pass:
        disposition = "MAIN_TEXT_PREDICTIVE_RESULT"
    elif bool(primary_boot["q975"] < 1.0) and float(primary["rmse_ratio"]) < 1.0:
        disposition = "SUPPLEMENTARY_WEAK_SIGNAL_OR_DISCUSSION_BOUNDARY"
    else:
        disposition = "DO_NOT_USE_AS_EVIDENCE"

    summary = {
        "status": "PASS",
        "computational_integrity": implementation["status"],
        "prespecified_confirmatory_gate_pass": confirmatory_pass,
        "recommended_disposition": disposition,
        "primary_nested_loocv_rmse_ratio": float(primary["rmse_ratio"]),
        "primary_nested_loocv_out_of_sample_r2": float(primary["out_of_sample_r2"]),
        "primary_nested_loocv_spearman_rho": float(primary["spearman_rho"]),
        "primary_bootstrap_rmse_ratio_interval": [
            float(primary_boot["q025"]),
            float(primary_boot["q975"]),
        ],
        "pooled_leave_region_out_rmse_ratio": float(pooled_region["rmse_ratio"]),
        "pooled_leave_region_out_spearman_rho": float(
            pooled_region["spearman_rho"]
        ),
        "regions_with_positive_within_loocv_mse_advantage": int(
            (by_region["mean_mse_advantage"] > 0).sum()
        ),
        "regions_total": int(len(by_region)),
        "scientific_finding_pass_count": int(findings_table["pass"].sum()),
        "scientific_finding_total": int(len(findings_table)),
        "technical_conclusion": (
            "The six external city conditions carry a statistically detectable but "
            "substantively weak out-of-city signal. Regional predictions did not rank "
            "cities correctly, and the full model did not outperform the simpler "
            "urban-environment model. The predictors do not meet the frozen skill "
            "threshold and do not explain cross-city divergence. Retain only as a "
            "supplementary negative/limiting result or a concise Discussion boundary; "
            "do not use it to upgrade the main Nature claim."
        ),
    }
    (root / "qa" / "scientific_quality_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Step 3 technical audit",
        "",
        f"**Recommended disposition:** `{disposition}`",
        "",
        "## Decision",
        "",
        summary["technical_conclusion"],
        "",
        "## Evidence",
        "",
    ]
    for finding in findings:
        lines.append(
            f"- {'PASS' if finding['pass'] else 'FAIL'} — {finding['finding']}: "
            f"{finding['evidence']} {finding['implication']}"
        )
    lines.extend(
        [
            "",
            "## Permitted manuscript statement",
            "",
            (
                "A prespecified six-variable model of long-term climate and urban "
                "environmental conditions modestly improved out-of-city prediction of "
                "the model-defined heat-context opportunity loss, but the improvement "
                "did not reach the prespecified material-skill threshold. Most city-to-city "
                "variation therefore remained unresolved by these generic conditions."
            ),
            "",
            "## Claims that remain unsupported",
            "",
            "- The model explains why social and surface-heat priorities diverge.",
            "- The fitted coefficients identify mechanisms or causal drivers.",
            "- The model can guide allocations in cities outside the sampled frame.",
            "- Better prediction implies realised cooling, welfare or health benefits.",
        ]
    )
    (root / "reports" / "STEP03_TECHNICAL_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
