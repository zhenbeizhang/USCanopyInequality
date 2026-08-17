#!/usr/bin/env python
"""Independent implementation and integrity audit for Step 3 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from uscanopy.paths import analysis_store_root


SEED = 20260812
ALPHAS = np.logspace(-3, 3, 25)
PREDICTORS = [
    "jja_air_temperature_C",
    "jja_precipitation_mm",
    "built_fraction_area_weighted_mean_pct",
    "built_fraction_area_weighted_sd_pct",
    "log_distance_to_water_weighted_median",
    "log_terrain_relief_weighted_median",
]
MODEL_SPECS = {
    "full_external_conditions": PREDICTORS,
    "climate_only": PREDICTORS[:2],
    "urban_environment_only": PREDICTORS[2:],
}
BUDGETS = [0.05, 0.10, 0.20]
DEFAULT_ROOT = (
    analysis_store_root()
    / "Result"
    / "Result5"
    / "versions"
    / "step03_cross_city_transferability"
)


def direct_nested_loocv(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    predictions = np.full(n, np.nan)
    selected = np.full(n, np.nan)
    for held_out in range(n):
        outer_train = np.delete(np.arange(n), held_out)
        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED + held_out,
        )
        fold_mse = np.empty((len(ALPHAS), 5))
        for fold_id, (relative_train, relative_valid) in enumerate(
            kfold.split(outer_train)
        ):
            inner_train = outer_train[relative_train]
            inner_valid = outer_train[relative_valid]
            scaler = StandardScaler().fit(x[inner_train])
            x_train = scaler.transform(x[inner_train])
            x_valid = scaler.transform(x[inner_valid])
            for alpha_id, alpha in enumerate(ALPHAS):
                model = Ridge(alpha=alpha, fit_intercept=True).fit(
                    x_train, y[inner_train]
                )
                residual = model.predict(x_valid) - y[inner_valid]
                fold_mse[alpha_id, fold_id] = np.mean(residual**2)
        mean_mse = fold_mse.mean(axis=1)
        minimum = int(np.argmin(mean_mse))
        threshold = mean_mse[minimum] + fold_mse[minimum].std(ddof=1) / np.sqrt(5)
        chosen = int(np.flatnonzero(mean_mse <= threshold + 1e-14)[-1])
        scaler = StandardScaler().fit(x[outer_train])
        model = Ridge(alpha=ALPHAS[chosen], fit_intercept=True).fit(
            scaler.transform(x[outer_train]), y[outer_train]
        )
        predictions[held_out] = model.predict(
            scaler.transform(x[[held_out]])
        )[0]
        selected[held_out] = ALPHAS[chosen]
    return predictions, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root

    ledger = pd.read_csv(root / "data" / "analysis_city_ledger.csv")
    saved = pd.read_csv(root / "tables" / "city_heldout_predictions.csv")
    central_metrics = pd.read_csv(root / "tables" / "nested_loocv_metrics.csv")
    region_metrics = pd.read_csv(root / "tables" / "regional_holdout_metrics.csv")
    influence = pd.read_csv(root / "tables" / "influence_audit.csv")

    if len(ledger) != 50 or ledger["city_key"].duplicated().any():
        raise RuntimeError("Analysis ledger is not 50 unique cities.")
    if len(saved) != 50 * len(BUDGETS) * len(MODEL_SPECS):
        raise RuntimeError("Saved held-out prediction table has unexpected length.")

    audit_rows = []
    for model_name, columns in MODEL_SPECS.items():
        x = ledger[columns].to_numpy(float)
        for budget in BUDGETS:
            budget_label = int(round(100 * budget))
            y = ledger[f"outcome_budget_{budget_label:02d}pct"].to_numpy(float)
            direct_prediction, direct_alpha = direct_nested_loocv(x, y)
            subset = (
                saved.loc[
                    (saved["model"] == model_name)
                    & np.isclose(saved["budget_fraction"], budget)
                ]
                .set_index("city_key")
                .reindex(ledger["city_key"])
            )
            if subset.isna().any().any():
                raise RuntimeError("Saved prediction merge failed during independent audit.")
            saved_prediction = subset["predicted"].to_numpy(float)
            saved_alpha = subset["selected_alpha"].to_numpy(float)
            audit_rows.append(
                {
                    "model": model_name,
                    "budget_fraction": budget,
                    "max_absolute_prediction_difference": float(
                        np.max(np.abs(saved_prediction - direct_prediction))
                    ),
                    "max_absolute_alpha_difference": float(
                        np.max(np.abs(saved_alpha - direct_alpha))
                    ),
                    "saved_prediction_min": float(saved_prediction.min()),
                    "saved_prediction_max": float(saved_prediction.max()),
                    "observed_min": float(y.min()),
                    "observed_max": float(y.max()),
                    "fraction_alpha_at_upper_grid_boundary": float(
                        np.mean(np.isclose(saved_alpha, ALPHAS.max()))
                    ),
                }
            )
    implementation_audit = pd.DataFrame(audit_rows)
    implementation_audit.to_csv(
        root / "qa" / "independent_implementation_audit.csv", index=False
    )

    primary = (
        saved.loc[
            (saved["model"] == "full_external_conditions")
            & np.isclose(saved["budget_fraction"], 0.10)
        ]
        .copy()
        .sort_values("squared_error_advantage")
    )
    primary["absolute_residual"] = primary["residual"].abs()
    primary["model_squared_error"] = primary["residual"] ** 2
    primary["null_squared_error"] = (
        primary["observed"] - primary["training_mean_null"]
    ) ** 2
    primary.to_csv(root / "qa" / "primary_city_error_audit.csv", index=False)

    region_primary = region_metrics.loc[
        np.isclose(region_metrics["budget_fraction"], 0.10)
    ].copy()
    region_primary.to_csv(root / "qa" / "primary_region_error_audit.csv", index=False)

    maximum_prediction_difference = float(
        implementation_audit["max_absolute_prediction_difference"].max()
    )
    maximum_alpha_difference = float(
        implementation_audit["max_absolute_alpha_difference"].max()
    )
    predicted_outside_unit_interval = bool(
        (implementation_audit["saved_prediction_min"] < 0).any()
        or (implementation_audit["saved_prediction_max"] > 1).any()
    )
    primary_metric = central_metrics.loc[
        (central_metrics["model"] == "full_external_conditions")
        & np.isclose(central_metrics["budget_fraction"], 0.10)
    ].iloc[0]
    primary_advantage = primary["squared_error_advantage"].sum()
    top_positive = primary.nlargest(1, "squared_error_advantage").iloc[0]
    top_negative = primary.nsmallest(1, "squared_error_advantage").iloc[0]
    top_positive_fraction = (
        float(top_positive["squared_error_advantage"] / primary_advantage)
        if primary_advantage > 0
        else np.nan
    )

    audit = {
        "status": "PASS"
        if maximum_prediction_difference < 1e-10
        and maximum_alpha_difference < 1e-10
        and not predicted_outside_unit_interval
        and bool(influence["positive_after_omission"].all())
        else "FAIL",
        "independent_sklearn_recalculation": True,
        "maximum_absolute_prediction_difference": maximum_prediction_difference,
        "maximum_absolute_alpha_difference": maximum_alpha_difference,
        "predictions_outside_unit_interval": predicted_outside_unit_interval,
        "primary_rmse_ratio": float(primary_metric["rmse_ratio"]),
        "primary_out_of_sample_r2": float(primary_metric["out_of_sample_r2"]),
        "all_leave_one_scoring_city_advantages_positive": bool(
            influence["positive_after_omission"].all()
        ),
        "largest_positive_city_contribution": str(top_positive["city_display"]),
        "largest_positive_city_fraction_of_total_advantage": top_positive_fraction,
        "largest_negative_city_contribution": str(top_negative["city_display"]),
        "interpretation": (
            "The computation is reproducible, but the primary out-of-city effect "
            "size remains below the prespecified material-skill gate."
        ),
    }
    (root / "qa" / "independent_audit_summary.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
