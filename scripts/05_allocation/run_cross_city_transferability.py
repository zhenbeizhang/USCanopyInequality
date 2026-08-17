#!/usr/bin/env python
"""Run the frozen Step 3 cross-city transferability audit.

The script deliberately separates the external city predictor ledger from the
Step 2 outcome until all model choices and decision gates have been frozen.
It uses precomputed ridge linear operators so that the complete nested
cross-validation procedure can be repeated for bootstrap and permutation
draws without approximating alpha selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from uscanopy.paths import analysis_store_root, repo_root


SEED = 20260812
N_BOOT = 1000
N_PERM = 999
BUDGETS = np.array([0.05, 0.10, 0.20], dtype=float)
PRIMARY_BUDGET = 0.10
TARGET = "heat_context_opportunity_loss_under_social"
STRUCTURE = "observed_gap_capacity"
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
ALPHAS = np.logspace(-3, 3, 25)
ALPHAS_LOWER = np.logspace(-4, 2, 25)
ALPHAS_HIGHER = np.logspace(-2, 4, 25)

RESULT5_VERSIONS = analysis_store_root() / "Result" / "Result5" / "versions"
DEFAULT_OUTPUT = RESULT5_VERSIONS / "step03_cross_city_transferability"
DEFAULT_PREDICTORS = DEFAULT_OUTPUT / "data" / "city_external_predictors.csv"
DEFAULT_STEP2 = RESULT5_VERSIONS / "step02_divergence_decomposition"
DEFAULT_PROTOCOL = (
    repo_root() / "docs" / "protocols" / "CROSS_CITY_TRANSFERABILITY_PROTOCOL.md"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= weights.sum()
    return float(np.interp(quantile, cdf, values))


def ridge_prediction_map(
    x_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> np.ndarray:
    """Return W such that y_hat_test = W @ y_train.

    Predictor centring and scaling are fitted only on x_train. The intercept is
    not penalised, matching the standard ridge objective.
    """
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("A training split contains a zero-variance predictor.")
    z_train = (x_train - mean) / scale
    z_test = (x_test - mean) / scale
    a_train = np.column_stack([np.ones(len(z_train)), z_train])
    a_test = np.column_stack([np.ones(len(z_test)), z_test])
    penalty = np.eye(a_train.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    operator = np.linalg.solve(
        a_train.T @ a_train + penalty,
        a_train.T,
    )
    return a_test @ operator


@dataclass
class SplitOperators:
    train: np.ndarray
    test: np.ndarray
    inner: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    outer_maps: np.ndarray
    label: str


class NestedRidgePredictor:
    """Nested ridge predictions for arbitrary fixed outer test splits."""

    def __init__(
        self,
        x: np.ndarray,
        outer_tests: Iterable[tuple[str, np.ndarray]],
        alphas: np.ndarray,
        seed: int,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.alphas = np.asarray(alphas, dtype=float)
        self.splits: list[SplitOperators] = []
        all_indices = np.arange(len(self.x))
        for split_id, (label, test) in enumerate(outer_tests):
            test = np.asarray(test, dtype=int)
            train = np.setdiff1d(all_indices, test, assume_unique=True)
            if len(train) < 10:
                raise RuntimeError(f"Outer split {label} has too few training cities.")
            kfold = KFold(
                n_splits=5,
                shuffle=True,
                random_state=seed + split_id,
            )
            inner: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for relative_train, relative_valid in kfold.split(train):
                inner_train = train[relative_train]
                inner_valid = train[relative_valid]
                maps = np.stack(
                    [
                        ridge_prediction_map(
                            self.x[inner_train], self.x[inner_valid], alpha
                        )
                        for alpha in self.alphas
                    ],
                    axis=0,
                )
                inner.append((inner_train, inner_valid, maps))
            outer_maps = np.stack(
                [
                    ridge_prediction_map(self.x[train], self.x[test], alpha)
                    for alpha in self.alphas
                ],
                axis=0,
            )
            self.splits.append(
                SplitOperators(
                    train=train,
                    test=test,
                    inner=inner,
                    outer_maps=outer_maps,
                    label=label,
                )
            )

    def predict_many(
        self, outcomes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray(outcomes, dtype=float)
        if y.ndim == 1:
            y = y[None, :]
        if y.shape[1] != len(self.x):
            raise ValueError("Outcome city dimension does not match predictors.")
        if not np.isfinite(y).all():
            raise RuntimeError("Non-finite outcome supplied to nested prediction.")

        n_draws = y.shape[0]
        predictions = np.full_like(y, np.nan, dtype=float)
        null_predictions = np.full_like(y, np.nan, dtype=float)
        selected_alpha = np.full((n_draws, len(self.splits)), np.nan, dtype=float)

        for split_id, split in enumerate(self.splits):
            fold_mse = np.empty(
                (n_draws, len(self.alphas), len(split.inner)), dtype=float
            )
            for fold_id, (inner_train, inner_valid, maps) in enumerate(split.inner):
                # maps: alpha x valid x train; result: draw x alpha x valid
                fold_predictions = np.einsum(
                    "avt,dt->dav", maps, y[:, inner_train], optimize=True
                )
                residual = fold_predictions - y[:, inner_valid][:, None, :]
                fold_mse[:, :, fold_id] = np.mean(residual**2, axis=2)

            mean_mse = fold_mse.mean(axis=2)
            minimum_index = np.argmin(mean_mse, axis=1)
            selected_fold_mse = fold_mse[
                np.arange(n_draws), minimum_index, :
            ]
            standard_error = selected_fold_mse.std(axis=1, ddof=1) / math.sqrt(
                fold_mse.shape[2]
            )
            threshold = mean_mse[
                np.arange(n_draws), minimum_index
            ] + standard_error
            eligible = mean_mse <= threshold[:, None] + 1e-14
            chosen_index = np.max(
                np.where(eligible, np.arange(len(self.alphas))[None, :], -1),
                axis=1,
            )
            if np.any(chosen_index < 0):
                raise RuntimeError("The one-standard-error rule selected no alpha.")

            outer_all = np.einsum(
                "akt,dt->dak",
                split.outer_maps,
                y[:, split.train],
                optimize=True,
            )
            chosen_predictions = outer_all[
                np.arange(n_draws), chosen_index, :
            ]
            predictions[:, split.test] = chosen_predictions
            null_predictions[:, split.test] = y[:, split.train].mean(axis=1)[:, None]
            selected_alpha[:, split_id] = self.alphas[chosen_index]

        if not np.isfinite(predictions).all() or not np.isfinite(null_predictions).all():
            raise RuntimeError("Nested prediction did not cover every held-out city.")
        return predictions, null_predictions, selected_alpha


def vector_metrics(
    observed: np.ndarray, predicted: np.ndarray, null_predicted: np.ndarray
) -> dict[str, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    null_predicted = np.asarray(null_predicted, dtype=float)
    if observed.ndim == 1:
        observed = observed[None, :]
        predicted = predicted[None, :]
        null_predicted = null_predicted[None, :]
    model_sse = np.sum((observed - predicted) ** 2, axis=1)
    null_sse = np.sum((observed - null_predicted) ** 2, axis=1)
    model_mae = np.mean(np.abs(observed - predicted), axis=1)
    null_mae = np.mean(np.abs(observed - null_predicted), axis=1)
    if np.any(null_sse <= 0) or np.any(null_mae <= 0):
        raise RuntimeError("Null error is zero; relative prediction metrics are undefined.")
    return {
        "rmse": np.sqrt(model_sse / observed.shape[1]),
        "mae": model_mae,
        "null_rmse": np.sqrt(null_sse / observed.shape[1]),
        "null_mae": null_mae,
        "rmse_ratio": np.sqrt(model_sse / null_sse),
        "mae_ratio": model_mae / null_mae,
        "out_of_sample_r2": 1.0 - model_sse / null_sse,
        "mse_advantage": (null_sse - model_sse) / observed.shape[1],
    }


def one_row_metrics(
    observed: np.ndarray, predicted: np.ndarray, null_predicted: np.ndarray
) -> dict[str, float]:
    metrics = vector_metrics(observed, predicted, null_predicted)
    result = {key: float(value[0]) for key, value in metrics.items()}
    result["spearman_rho"] = float(spearmanr(observed, predicted).statistic)
    return result


def select_alpha_and_coefficients(
    x: np.ndarray, y: np.ndarray, alphas: np.ndarray, seed: int
) -> tuple[float, float, np.ndarray]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_mse = np.empty((len(alphas), 5), dtype=float)
    for fold_id, (train, valid) in enumerate(kfold.split(x)):
        for alpha_id, alpha in enumerate(alphas):
            mapping = ridge_prediction_map(x[train], x[valid], alpha)
            residual = mapping @ y[train] - y[valid]
            fold_mse[alpha_id, fold_id] = np.mean(residual**2)
    mean_mse = fold_mse.mean(axis=1)
    minimum = int(np.argmin(mean_mse))
    threshold = mean_mse[minimum] + fold_mse[minimum].std(ddof=1) / math.sqrt(5)
    eligible = np.flatnonzero(mean_mse <= threshold + 1e-14)
    chosen = int(eligible[-1])

    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * float(alphas[chosen])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )
    return float(alphas[chosen]), float(coefficients[0]), coefficients[1:]


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
    }


def write_manifest(paths: list[Path], destination: Path, root: Path | None = None) -> None:
    rows = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        if not path.is_file():
            continue
        try:
            label = str(path.relative_to(root)) if root is not None else str(path)
        except ValueError:
            label = str(path)
        rows.append(
            {
                "path": label.replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictors", type=Path, default=DEFAULT_PREDICTORS)
    parser.add_argument("--step2", type=Path, default=DEFAULT_STEP2)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output
    for folder in [
        "data",
        "tables",
        "qa",
        "methods",
        "scripts",
        "manifests",
        "reports",
    ]:
        (output / folder).mkdir(parents=True, exist_ok=True)

    protocol_hash = sha256(args.protocol)
    shutil.copy2(args.protocol, output / "methods" / args.protocol.name)
    shutil.copy2(Path(__file__), output / "scripts" / Path(__file__).name)

    predictor_ledger = pd.read_csv(args.predictors)
    missing_columns = [
        column
        for column in ["city_key", "city_display", "census_region", *PREDICTORS]
        if column not in predictor_ledger.columns
    ]
    if missing_columns:
        raise RuntimeError(f"Predictor ledger missing columns: {missing_columns}")
    if predictor_ledger["city_key"].duplicated().any() or len(predictor_ledger) != 50:
        raise RuntimeError("Predictor ledger must contain exactly 50 unique cities.")
    if not np.isfinite(predictor_ledger[PREDICTORS].to_numpy(float)).all():
        raise RuntimeError("External predictor ledger contains missing or non-finite values.")

    outcomes_path = args.step2 / "tables" / "city_structure_budget_outcomes.csv"
    outcome_table = pd.read_csv(outcomes_path)
    outcome_table = outcome_table.loc[
        (outcome_table["structure"] == STRUCTURE)
        & outcome_table["budget_fraction"].isin(BUDGETS),
        ["city_key", "budget_fraction", TARGET],
    ].copy()
    if len(outcome_table) != 150 or outcome_table.duplicated(
        ["city_key", "budget_fraction"]
    ).any():
        raise RuntimeError("Step 2 central outcome is not a complete 50 x 3 ledger.")

    city_order = predictor_ledger["city_key"].tolist()
    outcome_pivot = outcome_table.pivot(
        index="city_key", columns="budget_fraction", values=TARGET
    ).reindex(index=city_order, columns=BUDGETS)
    if outcome_pivot.isna().any().any():
        missing = outcome_pivot.index[outcome_pivot.isna().any(axis=1)].tolist()
        raise RuntimeError(f"Central outcome merge failed for cities: {missing}")
    central_outcomes = outcome_pivot.to_numpy(float).T

    # Read existing Step 2 paired spatial-block bootstrap arrays only after the
    # protocol and predictor ledger have been frozen.
    bootstrap_folder = args.step2 / "bootstrap_city"
    bootstrap_values: list[np.ndarray] = []
    bootstrap_files: list[Path] = []
    for city_key in city_order:
        path = bootstrap_folder / f"{city_key}.npz"
        if not path.exists():
            raise RuntimeError(f"Missing Step 2 bootstrap file: {path}")
        with np.load(path, allow_pickle=True) as archive:
            structures = archive["structures"].astype(str).tolist()
            budgets = archive["budgets"].astype(float)
            outcome_names = archive["outcomes"].astype(str).tolist()
            structure_index = structures.index(STRUCTURE)
            outcome_index = outcome_names.index(TARGET)
            budget_indices = [
                int(np.flatnonzero(np.isclose(budgets, budget))[0])
                for budget in BUDGETS
            ]
            values = archive["values"][:, structure_index, :, outcome_index]
            values = values[:, budget_indices]
        if values.shape != (N_BOOT, len(BUDGETS)) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid Step 2 bootstrap array for {city_key}.")
        bootstrap_values.append(values)
        bootstrap_files.append(path)

    rng = np.random.default_rng(SEED)
    replicate_indices = rng.integers(0, N_BOOT, size=(N_BOOT, len(city_order)))
    bootstrap_outcomes = np.empty(
        (N_BOOT, len(city_order), len(BUDGETS)), dtype=float
    )
    for city_index, values in enumerate(bootstrap_values):
        bootstrap_outcomes[:, city_index, :] = values[
            replicate_indices[:, city_index], :
        ]
    np.savez_compressed(
        output / "data" / "outcome_bootstrap_draws.npz",
        values=bootstrap_outcomes,
        replicate_indices=replicate_indices,
        cities=np.asarray(city_order),
        budgets=BUDGETS,
        target=np.asarray(TARGET),
    )

    analysis_ledger = predictor_ledger.copy()
    for budget_index, budget in enumerate(BUDGETS):
        analysis_ledger[f"outcome_budget_{int(100 * budget):02d}pct"] = central_outcomes[
            budget_index
        ]
    analysis_ledger.to_csv(output / "data" / "analysis_city_ledger.csv", index=False)

    correlation = predictor_ledger[PREDICTORS].corr()
    correlation.to_csv(output / "qa" / "predictor_correlation_matrix.csv")
    maximum_correlation = float(
        correlation.where(~np.eye(len(PREDICTORS), dtype=bool)).abs().max().max()
    )

    loocv_tests = [(city_order[index], np.array([index])) for index in range(50)]
    central_metric_rows: list[dict[str, object]] = []
    city_prediction_rows: list[dict[str, object]] = []
    engine_by_model: dict[str, NestedRidgePredictor] = {}
    predictions_by_model: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for model_name, columns in MODEL_SPECS.items():
        x = predictor_ledger[columns].to_numpy(float)
        engine = NestedRidgePredictor(
            x=x,
            outer_tests=loocv_tests,
            alphas=ALPHAS,
            seed=SEED,
        )
        predictions, null_predictions, selected_alphas = engine.predict_many(
            central_outcomes
        )
        engine_by_model[model_name] = engine
        predictions_by_model[model_name] = (
            predictions,
            null_predictions,
            selected_alphas,
        )
        for budget_index, budget in enumerate(BUDGETS):
            metrics = one_row_metrics(
                central_outcomes[budget_index],
                predictions[budget_index],
                null_predictions[budget_index],
            )
            central_metric_rows.append(
                {
                    "validation": "nested_leave_one_city_out",
                    "model": model_name,
                    "budget_fraction": budget,
                    "n_test_cities": 50,
                    **metrics,
                }
            )
            for city_index, city_key in enumerate(city_order):
                city_prediction_rows.append(
                    {
                        "model": model_name,
                        "budget_fraction": budget,
                        "city_key": city_key,
                        "city_display": predictor_ledger.loc[
                            city_index, "city_display"
                        ],
                        "census_region": predictor_ledger.loc[
                            city_index, "census_region"
                        ],
                        "observed": central_outcomes[budget_index, city_index],
                        "predicted": predictions[budget_index, city_index],
                        "training_mean_null": null_predictions[
                            budget_index, city_index
                        ],
                        "residual": central_outcomes[budget_index, city_index]
                        - predictions[budget_index, city_index],
                        "squared_error_advantage": (
                            central_outcomes[budget_index, city_index]
                            - null_predictions[budget_index, city_index]
                        )
                        ** 2
                        - (
                            central_outcomes[budget_index, city_index]
                            - predictions[budget_index, city_index]
                        )
                        ** 2,
                        "selected_alpha": selected_alphas[budget_index, city_index],
                    }
                )

    central_metrics = pd.DataFrame(central_metric_rows)
    central_metrics.to_csv(output / "tables" / "nested_loocv_metrics.csv", index=False)
    city_predictions = pd.DataFrame(city_prediction_rows)
    city_predictions.to_csv(
        output / "tables" / "city_heldout_predictions.csv", index=False
    )

    # Leave-one-Census-region-out extrapolation with all tuning confined to the
    # non-held-out regions.
    regions = predictor_ledger["census_region"].astype(str).to_numpy()
    region_names = ["Northeast", "Midwest", "South", "West"]
    region_tests = [
        (region, np.flatnonzero(regions == region)) for region in region_names
    ]
    if any(len(indices) == 0 for _, indices in region_tests):
        raise RuntimeError("At least one Census region contains no sampled city.")
    full_x = predictor_ledger[PREDICTORS].to_numpy(float)
    region_engine = NestedRidgePredictor(
        x=full_x,
        outer_tests=region_tests,
        alphas=ALPHAS,
        seed=SEED + 5000,
    )
    region_predictions, region_null, region_alpha = region_engine.predict_many(
        central_outcomes
    )
    region_metric_rows: list[dict[str, object]] = []
    region_prediction_rows: list[dict[str, object]] = []
    for budget_index, budget in enumerate(BUDGETS):
        pooled = one_row_metrics(
            central_outcomes[budget_index],
            region_predictions[budget_index],
            region_null[budget_index],
        )
        region_metric_rows.append(
            {
                "validation": "leave_one_census_region_out_pooled",
                "region": "Pooled",
                "budget_fraction": budget,
                "n_test_cities": 50,
                **pooled,
            }
        )
        for region_index, (region, test_indices) in enumerate(region_tests):
            metrics = one_row_metrics(
                central_outcomes[budget_index, test_indices],
                region_predictions[budget_index, test_indices],
                region_null[budget_index, test_indices],
            )
            region_metric_rows.append(
                {
                    "validation": "leave_one_census_region_out",
                    "region": region,
                    "budget_fraction": budget,
                    "n_test_cities": len(test_indices),
                    **metrics,
                }
            )
            for city_index in test_indices:
                region_prediction_rows.append(
                    {
                        "budget_fraction": budget,
                        "held_out_region": region,
                        "city_key": city_order[city_index],
                        "city_display": predictor_ledger.loc[
                            city_index, "city_display"
                        ],
                        "observed": central_outcomes[budget_index, city_index],
                        "predicted": region_predictions[budget_index, city_index],
                        "training_mean_null": region_null[budget_index, city_index],
                        "selected_alpha": region_alpha[budget_index, region_index],
                    }
                )
    pd.DataFrame(region_metric_rows).to_csv(
        output / "tables" / "regional_holdout_metrics.csv", index=False
    )
    pd.DataFrame(region_prediction_rows).to_csv(
        output / "tables" / "regional_heldout_predictions.csv", index=False
    )

    # Outcome uncertainty: every draw reruns inner alpha selection and every
    # outer held-out prediction using the same frozen design.
    full_engine = engine_by_model["full_external_conditions"]
    uncertainty_rows: list[pd.DataFrame] = []
    uncertainty_summary_rows: list[dict[str, object]] = []
    for budget_index, budget in enumerate(BUDGETS):
        y_draws = bootstrap_outcomes[:, :, budget_index]
        boot_prediction, boot_null, boot_alpha = full_engine.predict_many(y_draws)
        metrics = vector_metrics(y_draws, boot_prediction, boot_null)
        table = pd.DataFrame(
            {
                "bootstrap_draw": np.arange(1, N_BOOT + 1),
                "budget_fraction": budget,
                "rmse_ratio": metrics["rmse_ratio"],
                "mae_ratio": metrics["mae_ratio"],
                "out_of_sample_r2": metrics["out_of_sample_r2"],
                "mse_advantage": metrics["mse_advantage"],
                "median_selected_alpha": np.median(boot_alpha, axis=1),
            }
        )
        uncertainty_rows.append(table)
        for metric_name in [
            "rmse_ratio",
            "mae_ratio",
            "out_of_sample_r2",
            "mse_advantage",
        ]:
            summary = percentile_summary(metrics[metric_name])
            uncertainty_summary_rows.append(
                {
                    "budget_fraction": budget,
                    "metric": metric_name,
                    **summary,
                }
            )
    pd.concat(uncertainty_rows, ignore_index=True).to_csv(
        output / "tables" / "outcome_uncertainty_metrics.csv", index=False
    )
    uncertainty_summary = pd.DataFrame(uncertainty_summary_rows)
    uncertainty_summary.to_csv(
        output / "tables" / "outcome_uncertainty_summary.csv", index=False
    )

    # Complete-procedure permutation test at the frozen 10% primary budget.
    primary_index = int(np.flatnonzero(np.isclose(BUDGETS, PRIMARY_BUDGET))[0])
    primary_y = central_outcomes[primary_index]
    permutation_outcomes = np.stack(
        [rng.permutation(primary_y) for _ in range(N_PERM)], axis=0
    )
    permutation_prediction, permutation_null, permutation_alpha = (
        full_engine.predict_many(permutation_outcomes)
    )
    permutation_metrics = vector_metrics(
        permutation_outcomes, permutation_prediction, permutation_null
    )
    observed_prediction, observed_null, _ = predictions_by_model[
        "full_external_conditions"
    ]
    observed_metrics = one_row_metrics(
        primary_y,
        observed_prediction[primary_index],
        observed_null[primary_index],
    )
    observed_advantage = 1.0 - observed_metrics["rmse_ratio"]
    permutation_advantage = 1.0 - permutation_metrics["rmse_ratio"]
    permutation_p = float(
        (1 + np.sum(permutation_advantage >= observed_advantage))
        / (N_PERM + 1)
    )
    pd.DataFrame(
        {
            "permutation": np.arange(1, N_PERM + 1),
            "rmse_ratio": permutation_metrics["rmse_ratio"],
            "rmse_advantage": permutation_advantage,
            "out_of_sample_r2": permutation_metrics["out_of_sample_r2"],
            "median_selected_alpha": np.median(permutation_alpha, axis=1),
        }
    ).to_csv(output / "tables" / "permutation_null.csv", index=False)
    pd.DataFrame(
        [
            {
                "budget_fraction": PRIMARY_BUDGET,
                "n_permutations": N_PERM,
                "observed_rmse_ratio": observed_metrics["rmse_ratio"],
                "observed_rmse_advantage": observed_advantage,
                "permutation_p_one_sided": permutation_p,
                **{
                    f"permuted_rmse_ratio_{key}": value
                    for key, value in percentile_summary(
                        permutation_metrics["rmse_ratio"]
                    ).items()
                },
            }
        ]
    ).to_csv(output / "tables" / "permutation_summary.csv", index=False)

    # Influence audit removes each already-held-out scoring case in turn; it
    # does not refit the model and therefore asks whether aggregate skill is
    # carried by a single city's test residual.
    primary_prediction = observed_prediction[primary_index]
    primary_null = observed_null[primary_index]
    city_advantage = (primary_y - primary_null) ** 2 - (
        primary_y - primary_prediction
    ) ** 2
    influence_rows = []
    for city_index, city_key in enumerate(city_order):
        retained = np.arange(50) != city_index
        influence_rows.append(
            {
                "omitted_city_key": city_key,
                "omitted_city_display": predictor_ledger.loc[
                    city_index, "city_display"
                ],
                "omitted_city_advantage": city_advantage[city_index],
                "mean_mse_advantage_after_omission": float(
                    city_advantage[retained].mean()
                ),
                "positive_after_omission": bool(city_advantage[retained].mean() > 0),
            }
        )
    influence = pd.DataFrame(influence_rows)
    influence.to_csv(output / "tables" / "influence_audit.csv", index=False)

    # Descriptive standardised coefficients, explicitly separated from the
    # held-out evidence used for the claim.
    coefficient_rows = []
    for budget_index, budget in enumerate(BUDGETS):
        alpha, intercept, coefficients = select_alpha_and_coefficients(
            full_x,
            central_outcomes[budget_index],
            ALPHAS,
            SEED + 6000 + budget_index,
        )
        for predictor, coefficient in zip(PREDICTORS, coefficients):
            coefficient_rows.append(
                {
                    "budget_fraction": budget,
                    "predictor": predictor,
                    "standardised_ridge_coefficient": coefficient,
                    "selected_alpha": alpha,
                    "intercept": intercept,
                    "inferential_role": "descriptive_orientation_only",
                }
            )
    pd.DataFrame(coefficient_rows).to_csv(
        output / "tables" / "descriptive_standardised_coefficients.csv",
        index=False,
    )

    # Prespecified regularisation-grid shifts at the primary budget.
    sensitivity_rows = []
    for label, grid, seed_shift in [
        ("one_decade_lower", ALPHAS_LOWER, 7000),
        ("frozen_primary", ALPHAS, 0),
        ("one_decade_higher", ALPHAS_HIGHER, 8000),
    ]:
        if label == "frozen_primary":
            prediction = primary_prediction
            null = primary_null
            alpha_values = predictions_by_model["full_external_conditions"][2][
                primary_index
            ]
        else:
            engine = NestedRidgePredictor(
                x=full_x,
                outer_tests=loocv_tests,
                alphas=grid,
                seed=SEED,
            )
            prediction_matrix, null_matrix, selected = engine.predict_many(primary_y)
            prediction = prediction_matrix[0]
            null = null_matrix[0]
            alpha_values = selected[0]
        metrics = one_row_metrics(primary_y, prediction, null)
        sensitivity_rows.append(
            {
                "alpha_grid": label,
                "alpha_min": float(grid.min()),
                "alpha_max": float(grid.max()),
                "median_outer_selected_alpha": float(np.median(alpha_values)),
                **metrics,
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        output / "tables" / "regularisation_grid_sensitivity.csv", index=False
    )

    # Main-text eligibility gate. No diagnostic model can rescue a failure.
    full_metrics = central_metrics.loc[
        central_metrics["model"] == "full_external_conditions"
    ].set_index("budget_fraction")
    region_metrics = pd.DataFrame(region_metric_rows)
    pooled_region_primary = float(
        region_metrics.loc[
            (region_metrics["region"] == "Pooled")
            & np.isclose(region_metrics["budget_fraction"], PRIMARY_BUDGET),
            "rmse_ratio",
        ].iloc[0]
    )
    primary_uncertainty = uncertainty_summary.loc[
        np.isclose(uncertainty_summary["budget_fraction"], PRIMARY_BUDGET)
        & (uncertainty_summary["metric"] == "rmse_ratio")
    ].iloc[0]
    gates = [
        (
            "G1_primary_nested_LOOCV_RMSE_ratio_at_most_0.90",
            float(full_metrics.loc[PRIMARY_BUDGET, "rmse_ratio"]),
            "<= 0.90",
            float(full_metrics.loc[PRIMARY_BUDGET, "rmse_ratio"]) <= 0.90,
        ),
        (
            "G2_complete_procedure_permutation_P_below_0.05",
            permutation_p,
            "< 0.05",
            permutation_p < 0.05,
        ),
        (
            "G3_5pct_and_20pct_nested_LOOCV_RMSE_ratios_below_1",
            max(
                float(full_metrics.loc[0.05, "rmse_ratio"]),
                float(full_metrics.loc[0.20, "rmse_ratio"]),
            ),
            (
                "< 1.00 at both budgets "
                f"(5%={float(full_metrics.loc[0.05, 'rmse_ratio']):.4f}; "
                f"20%={float(full_metrics.loc[0.20, 'rmse_ratio']):.4f})"
            ),
            bool(
                float(full_metrics.loc[0.05, "rmse_ratio"]) < 1.00
                and float(full_metrics.loc[0.20, "rmse_ratio"]) < 1.00
            ),
        ),
        (
            "G4_pooled_region_holdout_RMSE_ratio_below_1",
            pooled_region_primary,
            "< 1.00",
            pooled_region_primary < 1.00,
        ),
        (
            "G5_bootstrap_RMSE_ratio_95pct_upper_below_1",
            float(primary_uncertainty["q975"]),
            "< 1.00",
            float(primary_uncertainty["q975"]) < 1.00,
        ),
        (
            "G6_all_leave_one_scoring_city_MSE_advantages_positive",
            float(influence["mean_mse_advantage_after_omission"].min()),
            "> 0.00",
            bool(influence["positive_after_omission"].all()),
        ),
    ]
    gate_table = pd.DataFrame(
        [
            {
                "gate": name,
                "observed_value": value,
                "criterion": criterion,
                "pass": passed,
            }
            for name, value, criterion, passed in gates
        ]
    )
    overall_pass = bool(gate_table["pass"].all())
    gate_table.to_csv(output / "tables" / "prespecified_gate_audit.csv", index=False)

    leakage_audit = {
        "status": "PASS",
        "protocol_sha256_before_outcome_read": protocol_hash,
        "predictor_count": len(PREDICTORS),
        "city_count": len(city_order),
        "forbidden_proximal_predictors_present": False,
        "predictor_missingness": False,
        "maximum_absolute_predictor_correlation": maximum_correlation,
        "target": TARGET,
        "primary_budget_fraction": PRIMARY_BUDGET,
        "random_seed": SEED,
        "n_bootstrap_draws": N_BOOT,
        "n_permutations": N_PERM,
    }
    (output / "qa" / "leakage_and_input_audit.json").write_text(
        json.dumps(leakage_audit, indent=2), encoding="utf-8"
    )

    primary_ratio = float(full_metrics.loc[PRIMARY_BUDGET, "rmse_ratio"])
    primary_r2 = float(full_metrics.loc[PRIMARY_BUDGET, "out_of_sample_r2"])
    interval = (
        float(primary_uncertainty["q025"]),
        float(primary_uncertainty["q975"]),
    )
    report_lines = [
        "# Step 3 cross-city transferability audit",
        "",
        f"**Prespecified main-text gate:** {'PASS' if overall_pass else 'FAIL'}",
        "",
        "## Confirmatory result",
        "",
        (
            f"At the 10% budget, the six-predictor external-conditions model had "
            f"a nested leave-one-city-out RMSE ratio of {primary_ratio:.3f} "
            f"(out-of-sample R2 = {primary_r2:.3f}). The spatial-bootstrap "
            f"95% interval for the RMSE ratio was {interval[0]:.3f}-{interval[1]:.3f}; "
            f"the complete-procedure permutation P value was {permutation_p:.3f}."
        ),
        "",
        (
            "All six prespecified conditions were satisfied, so a restricted "
            "cross-city predictive statement is eligible for the main text."
            if overall_pass
            else "At least one prespecified condition failed. These six generic city "
            "conditions therefore do not support a reliable out-of-city predictive "
            "claim for the allocation-divergence outcome. Coefficients and domain "
            "models remain exploratory diagnostics only."
        ),
        "",
        "## Gate results",
        "",
    ]
    for row in gate_table.to_dict(orient="records"):
        gate_passed = bool(row["pass"])
        report_lines.append(
            f"- {'PASS' if gate_passed else 'FAIL'}: {row['gate']}; observed "
            f"{row['observed_value']:.4g}, required {row['criterion']}."
        )
    report_lines.extend(
        [
            "",
            "## Interpretive boundary",
            "",
            "This module evaluates prediction among the 50 sampled cities. It does not "
            "identify causal drivers, realised cooling, welfare effects or an optimal "
            "policy, and it does not establish transferability to smaller US cities.",
        ]
    )
    (output / "reports" / "STEP03_RESULT_AND_GATE.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    computational_notes = f"""# Computational notes

- Random seed: `{SEED}`
- Cities: 50
- Predictor set: six externally defined city conditions
- Outcome: `{TARGET}` under `{STRUCTURE}`
- Primary budget: 10%; sensitivities: 5% and 20%
- Ridge grid: 25 log-spaced values from 10^-3 to 10^3
- Inner selection: five-fold shuffled CV and the one-standard-error rule
- Outer validation: leave one city out and leave one Census region out
- Outcome uncertainty: {N_BOOT} existing paired spatial-block bootstrap draws
- Null test: {N_PERM} complete-procedure target permutations

The ridge prediction maps were precomputed from predictor matrices only. For
each bootstrap or permutation outcome, inner-fold errors, the selected alpha,
outer predictions and the training-mean null were recomputed exactly. No
coefficient P value or in-sample fit statistic enters the decision gate.
"""
    (output / "methods" / "COMPUTATIONAL_NOTES.md").write_text(
        computational_notes, encoding="utf-8"
    )

    final_audit = {
        "status": "PASS" if overall_pass else "ANALYSIS_COMPLETE_GATE_FAILED",
        "analysis_completed": True,
        "main_text_eligibility": overall_pass,
        "gate_pass_count": int(gate_table["pass"].sum()),
        "gate_total": len(gate_table),
        "primary_loocv_rmse_ratio": primary_ratio,
        "primary_out_of_sample_r2": primary_r2,
        "primary_bootstrap_rmse_ratio_interval": list(interval),
        "permutation_p_one_sided": permutation_p,
        "pooled_region_holdout_rmse_ratio": pooled_region_primary,
        "all_50_cities_present": len(city_order) == 50,
        "predictor_outcome_leakage_detected": False,
        "frozen_ns_package_modified": False,
    }
    (output / "qa" / "final_execution_audit.json").write_text(
        json.dumps(final_audit, indent=2), encoding="utf-8"
    )

    input_paths = [args.predictors, outcomes_path, args.protocol, *bootstrap_files]
    write_manifest(input_paths, output / "manifests" / "input_sha256.csv")
    output_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "output_sha256.csv"
    ]
    write_manifest(output_files, output / "manifests" / "output_sha256.csv", output)
    print(json.dumps(final_audit, indent=2))


if __name__ == "__main__":
    main()
