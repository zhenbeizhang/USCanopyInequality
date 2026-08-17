from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


PROJECT = analysis_store_root()
R3 = PROJECT / "Result" / "Result3"
SUPPORT = R3 / "Result3_support_aligned"
OUT = R3 / "versions" / "nonlinearity_role_audit"
FRAME_PATH = SUPPORT / "data" / "support_aligned_model_frame_34566.csv"
BLOCK_PATH = SUPPORT / "spatial_block_bootstrap" / "data" / "spatial_block_assignments.csv"

SEED = 20260804
FOLDS = 5
RAW_Z_SPECS = {
    "worldcover_built_pct": "z_worldcover_built_pct",
    "log_distance_to_water_m": "z_log_distance_to_water_m",
    "elevation_m": "z_elevation_m",
    "log_terrain_relief_1km_m": "z_log_terrain_relief_1km_m",
}
COVARIATES = [
    "log_population_density",
    "z_worldcover_built_pct",
    "z_log_distance_to_water_m",
    "z_elevation_m",
    "z_log_terrain_relief_1km_m",
]
KNOT_PROBABILITIES = {
    "linear": (),
    "rcs3": (0.10, 0.50, 0.90),
    "rcs4": (0.05, 0.35, 0.65, 0.95),
    "rcs5": (0.05, 0.275, 0.50, 0.725, 0.95),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=500)
    return parser.parse_args()


def ensure_dirs() -> None:
    for name in ("data", "tables", "diagnostics", "qa", "methods", "text", "source_data"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[valid], weights=w[valid]))


def weighted_quantile(x: np.ndarray, w: np.ndarray, probabilities: tuple[float, ...] | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    q = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[valid]
    w = w[valid]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= np.sum(w)
    return np.interp(q, cumulative, x)


def weighted_metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(w) & (w > 0)
    y = y[valid]
    pred = pred[valid]
    w = w[valid]
    error = y - pred
    mean = np.average(y, weights=w)
    variance = np.sum(w * (y - mean) ** 2)
    return {
        "n": int(valid.sum()),
        "weighted_rmse_C": float(np.sqrt(np.average(error**2, weights=w))),
        "weighted_mae_C": float(np.average(np.abs(error), weights=w)),
        "weighted_r2": float(1 - np.sum(w * error**2) / variance),
    }


def standardize_from_training(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for source, target in RAW_Z_SPECS.items():
        for city_key, idx in train.groupby("city_key", sort=False).groups.items():
            values = pd.to_numeric(train.loc[idx, source], errors="coerce").to_numpy(float)
            weights = train.loc[idx, "population_model_weight"].to_numpy(float)
            mean = weighted_mean(values, weights)
            variance = weighted_mean((values - mean) ** 2, weights)
            sd = math.sqrt(max(variance, 0.0))
            train.loc[idx, target] = 0.0 if sd <= 1e-12 else (values - mean) / sd
            test_idx = test.index[test["city_key"].eq(city_key)]
            test_values = pd.to_numeric(test.loc[test_idx, source], errors="coerce").to_numpy(float)
            test.loc[test_idx, target] = 0.0 if sd <= 1e-12 else (test_values - mean) / sd
    return train, test


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if len(knots) < 3:
        return np.empty((len(x), 0), dtype=float)
    if np.any(np.diff(knots) <= 0):
        raise ValueError(f"Restricted cubic spline knots are not strictly increasing: {knots}")
    scale = max((knots[-1] - knots[0]) ** 2, 1e-12)
    d_last_1 = np.maximum(x - knots[-2], 0.0) ** 3
    d_last = np.maximum(x - knots[-1], 0.0) ** 3
    columns = []
    denominator = knots[-1] - knots[-2]
    for knot in knots[:-2]:
        d = np.maximum(x - knot, 0.0) ** 3
        term = (
            d
            - d_last_1 * (knots[-1] - knot) / denominator
            + d_last * (knots[-2] - knot) / denominator
        ) / scale
        columns.append(term)
    return np.column_stack(columns)


def make_design(frame: pd.DataFrame, model_id: str, knots: np.ndarray | None) -> np.ndarray:
    canopy = frame["canopy_pct_10pp"].to_numpy(float)
    columns = [canopy[:, None]]
    if model_id != "linear":
        if knots is None:
            raise ValueError("Spline knots are required.")
        columns.append(rcs_basis(canopy, knots))
    columns.append(frame[COVARIATES].to_numpy(float))
    return np.column_stack(columns)


def within_transform(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_out = np.empty_like(y, dtype=float)
    x_out = np.empty_like(x, dtype=float)
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        wg = w[idx]
        y_out[idx] = y[idx] - np.average(y[idx], weights=wg)
        x_out[idx] = x[idx] - np.average(x[idx], weights=wg, axis=0)
    return y_out, x_out


def fit_within(
    frame: pd.DataFrame,
    model_id: str,
    knots: np.ndarray | None,
    robust: bool,
) -> dict[str, object]:
    y = frame["heat_anomaly_C"].to_numpy(float)
    w = frame["population_model_weight"].to_numpy(float)
    groups = frame["city_key"].to_numpy(str)
    x = make_design(frame, model_id, knots)
    yw, xw = within_transform(y, x, w, groups)
    result = sm.WLS(yw, xw, weights=w).fit()
    if robust:
        result = result.get_robustcov_results(
            cov_type="cluster",
            groups=pd.factorize(groups, sort=True)[0],
            use_correction=True,
        )
    return {"result": result, "x": x, "yw": yw, "xw": xw}


def predict_held(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_id: str,
    knots: np.ndarray | None,
) -> np.ndarray:
    fitted = fit_within(train, model_id, knots, robust=False)["result"]
    beta = np.asarray(fitted.params, dtype=float)
    x_train = make_design(train, model_id, knots)
    x_test = make_design(test, model_id, knots)
    residualized = train["heat_anomaly_C"].to_numpy(float) - x_train @ beta
    intercept = {}
    for city_key, idx in train.groupby("city_key", sort=False).groups.items():
        intercept[city_key] = weighted_mean(
            residualized[train.index.get_indexer(idx)],
            train.loc[idx, "population_model_weight"].to_numpy(float),
        )
    return np.asarray([intercept[key] for key in test["city_key"]], dtype=float) + x_test @ beta


def curve_design(grid: np.ndarray, model_id: str, knots: np.ndarray | None, n_covariates: int = 5) -> np.ndarray:
    columns = [grid[:, None]]
    if model_id != "linear":
        columns.append(rcs_basis(grid, np.asarray(knots)))
    columns.append(np.zeros((len(grid), n_covariates), dtype=float))
    return np.column_stack(columns)


def nonlinear_wald(result, nonlinear_count: int, n_cities: int) -> tuple[float, float, int]:
    if nonlinear_count <= 0:
        return float("nan"), float("nan"), 0
    beta = np.asarray(result.params, dtype=float)[1 : 1 + nonlinear_count]
    covariance = np.asarray(result.cov_params(), dtype=float)[1 : 1 + nonlinear_count, 1 : 1 + nonlinear_count]
    statistic = float(beta.T @ np.linalg.pinv(covariance) @ beta)
    f_statistic = statistic / nonlinear_count
    p_value = float(stats.f.sf(f_statistic, nonlinear_count, n_cities - 1))
    return f_statistic, p_value, nonlinear_count


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(FRAME_PATH, dtype={"GEOID_str": str}, low_memory=False)
    if len(frame) != 34_566 or frame["city_key"].nunique() != 50:
        raise RuntimeError("The frozen Result 3 frame is not 34,566 units in 50 cities.")
    required = [
        "heat_anomaly_C",
        "canopy_pct_10pp",
        "population_model_weight",
        "spatial_fold",
        *COVARIATES,
        *RAW_Z_SPECS,
    ]
    if frame[required].isna().any().any():
        raise RuntimeError("The frozen frame contains missing model values.")
    if frame[["city_key", "GEOID_str"]].duplicated().any():
        raise RuntimeError("Duplicate city-unit pairs detected.")
    return frame.sort_values("support_row").reset_index(drop=True)


def run_cross_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    predictions = []
    for fold in range(FOLDS):
        train_raw = frame.loc[frame["spatial_fold"].ne(fold)].copy()
        test_raw = frame.loc[frame["spatial_fold"].eq(fold)].copy()
        train, test = standardize_from_training(train_raw, test_raw)
        for model_id, probabilities in KNOT_PROBABILITIES.items():
            knots = None
            if probabilities:
                knots = weighted_quantile(
                    train["canopy_pct_10pp"].to_numpy(float),
                    train["population_model_weight"].to_numpy(float),
                    probabilities,
                )
            pred = predict_held(train, test, model_id, knots)
            values = weighted_metrics(
                test["heat_anomaly_C"].to_numpy(float),
                pred,
                test["population_model_weight"].to_numpy(float),
            )
            metrics.append(
                {
                    "outer_fold": fold,
                    "model_id": model_id,
                    "knot_probabilities": ",".join(map(str, probabilities)) if probabilities else "none",
                    "knots_canopy_pct": ",".join(f"{value * 10:.3f}" for value in knots) if knots is not None else "none",
                    **values,
                }
            )
            part = test[["support_row", "analysis_record_id", "city_key", "GEOID_str"]].copy()
            part["outer_fold"] = fold
            part["model_id"] = model_id
            part["observed_C"] = test["heat_anomaly_C"].to_numpy(float)
            part["predicted_C"] = pred
            part["population_weight"] = test["population_model_weight"].to_numpy(float)
            predictions.append(part)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def bootstrap_curve(
    frame: pd.DataFrame,
    blocks: pd.DataFrame,
    knots: np.ndarray,
    grid: np.ndarray,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = frame.merge(blocks, on=["support_row", "analysis_record_id", "city_key", "GEOID_str"], validate="one_to_one")
    y = merged["heat_anomaly_C"].to_numpy(float)
    w = merged["population_model_weight"].to_numpy(float)
    x_rcs = make_design(merged, "rcs4", knots)
    linear_indices = np.r_[0, np.arange(1 + (len(knots) - 2), x_rcs.shape[1])]
    grid_rcs = curve_design(grid, "rcs4", knots)
    grid_linear = curve_design(grid, "linear", None)
    reference = float(weighted_quantile(
        merged["canopy_pct_10pp"].to_numpy(float), w, (0.50,)
    )[0])
    ref_rcs = curve_design(np.array([reference]), "rcs4", knots)[0]
    ref_linear = curve_design(np.array([reference]), "linear", None)[0]

    city_rows: dict[str, np.ndarray] = {
        city: np.flatnonzero(merged["city_key"].to_numpy(str) == city)
        for city in sorted(merged["city_key"].unique())
    }
    city_keys = sorted(city_rows)
    origins = [f"main_block_origin{i}" for i in range(4)]
    rng = np.random.default_rng(SEED)
    curve_rows = []
    summary_rows = []

    for replicate in range(replicates):
        sampled_cities = rng.choice(city_keys, size=len(city_keys), replace=True)
        xtwx_rcs = np.zeros((x_rcs.shape[1], x_rcs.shape[1]), dtype=float)
        xtwy_rcs = np.zeros(x_rcs.shape[1], dtype=float)
        xtwx_linear = np.zeros((len(linear_indices), len(linear_indices)), dtype=float)
        xtwy_linear = np.zeros(len(linear_indices), dtype=float)
        n_rows = 0

        for city_key in sampled_cities:
            base_idx = city_rows[city_key]
            origin = origins[int(rng.integers(0, len(origins)))]
            labels = merged.loc[base_idx, origin].to_numpy()
            unique = np.unique(labels)
            sampled_labels = rng.choice(unique, size=len(unique), replace=True)
            selected = np.concatenate([base_idx[labels == label] for label in sampled_labels])
            yg = y[selected]
            wg = w[selected]
            xg_rcs = x_rcs[selected]
            yg = yg - np.average(yg, weights=wg)
            xg_rcs = xg_rcs - np.average(xg_rcs, weights=wg, axis=0)
            sqrt_w = np.sqrt(wg)
            xw_rcs = xg_rcs * sqrt_w[:, None]
            yw = yg * sqrt_w
            xtwx_rcs += xw_rcs.T @ xw_rcs
            xtwy_rcs += xw_rcs.T @ yw
            xw_linear = xw_rcs[:, linear_indices]
            xtwx_linear += xw_linear.T @ xw_linear
            xtwy_linear += xw_linear.T @ yw
            n_rows += len(selected)

        beta_rcs = np.linalg.pinv(xtwx_rcs) @ xtwy_rcs
        beta_linear = np.linalg.pinv(xtwx_linear) @ xtwy_linear
        curve_rcs = (grid_rcs - ref_rcs) @ beta_rcs
        curve_linear = (grid_linear - ref_linear) @ beta_linear
        for canopy, rcs_value, linear_value in zip(grid * 10, curve_rcs, curve_linear, strict=True):
            curve_rows.append(
                {
                    "replicate": replicate,
                    "canopy_pct": canopy,
                    "rcs4_relative_LST_C": rcs_value,
                    "linear_relative_LST_C": linear_value,
                }
            )
        summary_rows.append(
            {
                "replicate": replicate,
                "sampled_rows": n_rows,
                "linear_slope_C_per_10pp": beta_linear[0],
                "rcs4_linear_component": beta_rcs[0],
                "maximum_absolute_curve_difference_C": float(np.max(np.abs(curve_rcs - curve_linear))),
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    frame = load_frame()
    weights = frame["population_model_weight"].to_numpy(float)
    canopy = frame["canopy_pct_10pp"].to_numpy(float)

    full_rows = []
    full_fits = {}
    knot_lookup = {}
    for model_id, probabilities in KNOT_PROBABILITIES.items():
        knots = weighted_quantile(canopy, weights, probabilities) if probabilities else None
        knot_lookup[model_id] = knots
        fit = fit_within(frame, model_id, knots, robust=True)["result"]
        full_fits[model_id] = fit
        nonlinear_count = 0 if knots is None else len(knots) - 2
        f_stat, p_value, df_num = nonlinear_wald(fit, nonlinear_count, frame["city_key"].nunique())
        full_rows.append(
            {
                "model_id": model_id,
                "knot_probabilities": ",".join(map(str, probabilities)) if probabilities else "none",
                "knots_canopy_pct": ",".join(f"{value * 10:.3f}" for value in knots) if knots is not None else "none",
                "parameters": len(fit.params),
                "linear_component_C_per_10pp": float(fit.params[0]),
                "linear_component_cluster_se": float(fit.bse[0]),
                "nonlinearity_wald_F": f_stat,
                "nonlinearity_df_num": df_num,
                "nonlinearity_df_den": 49 if df_num else np.nan,
                "nonlinearity_city_cluster_p": p_value,
            }
        )
    full_table = pd.DataFrame(full_rows)

    linear_beta = float(full_fits["linear"].params[0])
    existing = pd.read_csv(SUPPORT / "tables" / "national_canopy_lst_model_comparison.csv")
    expected = float(existing.loc[existing["model_id"].eq("M2_common_FE_environment"), "coefficient_C_per_10pp_canopy"].iloc[0])
    if abs(linear_beta - expected) > 1e-8:
        raise RuntimeError(f"Within estimator failed to reproduce M2 common FE: {linear_beta} vs {expected}")

    cv, cv_predictions = run_cross_validation(frame)
    cv_summary = []
    for model_id, sub in cv_predictions.groupby("model_id", sort=False):
        cv_summary.append(
            {
                "model_id": model_id,
                **weighted_metrics(
                    sub["observed_C"].to_numpy(float),
                    sub["predicted_C"].to_numpy(float),
                    sub["population_weight"].to_numpy(float),
                ),
            }
        )
    cv_summary = pd.DataFrame(cv_summary)
    linear_rmse = float(cv_summary.loc[cv_summary["model_id"].eq("linear"), "weighted_rmse_C"].iloc[0])
    cv_summary["rmse_change_vs_linear_C"] = cv_summary["weighted_rmse_C"] - linear_rmse

    q05, q50, q95 = weighted_quantile(canopy, weights, (0.05, 0.50, 0.95))
    grid = np.linspace(q05, q95, 61)
    rcs_fit = full_fits["rcs4"]
    design = curve_design(grid, "rcs4", knot_lookup["rcs4"])
    reference_design = curve_design(np.array([q50]), "rcs4", knot_lookup["rcs4"])[0]
    contrast = design - reference_design
    beta = np.asarray(rcs_fit.params, dtype=float)
    covariance = np.asarray(rcs_fit.cov_params(), dtype=float)
    estimate = contrast @ beta
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", contrast, covariance, contrast), 0.0))
    curve = pd.DataFrame(
        {
            "canopy_pct": grid * 10,
            "relative_to_weighted_median_LST_C": estimate,
            "city_cluster_ci_low": estimate - stats.t.ppf(0.975, 49) * se,
            "city_cluster_ci_high": estimate + stats.t.ppf(0.975, 49) * se,
            "weighted_median_canopy_pct": q50 * 10,
        }
    )

    blocks = pd.read_csv(BLOCK_PATH, dtype={"GEOID_str": str})
    bootstrap_curve_values, bootstrap_summary = bootstrap_curve(
        frame,
        blocks,
        knot_lookup["rcs4"],
        grid,
        args.bootstrap,
    )
    bootstrap_curve_summary = (
        bootstrap_curve_values.groupby("canopy_pct", as_index=False)
        .agg(
            rcs4_bootstrap_median_C=("rcs4_relative_LST_C", "median"),
            rcs4_bootstrap_ci_low=("rcs4_relative_LST_C", lambda x: x.quantile(0.025)),
            rcs4_bootstrap_ci_high=("rcs4_relative_LST_C", lambda x: x.quantile(0.975)),
            linear_bootstrap_median_C=("linear_relative_LST_C", "median"),
            linear_bootstrap_ci_low=("linear_relative_LST_C", lambda x: x.quantile(0.025)),
            linear_bootstrap_ci_high=("linear_relative_LST_C", lambda x: x.quantile(0.975)),
        )
    )
    curve = curve.merge(bootstrap_curve_summary, on="canopy_pct", validate="one_to_one")

    quartiles = weighted_quantile(canopy, weights, (0.25, 0.50, 0.75))
    local_rows = []
    for label, midpoint in zip(("lower_quartile", "median", "upper_quartile"), quartiles, strict=True):
        low = max(q05, midpoint - 0.5)
        high = min(q95, low + 1.0)
        low = high - 1.0
        delta = curve_design(np.array([high]), "rcs4", knot_lookup["rcs4"])[0] - curve_design(
            np.array([low]), "rcs4", knot_lookup["rcs4"]
        )[0]
        point = float(delta @ beta)
        boot = []
        for _, replicate in bootstrap_curve_values.groupby("replicate", sort=False):
            values = np.interp([low * 10, high * 10], replicate["canopy_pct"], replicate["rcs4_relative_LST_C"])
            boot.append(values[1] - values[0])
        local_rows.append(
            {
                "location": label,
                "canopy_interval_pct": f"{low * 10:.1f}-{high * 10:.1f}",
                "adjusted_LST_difference_C_per_10pp": point,
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            }
        )
    local = pd.DataFrame(local_rows)

    roles = existing.loc[existing["model_id"].isin(["M2_city_specific_REML", "M3_spatial_GAM_REML"])].copy()
    cv_existing = pd.read_csv(SUPPORT / "tables" / "nested_spatial_cv_outer_performance.csv")
    moran = pd.read_csv(SUPPORT / "tables" / "residual_moran_summary.csv")
    roles = roles.merge(cv_existing, on="model_id", suffixes=("", "_cv"), validate="one_to_one")
    roles = roles.merge(moran, on="model_id", validate="one_to_one")
    roles["inferential_role"] = roles["model_id"].map(
        {
            "M2_city_specific_REML": "Primary non-spatial adjusted association and city-heterogeneity estimator; lower held-block RMSE.",
            "M3_spatial_GAM_REML": "Spatial-sensitivity estimator and residual-structure diagnostic; lower in-sample residual Moran's I.",
        }
    )

    full_table.to_csv(OUT / "tables" / "result3_nonlinearity_full_sample.csv", index=False, encoding="utf-8-sig")
    cv.to_csv(OUT / "tables" / "result3_nonlinearity_cv_by_fold.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(OUT / "tables" / "result3_nonlinearity_cv_summary.csv", index=False, encoding="utf-8-sig")
    local.to_csv(OUT / "tables" / "result3_nonlinear_local_contrasts.csv", index=False, encoding="utf-8-sig")
    roles.to_csv(OUT / "tables" / "result3_m2_m3_complementary_roles.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(OUT / "source_data" / "result3_rcs4_adjusted_curve.csv", index=False, encoding="utf-8-sig")
    bootstrap_summary.to_csv(OUT / "diagnostics" / "result3_nonlinearity_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    cv_predictions.to_csv(OUT / "diagnostics" / "result3_nonlinearity_cv_predictions.csv.gz", index=False, compression="gzip")

    primary = full_table.loc[full_table["model_id"].eq("rcs4")].iloc[0]
    rcs_rmse = float(cv_summary.loc[cv_summary["model_id"].eq("rcs4"), "weighted_rmse_C"].iloc[0])
    interpretation = (
        "evidence_of_material_nonlinearity"
        if primary["nonlinearity_city_cluster_p"] < 0.05 and rcs_rmse < linear_rmse - 0.01
        else "no_material_predictive_advantage_over_linear_summary"
    )
    audit = {
        "status": "PASS",
        "analysis_units": int(len(frame)),
        "cities": int(frame["city_key"].nunique()),
        "bootstrap_replicates": int(args.bootstrap),
        "linear_m2_reproduction_error": abs(linear_beta - expected),
        "primary_nonlinear_model": "four-knot restricted cubic spline",
        "primary_knots_canopy_pct": [float(x * 10) for x in knot_lookup["rcs4"]],
        "nonlinearity_city_cluster_p": float(primary["nonlinearity_city_cluster_p"]),
        "linear_spatial_cv_rmse_C": linear_rmse,
        "rcs4_spatial_cv_rmse_C": rcs_rmse,
        "rcs4_minus_linear_rmse_C": rcs_rmse - linear_rmse,
        "interpretation": interpretation,
        "m2_m3_decision": "Retain both as complementary estimators; do not select M3 as universally superior.",
        "input_hashes": {
            str(FRAME_PATH): sha256(FRAME_PATH),
            str(BLOCK_PATH): sha256(BLOCK_PATH),
        },
    }
    (OUT / "qa" / "result3_nonlinearity_role_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "text" / "result3_interpretation.txt").write_text(
        "The four-knot restricted cubic spline was evaluated as a model-form audit on the same 34,566-unit frame. "
        f"Its city-clustered nonlinearity test was p={primary['nonlinearity_city_cluster_p']:.4g}, and its held-block "
        f"RMSE changed by {rcs_rmse - linear_rmse:+.4f} C relative to the linear M2 model. "
        "M2 and M3 are therefore reported as complementary estimators: M2 has lower held-block prediction error, "
        "whereas M3 leaves less in-sample spatial residual structure. Neither model is treated as universally superior.\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OUT / "methods" / Path(__file__).name)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
