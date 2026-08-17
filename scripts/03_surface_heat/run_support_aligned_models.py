from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


OLD_ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_spatial_residual_audit')))
OLD_SCRIPT = Path(__file__).with_name("spatial_model_core.py")
OLD_FRAME = OLD_ROOT / "data" / "model_analysis_frame.csv"
PRIMARY_SUPPORT = Path(
    str(analysis_store_root().joinpath('Result', 'Result2', 'Result2_module1_boundary_inference_audit', 'source_data', 'result2_primary_fully_contained_block_groups.csv'))
)
OUT_ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned')))

DATA_DIR = OUT_ROOT / "data"
TABLE_DIR = OUT_ROOT / "tables"
DIAG_DIR = OUT_ROOT / "diagnostics"
QA_DIR = OUT_ROOT / "qa"
TEXT_DIR = OUT_ROOT / "text"

SEED = 20260721
N_FOLDS = 5
N_NEIGHBOURS = 8
MORAN_PERMUTATIONS = 199
N_SPLINES_GRID = (6, 8, 10)
LAMBDA_GRID = (0.6, 3.0, 10.0)

MODEL1_COVARIATES = ["log_population_density"]
MODEL2_COVARIATES = [
    "log_population_density",
    "z_worldcover_built_pct",
    "z_log_distance_to_water_m",
    "z_elevation_m",
    "z_log_terrain_relief_1km_m",
]
GAM_LINEAR_COLUMNS = ["canopy_pct_10pp", *MODEL2_COVARIATES]
RAW_Z_SPECS = {
    "worldcover_built_pct": "z_worldcover_built_pct",
    "log_distance_to_water_m": "z_log_distance_to_water_m",
    "elevation_m": "z_elevation_m",
    "log_terrain_relief_1km_m": "z_log_terrain_relief_1km_m",
}


def load_old_module():
    spec = importlib.util.spec_from_file_location("old_result3_audit", OLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {OLD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


old = load_old_module()


@dataclass
class HierarchicalFit:
    national: dict[str, object]
    city: pd.DataFrame
    predictions: pd.DataFrame


def ensure_dirs() -> None:
    for path in (DATA_DIR, TABLE_DIR, DIAG_DIR, QA_DIR, TEXT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_geoid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2:
        return float("nan")
    x = values[valid]
    w = weights[valid]
    mean = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mean) ** 2, weights=w)))


def standardize_support(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, target in RAW_Z_SPECS.items():
        values_out = pd.Series(np.nan, index=out.index, dtype=float)
        for _, index in out.groupby("city_key", sort=False).groups.items():
            values = pd.to_numeric(out.loc[index, source], errors="coerce").to_numpy(float)
            weights = pd.to_numeric(
                out.loc[index, "population_model_weight"], errors="coerce"
            ).to_numpy(float)
            mean = weighted_mean(values, weights)
            sd = weighted_sd(values, weights)
            values_out.loc[index] = 0.0 if not np.isfinite(sd) or sd <= 1e-12 else (values - mean) / sd
        out[target] = values_out

    for _, index in out.groupby("city_key", sort=False).groups.items():
        x = pd.to_numeric(out.loc[index, "x_5070_m"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(out.loc[index, "y_5070_m"], errors="coerce").to_numpy(float)
        out.loc[index, "x_city_0_1"] = (x - np.nanmin(x)) / max(np.nanmax(x) - np.nanmin(x), 1.0)
        out.loc[index, "y_city_0_1"] = (y - np.nanmin(y)) / max(np.nanmax(y) - np.nanmin(y), 1.0)
    return out


def transform_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    for source, target in RAW_Z_SPECS.items():
        for city_key, index in train_out.groupby("city_key", sort=False).groups.items():
            tr_values = pd.to_numeric(train_out.loc[index, source], errors="coerce").to_numpy(float)
            tr_weights = pd.to_numeric(
                train_out.loc[index, "population_model_weight"], errors="coerce"
            ).to_numpy(float)
            mean = weighted_mean(tr_values, tr_weights)
            sd = weighted_sd(tr_values, tr_weights)
            if not np.isfinite(sd) or sd <= 1e-12:
                train_out.loc[index, target] = 0.0
                test_out.loc[test_out["city_key"].eq(city_key), target] = 0.0
            else:
                train_out.loc[index, target] = (tr_values - mean) / sd
                test_index = test_out.index[test_out["city_key"].eq(city_key)]
                test_values = pd.to_numeric(test_out.loc[test_index, source], errors="coerce").to_numpy(float)
                test_out.loc[test_index, target] = (test_values - mean) / sd
    return train_out, test_out


def build_support_frame() -> pd.DataFrame:
    ensure_dirs()
    frame = pd.read_csv(OLD_FRAME, dtype={"GEOID_str": str}, low_memory=False)
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    complete = frame["complete_spatial_model"].astype(str).str.lower().eq("true")
    frame = frame.loc[complete].copy()
    frame["city_intersection_fraction"] = (
        pd.to_numeric(frame["clipped_area_m2"], errors="coerce")
        / pd.to_numeric(frame["original_bg_area_m2"], errors="coerce")
    )

    support = pd.read_csv(PRIMARY_SUPPORT, dtype=str, low_memory=False)
    support["GEOID_str"] = normalize_geoid(support["spatial_unit_id"])
    support_ids = (
        support[["result_city_key", "GEOID_str"]]
        .drop_duplicates()
        .rename(columns={"result_city_key": "city_key"})
    )
    primary = frame.merge(
        support_ids,
        on=["city_key", "GEOID_str"],
        how="inner",
        validate="one_to_one",
    )
    primary = standardize_support(primary).sort_values(["city_key", "GEOID_str"]).reset_index(drop=True)
    primary["support_frame"] = "fully_contained_primary"
    primary["support_row"] = np.arange(len(primary), dtype=np.int32)

    if len(primary) != 34_566 or primary["city_key"].nunique() != 50:
        raise RuntimeError(
            f"Expected 34,566 support-aligned rows and 50 cities; found {len(primary):,} rows "
            f"and {primary['city_key'].nunique()} cities."
        )

    primary.to_csv(DATA_DIR / "support_aligned_model_frame_34566.csv", index=False, encoding="utf-8-sig")

    sensitivity_rows: list[dict[str, object]] = []
    for label, threshold in (("intersection_ge_0p95", 0.95), ("intersection_ge_0p80", 0.80)):
        selected = frame.loc[frame["city_intersection_fraction"].ge(threshold)].copy()
        selected = standardize_support(selected).sort_values(["city_key", "GEOID_str"]).reset_index(drop=True)
        selected["support_frame"] = label
        selected.to_csv(DATA_DIR / f"{label}_model_frame.csv", index=False, encoding="utf-8-sig")
        sensitivity_rows.append(
            {
                "support_frame": label,
                "threshold": threshold,
                "n_block_groups": int(len(selected)),
                "n_cities": int(selected["city_key"].nunique()),
            }
        )

    inventory = (
        primary.groupby(["city_key", "city_display"], as_index=False)
        .agg(
            n_block_groups=("GEOID_str", "size"),
            population=("population_model_weight", "sum"),
            minimum_intersection_fraction=("city_intersection_fraction", "min"),
            median_intersection_fraction=("city_intersection_fraction", "median"),
        )
        .sort_values("city_key")
    )
    inventory.to_csv(TABLE_DIR / "support_aligned_sample_inventory_by_city.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "support_frame": "fully_contained_primary",
                "threshold": "Result 2 audited fully-contained list",
                "n_block_groups": len(primary),
                "n_cities": primary["city_key"].nunique(),
            },
            *sensitivity_rows,
            {
                "support_frame": "legacy_complete_spatial_model",
                "threshold": "legacy clipped-city inclusion",
                "n_block_groups": len(frame),
                "n_cities": frame["city_key"].nunique(),
            },
        ]
    ).to_csv(TABLE_DIR / "support_frame_inventory.csv", index=False, encoding="utf-8-sig")

    primary_pairs = pd.MultiIndex.from_frame(primary[["city_key", "GEOID_str"]])
    frame_pairs = pd.MultiIndex.from_frame(frame[["city_key", "GEOID_str"]])
    excluded = frame.loc[~frame_pairs.isin(primary_pairs)].copy()
    excluded[[
        "analysis_record_id",
        "city_key",
        "city_display",
        "GEOID_str",
        "city_intersection_fraction",
        "population_model_weight",
        "canopy_pct",
        "mean_LST_C",
    ]].to_csv(DATA_DIR / "legacy_rows_excluded_from_primary_support.csv", index=False, encoding="utf-8-sig")

    write_json(
        QA_DIR / "support_alignment_validation.json",
        {
            "primary_rows": int(len(primary)),
            "primary_cities": int(primary["city_key"].nunique()),
            "legacy_complete_rows": int(len(frame)),
            "legacy_rows_excluded": int(len(excluded)),
            "primary_duplicate_city_geoid_pairs": int(
                primary.duplicated(["city_key", "GEOID_str"]).sum()
            ),
            "geoids_represented_in_multiple_cities": int(
                primary.loc[primary["GEOID_str"].duplicated(keep=False), "GEOID_str"].nunique()
            ),
            "primary_missing_model_fields": int(
                primary[["heat_anomaly_C", "canopy_pct_10pp", "population_model_weight", *MODEL2_COVARIATES]]
                .isna()
                .any(axis=1)
                .sum()
            ),
            "support_rule": "Intersection with the audited Result 2 fully-contained block-group list; all model inputs refer to the same census unit support.",
        },
    )
    return primary


def load_primary() -> pd.DataFrame:
    path = DATA_DIR / "support_aligned_model_frame_34566.csv"
    if not path.exists():
        return build_support_frame()
    frame = pd.read_csv(path, dtype={"GEOID_str": str}, low_memory=False)
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    return frame


def design_matrix(
    frame: pd.DataFrame, covariates: list[str], city_levels: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    if city_levels is None:
        city_levels = sorted(frame["city_key"].astype(str).unique())
    city = pd.Categorical(frame["city_key"].astype(str), categories=city_levels)
    dummies = pd.get_dummies(city, prefix="city", drop_first=True, dtype=float)
    numeric = frame[["canopy_pct_10pp", *covariates]].astype(float).reset_index(drop=True)
    matrix = pd.concat(
        [pd.Series(1.0, index=numeric.index, name="const"), numeric, dummies.reset_index(drop=True)],
        axis=1,
    )
    return matrix, city_levels


def fit_common_fe(
    frame: pd.DataFrame, model_id: str, covariates: list[str]
) -> tuple[dict[str, object], pd.DataFrame]:
    matrix, _ = design_matrix(frame, covariates)
    outcome = frame["heat_anomaly_C"].to_numpy(float)
    weights = frame["population_model_weight"].to_numpy(float)
    fitted = sm.WLS(outcome, matrix, weights=weights).fit()
    robust = fitted.get_robustcov_results(
        cov_type="cluster",
        groups=frame["city_key"].astype(str).to_numpy(),
        use_correction=True,
    )
    index = list(matrix.columns).index("canopy_pct_10pp")
    beta = float(robust.params[index])
    se = float(robust.bse[index])
    df = max(frame["city_key"].nunique() - 1, 1)
    critical = float(stats.t.ppf(0.975, df=df))
    prediction = np.asarray(fitted.predict(matrix), dtype=float)
    national = {
        "model_id": model_id,
        "model_family": "population-weighted city fixed effects with a common canopy slope",
        "covariates": " + ".join(covariates),
        "coefficient_C_per_10pp_canopy": beta,
        "standard_error": se,
        "ci_low": beta - critical * se,
        "ci_high": beta + critical * se,
        "p_value": float(2 * stats.t.sf(abs(beta / se), df=df)),
        "prediction_interval_low": np.nan,
        "prediction_interval_high": np.nan,
        "tau_between_city": np.nan,
        "i2_city_heterogeneity": np.nan,
        "n_block_groups": int(len(frame)),
        "n_cities": int(frame["city_key"].nunique()),
        "inference": "city-clustered t interval",
    }
    pred = frame[["support_row", "analysis_record_id", "city_key", "city_display", "GEOID_str"]].copy()
    pred["observed"] = outcome
    pred["predicted"] = prediction
    pred["residual"] = outcome - prediction
    pred["model_id"] = model_id
    return national, pred


def fit_predict_common_fe(
    train: pd.DataFrame, test: pd.DataFrame, covariates: list[str]
) -> np.ndarray:
    levels = sorted(train["city_key"].astype(str).unique())
    x_train, _ = design_matrix(train, covariates, levels)
    x_test, _ = design_matrix(test, covariates, levels)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
    fit = sm.WLS(
        train["heat_anomaly_C"].to_numpy(float),
        x_train,
        weights=train["population_model_weight"].to_numpy(float),
    ).fit()
    return np.asarray(fit.predict(x_test), dtype=float)


def reml_pool(beta: np.ndarray, se: np.ndarray) -> dict[str, object]:
    return old.reml_random_effects(beta, se)


def hierarchical_national_record(
    model_id: str,
    model_family: str,
    covariates: list[str],
    pool: dict[str, object],
    n: int,
    n_cities: int,
    inference: str = "REML random-effects t interval",
) -> dict[str, object]:
    critical = float(stats.t.ppf(0.975, df=max(n_cities - 1, 1)))
    mu = float(pool["mu"])
    mu_se = float(pool["mu_se"])
    tau2 = float(pool["tau2"])
    prediction_sd = math.sqrt(tau2 + mu_se**2)
    return {
        "model_id": model_id,
        "model_family": model_family,
        "covariates": " + ".join(covariates),
        "coefficient_C_per_10pp_canopy": mu,
        "standard_error": mu_se,
        "ci_low": mu - critical * mu_se,
        "ci_high": mu + critical * mu_se,
        "p_value": float(2 * stats.t.sf(abs(mu / mu_se), df=max(n_cities - 1, 1))),
        "prediction_interval_low": mu - critical * prediction_sd,
        "prediction_interval_high": mu + critical * prediction_sd,
        "tau_between_city": float(pool["tau"]),
        "i2_city_heterogeneity": float(pool["i2"]),
        "n_block_groups": int(n),
        "n_cities": int(n_cities),
        "inference": inference,
    }


def fit_city_hierarchical_linear(
    frame: pd.DataFrame,
    model_id: str = "M2_city_specific_REML",
    covariates: list[str] = MODEL2_COVARIATES,
) -> HierarchicalFit:
    city_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for city_key, sub in frame.groupby("city_key", sort=True):
        x = sm.add_constant(sub[["canopy_pct_10pp", *covariates]].astype(float), has_constant="add")
        y = sub["heat_anomaly_C"].to_numpy(float)
        w = sub["population_model_weight"].to_numpy(float)
        fitted = sm.WLS(y, x, weights=w).fit(cov_type="HC3")
        beta = float(fitted.params["canopy_pct_10pp"])
        se = float(fitted.bse["canopy_pct_10pp"])
        city_rows.append(
            {
                "model_id": model_id,
                "city_key": city_key,
                "city_display": sub["city_display"].iloc[0],
                "raw_coefficient_C_per_10pp_canopy": beta,
                "raw_standard_error": se,
                "raw_ci_low": beta - 1.96 * se,
                "raw_ci_high": beta + 1.96 * se,
                "n_block_groups": int(len(sub)),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "support_row": sub["support_row"].to_numpy(int),
                    "analysis_record_id": sub["analysis_record_id"].to_numpy(),
                    "city_key": city_key,
                    "city_display": sub["city_display"].to_numpy(),
                    "GEOID_str": sub["GEOID_str"].to_numpy(),
                    "observed": y,
                    "raw_predicted": np.asarray(fitted.predict(x), dtype=float),
                    "canopy_pct_10pp": sub["canopy_pct_10pp"].to_numpy(float),
                    "population_model_weight": w,
                }
            )
        )

    city = pd.DataFrame(city_rows)
    pool = reml_pool(
        city["raw_coefficient_C_per_10pp_canopy"].to_numpy(float),
        city["raw_standard_error"].to_numpy(float),
    )
    city["coefficient_C_per_10pp_canopy"] = np.asarray(pool["shrunk"], dtype=float)
    city["standard_error"] = np.asarray(pool["shrunk_se"], dtype=float)
    city["ci_low"] = city["coefficient_C_per_10pp_canopy"] - 1.96 * city["standard_error"]
    city["ci_high"] = city["coefficient_C_per_10pp_canopy"] + 1.96 * city["standard_error"]

    prediction = pd.concat(prediction_rows, ignore_index=True)
    slopes = city.set_index("city_key")[["raw_coefficient_C_per_10pp_canopy", "coefficient_C_per_10pp_canopy"]]
    prediction = prediction.join(slopes, on="city_key")
    canopy_mean = prediction.groupby("city_key").apply(
        lambda group: weighted_mean(
            group["canopy_pct_10pp"].to_numpy(float), group["population_model_weight"].to_numpy(float)
        ),
        include_groups=False,
    )
    prediction["predicted"] = prediction["raw_predicted"] + (
        prediction["coefficient_C_per_10pp_canopy"]
        - prediction["raw_coefficient_C_per_10pp_canopy"]
    ) * (prediction["canopy_pct_10pp"] - prediction["city_key"].map(canopy_mean))
    prediction["residual"] = prediction["observed"] - prediction["predicted"]
    prediction["model_id"] = model_id
    national = hierarchical_national_record(
        model_id,
        "city-specific environmental WLS slopes with REML partial pooling",
        covariates,
        pool,
        len(prediction),
        city["city_key"].nunique(),
    )
    return HierarchicalFit(national, city, prediction)


def fit_predict_city_hierarchical_linear(
    train: pd.DataFrame, test: pd.DataFrame, covariates: list[str] = MODEL2_COVARIATES
) -> np.ndarray:
    rows: list[dict[str, object]] = []
    fitted_by_city: dict[str, tuple[object, float, float]] = {}
    for city_key, sub in train.groupby("city_key", sort=True):
        x = sm.add_constant(sub[["canopy_pct_10pp", *covariates]].astype(float), has_constant="add")
        fit = sm.WLS(
            sub["heat_anomaly_C"].to_numpy(float),
            x,
            weights=sub["population_model_weight"].to_numpy(float),
        ).fit(cov_type="HC3")
        beta = float(fit.params["canopy_pct_10pp"])
        se = float(fit.bse["canopy_pct_10pp"])
        canopy_mean = weighted_mean(
            sub["canopy_pct_10pp"].to_numpy(float), sub["population_model_weight"].to_numpy(float)
        )
        rows.append({"city_key": city_key, "beta": beta, "se": se})
        fitted_by_city[city_key] = (fit, beta, canopy_mean)
    slopes = pd.DataFrame(rows)
    pool = reml_pool(slopes["beta"].to_numpy(float), slopes["se"].to_numpy(float))
    slopes["shrunk"] = np.asarray(pool["shrunk"], dtype=float)
    shrunk = slopes.set_index("city_key")["shrunk"].to_dict()
    prediction = pd.Series(np.nan, index=test.index, dtype=float)
    for city_key, sub in test.groupby("city_key", sort=True):
        fit, beta, canopy_mean = fitted_by_city[city_key]
        x = sm.add_constant(sub[["canopy_pct_10pp", *covariates]].astype(float), has_constant="add")
        x = x.reindex(columns=fit.model.exog_names, fill_value=0.0)
        raw = np.asarray(fit.predict(x), dtype=float)
        prediction.loc[sub.index] = raw + (shrunk[city_key] - beta) * (
            sub["canopy_pct_10pp"].to_numpy(float) - canopy_mean
        )
    return prediction.loc[test.index].to_numpy(float)


def fit_one_city_gam(
    sub: pd.DataFrame, n_splines: int, lam: float
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    gam, index, prediction, coefficient, _ = old.fit_one_city_gam(
        sub, n_splines=n_splines, lam=lam
    )
    return gam, index, prediction, coefficient


def fit_hierarchical_spatial_gam(
    frame: pd.DataFrame,
    n_splines: int,
    lam: float,
    model_id: str = "M3_spatial_GAM_REML",
) -> HierarchicalFit:
    city_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for number, (city_key, sub) in enumerate(frame.groupby("city_key", sort=True), start=1):
        print(f"[Final M3 {number:02d}/50] {city_key}", flush=True)
        _, index, prediction, coefficient = fit_one_city_gam(sub, n_splines, lam)
        beta, se = map(float, coefficient)
        city_rows.append(
            {
                "model_id": model_id,
                "city_key": city_key,
                "city_display": sub["city_display"].iloc[0],
                "raw_coefficient_C_per_10pp_canopy": beta,
                "raw_standard_error": se,
                "raw_ci_low": beta - 1.96 * se,
                "raw_ci_high": beta + 1.96 * se,
                "n_block_groups": int(len(index)),
                "n_splines_each_dimension": n_splines,
                "lambda": lam,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "support_row": frame.loc[index, "support_row"].to_numpy(int),
                    "analysis_record_id": frame.loc[index, "analysis_record_id"].to_numpy(),
                    "city_key": city_key,
                    "city_display": frame.loc[index, "city_display"].to_numpy(),
                    "GEOID_str": frame.loc[index, "GEOID_str"].to_numpy(),
                    "observed": frame.loc[index, "heat_anomaly_C"].to_numpy(float),
                    "raw_predicted": prediction,
                    "canopy_pct_10pp": frame.loc[index, "canopy_pct_10pp"].to_numpy(float),
                    "population_model_weight": frame.loc[index, "population_model_weight"].to_numpy(float),
                }
            )
        )
    city = pd.DataFrame(city_rows)
    pool = reml_pool(
        city["raw_coefficient_C_per_10pp_canopy"].to_numpy(float),
        city["raw_standard_error"].to_numpy(float),
    )
    city["coefficient_C_per_10pp_canopy"] = np.asarray(pool["shrunk"], dtype=float)
    city["standard_error"] = np.asarray(pool["shrunk_se"], dtype=float)
    city["ci_low"] = city["coefficient_C_per_10pp_canopy"] - 1.96 * city["standard_error"]
    city["ci_high"] = city["coefficient_C_per_10pp_canopy"] + 1.96 * city["standard_error"]

    pred = pd.concat(prediction_rows, ignore_index=True)
    slopes = city.set_index("city_key")[["raw_coefficient_C_per_10pp_canopy", "coefficient_C_per_10pp_canopy"]]
    pred = pred.join(slopes, on="city_key")
    canopy_mean = pred.groupby("city_key").apply(
        lambda group: weighted_mean(
            group["canopy_pct_10pp"].to_numpy(float), group["population_model_weight"].to_numpy(float)
        ),
        include_groups=False,
    )
    pred["predicted"] = pred["raw_predicted"] + (
        pred["coefficient_C_per_10pp_canopy"] - pred["raw_coefficient_C_per_10pp_canopy"]
    ) * (pred["canopy_pct_10pp"] - pred["city_key"].map(canopy_mean))
    pred["residual"] = pred["observed"] - pred["predicted"]
    pred["model_id"] = model_id
    national = hierarchical_national_record(
        model_id,
        f"city-specific environmental GAM with {n_splines}x{n_splines} tensor smooth and REML partial pooling",
        MODEL2_COVARIATES,
        pool,
        len(pred),
        city["city_key"].nunique(),
    )
    national["n_splines_each_dimension"] = n_splines
    national["lambda"] = lam
    return HierarchicalFit(national, city, pred)


def fit_predict_spatial_gam(
    train: pd.DataFrame, test: pd.DataFrame, n_splines: int, lam: float
) -> np.ndarray:
    rows: list[dict[str, object]] = []
    fitted: dict[str, tuple[object, float, float]] = {}
    for city_key, sub in train.groupby("city_key", sort=True):
        gam, _, _, coefficient = fit_one_city_gam(sub, n_splines, lam)
        beta, se = map(float, coefficient)
        canopy_mean = weighted_mean(
            sub["canopy_pct_10pp"].to_numpy(float), sub["population_model_weight"].to_numpy(float)
        )
        rows.append({"city_key": city_key, "beta": beta, "se": se})
        fitted[city_key] = (gam, beta, canopy_mean)
    city = pd.DataFrame(rows)
    pool = reml_pool(city["beta"].to_numpy(float), city["se"].to_numpy(float))
    city["shrunk"] = np.asarray(pool["shrunk"], dtype=float)
    shrunk = city.set_index("city_key")["shrunk"].to_dict()
    prediction = pd.Series(np.nan, index=test.index, dtype=float)
    for city_key, sub in test.groupby("city_key", sort=True):
        gam, beta, canopy_mean = fitted[city_key]
        x = sub[[*GAM_LINEAR_COLUMNS, "x_city_0_1", "y_city_0_1"]].to_numpy(float)
        raw = gam.predict(x)
        prediction.loc[sub.index] = raw + (shrunk[city_key] - beta) * (
            sub["canopy_pct_10pp"].to_numpy(float) - canopy_mean
        )
    return prediction.loc[test.index].to_numpy(float)


def metric_record(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    model_id: str,
    outer_fold: int,
) -> dict[str, object]:
    valid = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(weights) & (weights > 0)
    y = observed[valid]
    p = predicted[valid]
    w = weights[valid]
    error = y - p
    mean = np.average(y, weights=w)
    return {
        "model_id": model_id,
        "outer_fold": outer_fold,
        "n": int(len(y)),
        "weighted_rmse_C": float(np.sqrt(np.average(error**2, weights=w))),
        "weighted_mae_C": float(np.average(np.abs(error), weights=w)),
        "weighted_r2": float(1 - np.sum(w * error**2) / np.sum(w * (y - mean) ** 2)),
    }


def spatial_folds(frame: pd.DataFrame) -> pd.Series:
    folds = pd.Series(-1, index=frame.index, dtype=np.int16)
    for city_key, index in frame.groupby("city_key", sort=True).groups.items():
        coords = frame.loc[index, ["x_5070_m", "y_5070_m"]].to_numpy(float)
        labels = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=30).fit_predict(coords)
        centers = np.asarray([coords[labels == value].mean(axis=0) for value in range(N_FOLDS)])
        order = np.lexsort((centers[:, 1], centers[:, 0]))
        relabel = {old_label: new_label for new_label, old_label in enumerate(order)}
        folds.loc[index] = np.asarray([relabel[value] for value in labels], dtype=np.int16)
    if (folds < 0).any():
        raise RuntimeError("Spatial fold assignment failed.")
    return folds


def pooled_metric_from_parts(parts: list[pd.DataFrame]) -> dict[str, float]:
    data = pd.concat(parts, ignore_index=True)
    y = data["observed"].to_numpy(float)
    p = data["predicted"].to_numpy(float)
    w = data["weight"].to_numpy(float)
    error = y - p
    mean = np.average(y, weights=w)
    return {
        "weighted_rmse_C": float(np.sqrt(np.average(error**2, weights=w))),
        "weighted_mae_C": float(np.average(np.abs(error), weights=w)),
        "weighted_r2": float(1 - np.sum(w * error**2) / np.sum(w * (y - mean) ** 2)),
    }


def run_nested_spatial_cv(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    work = frame.copy()
    work["spatial_fold"] = spatial_folds(work)
    work[["support_row", "analysis_record_id", "city_key", "city_display", "GEOID_str", "spatial_fold"]].to_csv(
        DATA_DIR / "nested_spatial_fold_assignment.csv", index=False, encoding="utf-8-sig"
    )

    inner_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    outer_metrics: list[dict[str, object]] = []
    outer_predictions: list[pd.DataFrame] = []

    for outer in range(N_FOLDS):
        print(f"[Nested CV outer {outer + 1}/{N_FOLDS}]", flush=True)
        outer_train_raw = work.loc[work["spatial_fold"].ne(outer)].copy()
        outer_test_raw = work.loc[work["spatial_fold"].eq(outer)].copy()
        candidate_results: list[dict[str, object]] = []
        for n_splines in N_SPLINES_GRID:
            for lam in LAMBDA_GRID:
                parts: list[pd.DataFrame] = []
                for inner in sorted(set(range(N_FOLDS)) - {outer}):
                    train_raw = outer_train_raw.loc[outer_train_raw["spatial_fold"].ne(inner)].copy()
                    valid_raw = outer_train_raw.loc[outer_train_raw["spatial_fold"].eq(inner)].copy()
                    train, valid = transform_train_test(train_raw, valid_raw)
                    prediction = fit_predict_spatial_gam(train, valid, n_splines, lam)
                    part = pd.DataFrame(
                        {
                            "observed": valid["heat_anomaly_C"].to_numpy(float),
                            "predicted": prediction,
                            "weight": valid["population_model_weight"].to_numpy(float),
                        }
                    )
                    parts.append(part)
                    fold_metric = pooled_metric_from_parts([part])
                    inner_rows.append(
                        {
                            "outer_fold": outer,
                            "inner_validation_fold": inner,
                            "n_splines_each_dimension": n_splines,
                            "lambda": lam,
                            "n": len(part),
                            **fold_metric,
                        }
                    )
                pooled = pooled_metric_from_parts(parts)
                candidate_results.append(
                    {
                        "outer_fold": outer,
                        "n_splines_each_dimension": n_splines,
                        "lambda": lam,
                        "n_inner_validation": int(sum(len(part) for part in parts)),
                        **pooled,
                    }
                )
                print(
                    f"  candidate n={n_splines}, lambda={lam:g}: inner RMSE={pooled['weighted_rmse_C']:.4f}",
                    flush=True,
                )

        candidates = pd.DataFrame(candidate_results).sort_values(
            ["weighted_rmse_C", "n_splines_each_dimension", "lambda"]
        )
        selected = candidates.iloc[0].to_dict()
        selected["selection_rule"] = "minimum pooled inner-fold population-weighted RMSE; ties favour fewer bases"
        selection_rows.extend(candidates.to_dict("records"))

        outer_train, outer_test = transform_train_test(outer_train_raw, outer_test_raw)
        predictions = {
            "M1_common_FE_density": fit_predict_common_fe(outer_train, outer_test, MODEL1_COVARIATES),
            "M2_common_FE_environment": fit_predict_common_fe(outer_train, outer_test, MODEL2_COVARIATES),
            "M2_city_specific_REML": fit_predict_city_hierarchical_linear(outer_train, outer_test),
            "M3_spatial_GAM_REML": fit_predict_spatial_gam(
                outer_train,
                outer_test,
                int(selected["n_splines_each_dimension"]),
                float(selected["lambda"]),
            ),
        }
        for model_id, prediction in predictions.items():
            outer_metrics.append(
                metric_record(
                    outer_test["heat_anomaly_C"].to_numpy(float),
                    prediction,
                    outer_test["population_model_weight"].to_numpy(float),
                    model_id,
                    outer,
                )
            )
            pred = outer_test[["support_row", "analysis_record_id", "city_key", "city_display", "GEOID_str"]].copy()
            pred["observed"] = outer_test["heat_anomaly_C"].to_numpy(float)
            pred["predicted"] = prediction
            pred["residual"] = pred["observed"] - pred["predicted"]
            pred["population_model_weight"] = outer_test["population_model_weight"].to_numpy(float)
            pred["model_id"] = model_id
            pred["outer_fold"] = outer
            pred["selected_n_splines_each_dimension"] = (
                int(selected["n_splines_each_dimension"]) if model_id == "M3_spatial_GAM_REML" else np.nan
            )
            pred["selected_lambda"] = float(selected["lambda"]) if model_id == "M3_spatial_GAM_REML" else np.nan
            outer_predictions.append(pred)

    inner = pd.DataFrame(inner_rows)
    candidate = pd.DataFrame(selection_rows)
    metrics = pd.DataFrame(outer_metrics)
    predictions = pd.concat(outer_predictions, ignore_index=True)
    inner.to_csv(DIAG_DIR / "nested_cv_inner_fold_metrics.csv", index=False, encoding="utf-8-sig")
    candidate.to_csv(TABLE_DIR / "nested_cv_candidate_summary_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(DIAG_DIR / "nested_cv_outer_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(DIAG_DIR / "nested_cv_outer_predictions.csv", index=False, encoding="utf-8-sig")

    pooled_rows: list[dict[str, object]] = []
    for model_id, sub in predictions.groupby("model_id", sort=False):
        pooled = pooled_metric_from_parts(
            [
                pd.DataFrame(
                    {
                        "observed": sub["observed"],
                        "predicted": sub["predicted"],
                        "weight": sub["population_model_weight"],
                    }
                )
            ]
        )
        pooled_rows.append({"model_id": model_id, "n": len(sub), **pooled})
    pooled_table = pd.DataFrame(pooled_rows)
    pooled_table.to_csv(TABLE_DIR / "nested_spatial_cv_outer_performance.csv", index=False, encoding="utf-8-sig")

    selected_each_outer = (
        candidate.sort_values(["outer_fold", "weighted_rmse_C", "n_splines_each_dimension", "lambda"])
        .groupby("outer_fold", as_index=False)
        .first()
    )
    final_choice = (
        candidate.groupby(["n_splines_each_dimension", "lambda"], as_index=False)
        .agg(mean_outer_inner_rmse_C=("weighted_rmse_C", "mean"), selected_in_outer_folds=("outer_fold", "nunique"))
        .sort_values(["mean_outer_inner_rmse_C", "n_splines_each_dimension", "lambda"])
        .iloc[0]
    )
    selected_each_outer.to_csv(TABLE_DIR / "nested_cv_selected_candidate_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    final = {
        "n_splines_each_dimension": int(final_choice["n_splines_each_dimension"]),
        "lambda": float(final_choice["lambda"]),
        "selection_rule": "minimum mean inner-CV RMSE across the five outer-training partitions; outer test folds never enter tuning",
        "mean_outer_inner_rmse_C": float(final_choice["mean_outer_inner_rmse_C"]),
    }
    write_json(QA_DIR / "nested_spatial_cv_selection.json", final)
    return work, final


def moran_i(residual: np.ndarray, coords: np.ndarray, permutations: int, seed: int) -> tuple[float, float]:
    valid = np.isfinite(residual) & np.isfinite(coords).all(axis=1)
    residual = residual[valid]
    coords = coords[valid]
    if len(residual) <= N_NEIGHBOURS + 2:
        return np.nan, np.nan
    neighbours = NearestNeighbors(n_neighbors=N_NEIGHBOURS + 1).fit(coords).kneighbors(return_distance=False)[:, 1:]
    z = residual - residual.mean()
    denominator = float(np.sum(z**2))
    observed = float(np.sum(z * z[neighbours].mean(axis=1)) / denominator)
    rng = np.random.default_rng(seed)
    permuted = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = rng.permutation(z)
        permuted[index] = float(np.sum(shuffled * shuffled[neighbours].mean(axis=1)) / denominator)
    p = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (permutations + 1))
    return observed, p


def residual_diagnostics(frame: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model_id, model_pred in predictions.groupby("model_id", sort=False):
        numerator = 0.0
        denominator = 0.0
        values: list[float] = []
        for city_number, (city_key, sub) in enumerate(model_pred.groupby("city_key", sort=True)):
            index = sub["support_row"].to_numpy(int)
            residual = sub["residual"].to_numpy(float)
            coords = frame.iloc[index][["x_5070_m", "y_5070_m"]].to_numpy(float)
            value, p = moran_i(residual, coords, MORAN_PERMUTATIONS, SEED + city_number)
            centered = residual - residual.mean()
            neighbours = NearestNeighbors(n_neighbors=N_NEIGHBOURS + 1).fit(coords).kneighbors(return_distance=False)[:, 1:]
            city_num = float(np.sum(centered * centered[neighbours].mean(axis=1)))
            city_den = float(np.sum(centered**2))
            numerator += city_num
            denominator += city_den
            values.append(value)
            rows.append(
                {
                    "model_id": model_id,
                    "city_key": city_key,
                    "city_display": sub["city_display"].iloc[0],
                    "n_block_groups": len(sub),
                    "moran_i_8nn": value,
                    "permutation_p_value": p,
                }
            )
        summary_rows.append(
            {
                "model_id": model_id,
                "pooled_block_diagonal_moran_i_8nn": numerator / denominator,
                "median_city_moran_i_8nn": float(np.nanmedian(values)),
                "minimum_city_moran_i_8nn": float(np.nanmin(values)),
                "maximum_city_moran_i_8nn": float(np.nanmax(values)),
                "cities_with_permutation_p_lt_0p05": int(
                    sum(row["permutation_p_value"] < 0.05 for row in rows if row["model_id"] == model_id)
                ),
            }
        )
    city = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    city.to_csv(DIAG_DIR / "residual_moran_by_city.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "residual_moran_summary.csv", index=False, encoding="utf-8-sig")
    return city, summary


def run_primary_models(frame: pd.DataFrame, final: dict[str, object]) -> None:
    national_rows: list[dict[str, object]] = []
    city_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []

    for model_id, covariates in (
        ("M1_common_FE_density", MODEL1_COVARIATES),
        ("M2_common_FE_environment", MODEL2_COVARIATES),
    ):
        national, prediction = fit_common_fe(frame, model_id, covariates)
        national_rows.append(national)
        prediction_tables.append(prediction)

    m2_city = fit_city_hierarchical_linear(frame)
    national_rows.append(m2_city.national)
    city_tables.append(m2_city.city)
    prediction_tables.append(m2_city.predictions)

    m3 = fit_hierarchical_spatial_gam(
        frame,
        int(final["n_splines_each_dimension"]),
        float(final["lambda"]),
    )
    national_rows.append(m3.national)
    city_tables.append(m3.city)
    prediction_tables.append(m3.predictions)

    national = pd.DataFrame(national_rows)
    cities = pd.concat(city_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    national.to_csv(TABLE_DIR / "national_canopy_lst_model_comparison.csv", index=False, encoding="utf-8-sig")
    cities.to_csv(TABLE_DIR / "city_canopy_lst_coefficients_hierarchical_models.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(DIAG_DIR / "in_sample_predictions_and_residuals.csv", index=False, encoding="utf-8-sig")
    residual_diagnostics(frame, predictions)


def run_support_sensitivity(final: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    frames = [
        ("fully_contained_primary", DATA_DIR / "support_aligned_model_frame_34566.csv"),
        ("intersection_ge_0p95", DATA_DIR / "intersection_ge_0p95_model_frame.csv"),
        ("intersection_ge_0p80", DATA_DIR / "intersection_ge_0p80_model_frame.csv"),
    ]
    for label, path in frames:
        print(f"[Support sensitivity] {label}", flush=True)
        frame = pd.read_csv(path, dtype={"GEOID_str": str}, low_memory=False)
        if "support_row" not in frame:
            frame["support_row"] = np.arange(len(frame), dtype=np.int32)
        for model_id, covariates in (
            ("M1_common_FE_density", MODEL1_COVARIATES),
            ("M2_common_FE_environment", MODEL2_COVARIATES),
        ):
            record, _ = fit_common_fe(frame, model_id, covariates)
            rows.append({"support_frame": label, **record})
        m2 = fit_city_hierarchical_linear(frame)
        rows.append({"support_frame": label, **m2.national})
        m3 = fit_hierarchical_spatial_gam(
            frame,
            int(final["n_splines_each_dimension"]),
            float(final["lambda"]),
        )
        rows.append({"support_frame": label, **m3.national})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "support_threshold_model_sensitivity.csv", index=False, encoding="utf-8-sig")


def write_report() -> None:
    national = pd.read_csv(TABLE_DIR / "national_canopy_lst_model_comparison.csv")
    cv = pd.read_csv(TABLE_DIR / "nested_spatial_cv_outer_performance.csv")
    moran = pd.read_csv(TABLE_DIR / "residual_moran_summary.csv")
    sensitivity = pd.read_csv(TABLE_DIR / "support_threshold_model_sensitivity.csv")
    n = national.set_index("model_id")
    m = moran.set_index("model_id")
    lines = [
        "# Support-aligned Result 3 audit",
        "",
        "## Frozen analytical support",
        "",
        "The primary analysis contains 34,566 fully contained census block groups in all 50 cities. Canopy, LST, population, land area and covariates therefore refer to the same census-unit support. The legacy 38,447-row frame is not used for primary inference.",
        "",
        "## Unified model sequence",
        "",
    ]
    for model_id, row in n.iterrows():
        lines.append(
            f"- {model_id}: {row['coefficient_C_per_10pp_canopy']:.3f} C per 10 percentage points "
            f"(95% interval {row['ci_low']:.3f} to {row['ci_high']:.3f})."
        )
    lines.extend(["", "## Nested spatial cross-validation", ""])
    for _, row in cv.iterrows():
        lines.append(
            f"- {row['model_id']}: outer-fold weighted RMSE {row['weighted_rmse_C']:.3f} C; "
            f"weighted R2 {row['weighted_r2']:.3f}."
        )
    lines.extend(["", "## Residual spatial dependence", ""])
    for model_id, row in m.iterrows():
        lines.append(
            f"- {model_id}: pooled 8-NN Moran's I {row['pooled_block_diagonal_moran_i_8nn']:.3f}; "
            f"median city Moran's I {row['median_city_moran_i_8nn']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "All estimates remain cross-sectional associations with Landsat land-surface temperature. Spatial adjustment, nested validation and support alignment improve inference but do not identify a causal cooling effect, air-temperature response, personal exposure or health effect.",
            "",
            "## Boundary sensitivity",
            "",
            sensitivity.to_markdown(index=False),
        ]
    )
    (TEXT_DIR / "result3_support_aligned_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Support-aligned Result 3 reanalysis.")
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "cv", "models", "sensitivity", "report"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    if args.stage in ("all", "prepare"):
        build_support_frame()
        if args.stage == "prepare":
            return 0
    frame = load_primary()
    if args.stage in ("all", "cv"):
        frame, final = run_nested_spatial_cv(frame)
        frame.to_csv(DATA_DIR / "support_aligned_model_frame_34566.csv", index=False, encoding="utf-8-sig")
        if args.stage == "cv":
            return 0
    else:
        final = json.loads((QA_DIR / "nested_spatial_cv_selection.json").read_text(encoding="utf-8"))
    if args.stage in ("all", "models"):
        run_primary_models(frame, final)
    if args.stage in ("all", "sensitivity"):
        run_support_sensitivity(final)
    if args.stage in ("all", "report"):
        write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
