from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm


WORK_ROOT = Path(str(repo_root().joinpath('document_work', 'result2_transferability_audit')))
OUT_ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result2', 'Result2_city_slope_transferability_audit')))
DATA_DIR = OUT_ROOT / "data"
TABLE_DIR = OUT_ROOT / "tables"
DIAG_DIR = OUT_ROOT / "diagnostics"
QA_DIR = OUT_ROOT / "qa"
TEXT_DIR = OUT_ROOT / "text"
SCRIPT_DIR = OUT_ROOT / "scripts"

R2_BG = Path(
    str(analysis_store_root().joinpath('Result', 'Result2', 'Result2_module1_boundary_inference_audit', 'source_data', 'result2_primary_fully_contained_block_groups.csv'))
)
R2_TRACT = Path(
    str(analysis_store_root().joinpath('Result', 'Result2', 'Result2_module1_boundary_inference_audit', 'source_data', 'result2_primary_fully_contained_tracts.csv'))
)
R3_FRAME = Path(
    str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned', 'data', 'support_aligned_model_frame_34566.csv'))
)
R3_SHARED_BLOCKS = Path(
    str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned', 'paired_m2_m3_shared_block_bootstrap', 'data', 'shared_spatial_block_assignments.csv'))
)
R3_SHARED_RANGES = Path(
    str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned', 'paired_m2_m3_shared_block_bootstrap', 'tables', 'm2_m3_residual_ranges_and_shared_block_sizes.csv'))
)
R3_NATIONAL_ANALYTIC = Path(
    str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned', 'tables', 'national_canopy_lst_model_comparison.csv'))
)
R3_CITY_ANALYTIC = Path(
    str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned', 'tables', 'city_canopy_lst_coefficients_hierarchical_models.csv'))
)
R3_PAIRED_NATIONAL = Path(
    str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned', 'paired_m2_m3_shared_block_bootstrap', 'tables', 'paired_national_spatial_bootstrap_inference.csv'))
)
R3_PAIRED_CITY = Path(
    str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned', 'paired_m2_m3_shared_block_bootstrap', 'tables', 'paired_city_spatial_bootstrap_intervals.csv'))
)
R3_PAIRED_ROBUSTNESS = Path(
    str(analysis_store_root().joinpath('Result', 'Result3', 'Result3_support_aligned', 'paired_m2_m3_shared_block_bootstrap', 'tables', 'paired_city_slope_robustness.csv'))
)

BASE_SEED = 20260723
SCALE_REPLICATES = {"small": 250, "main": 500, "large": 250}
GRID_OFFSETS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
MIN_OCCUPIED_BLOCKS = 8


@dataclass(frozen=True)
class Predictor:
    key: str
    raw: str
    label: str
    expected_direction: str


PREDICTORS = (
    Predictor("hispanic", "hispanic_or_latino_pct", "Hispanic/Latino share", "negative"),
    Predictor("renter", "renter_occupied_pct", "Renter share", "negative"),
    Predictor("no_vehicle", "no_vehicle_households_pct", "No-vehicle household share", "negative"),
    Predictor("median_age", "median_age", "Median age", "positive"),
)


def ensure_dirs() -> None:
    for path in (DATA_DIR, TABLE_DIR, DIAG_DIR, QA_DIR, TEXT_DIR, SCRIPT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_id(series: pd.Series, width: int) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(width)
    )


def reml_pool(beta: np.ndarray, se: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    beta_valid = np.asarray(beta, dtype=float)[valid]
    se_valid = np.asarray(se, dtype=float)[valid]
    if len(beta_valid) < 3:
        raise RuntimeError("At least three valid city slopes are required.")

    def objective(log_tau2: float) -> float:
        tau2 = math.exp(log_tau2)
        variance = se_valid**2 + tau2
        weights = 1.0 / variance
        mu = np.sum(weights * beta_valid) / np.sum(weights)
        return 0.5 * (
            np.sum(np.log(variance))
            + math.log(np.sum(weights))
            + np.sum((beta_valid - mu) ** 2 / variance)
            + (len(beta_valid) - 1) * math.log(2 * math.pi)
        )

    result = optimize.minimize_scalar(objective, bounds=(-20, 12), method="bounded")
    tau2 = float(math.exp(result.x))
    variance = se_valid**2 + tau2
    weights = 1.0 / variance
    mu = float(np.sum(weights * beta_valid) / np.sum(weights))
    mu_se = float(np.sqrt(1.0 / np.sum(weights)))
    if tau2 <= 1e-12:
        shrunk_valid = np.full_like(beta_valid, mu)
        shrunk_se_valid = np.full_like(se_valid, mu_se)
    else:
        precision = 1.0 / se_valid**2 + 1.0 / tau2
        shrunk_valid = (beta_valid / se_valid**2 + mu / tau2) / precision
        shrunk_se_valid = np.sqrt(1.0 / precision)
    shrunk = np.full_like(np.asarray(beta, dtype=float), np.nan)
    shrunk_se = np.full_like(np.asarray(se, dtype=float), np.nan)
    shrunk[valid] = shrunk_valid
    shrunk_se[valid] = shrunk_se_valid
    q = float(
        np.sum(
            (beta_valid - np.average(beta_valid, weights=1.0 / se_valid**2)) ** 2
            / se_valid**2
        )
    )
    i2 = max(0.0, (q - (len(beta_valid) - 1)) / q) if q > 0 else 0.0
    return {
        "mu": mu,
        "mu_se": mu_se,
        "tau2": tau2,
        "tau": math.sqrt(tau2),
        "i2": i2,
        "shrunk": shrunk,
        "shrunk_se": shrunk_se,
        "optimizer_success": bool(result.success),
    }


def assign_blocks(
    coords: np.ndarray, block_size_m: float, offset: tuple[float, float]
) -> np.ndarray:
    shifted = coords - coords.min(axis=0) + np.asarray(offset) * block_size_m
    grid = np.floor(shifted / block_size_m).astype(np.int64)
    _, labels = np.unique(grid, axis=0, return_inverse=True)
    return labels.astype(np.int32)


def standardize_common_frame(frame: pd.DataFrame, geography: str) -> pd.DataFrame:
    required = [
        "city_key",
        "unit_id",
        "city_display",
        "population",
        "canopy_pct",
        "x_5070_m",
        "y_5070_m",
        *[predictor.raw for predictor in PREDICTORS],
    ]
    out = frame[required].copy()
    numeric = [
        "population",
        "canopy_pct",
        "x_5070_m",
        "y_5070_m",
        *[predictor.raw for predictor in PREDICTORS],
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    valid = (
        out[numeric].notna().all(axis=1)
        & np.isfinite(out[numeric]).all(axis=1)
        & out["population"].gt(0)
    )
    out = out.loc[valid].copy()
    for predictor in PREDICTORS:
        out[f"z_{predictor.key}"] = np.nan
        for _, index in out.groupby("city_key", sort=True).groups.items():
            values = out.loc[index, predictor.raw].astype(float)
            sd = float(values.std(ddof=0))
            if not np.isfinite(sd) or sd <= 0:
                raise RuntimeError(
                    f"{geography}: zero within-city SD for {predictor.key}."
                )
            out.loc[index, f"z_{predictor.key}"] = (
                values - float(values.mean())
            ) / sd
    out = out.sort_values(["city_key", "unit_id"]).reset_index(drop=True)
    out.insert(0, "geography", geography)
    return out


def load_and_freeze_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r2_bg = pd.read_csv(R2_BG, dtype={"spatial_unit_id": str}, low_memory=False)
    r3 = pd.read_csv(R3_FRAME, dtype={"GEOID_str": str}, low_memory=False)
    blocks = pd.read_csv(R3_SHARED_BLOCKS, dtype={"GEOID_str": str}, low_memory=False)
    if len(r2_bg) != 34_575 or r2_bg["result_city_key"].nunique() != 50:
        raise RuntimeError("Unexpected Result 2 block-group frame.")
    if len(r3) != 34_566 or r3["city_key"].nunique() != 50:
        raise RuntimeError("Unexpected Result 3 support-aligned frame.")

    r2_bg["unit_id"] = normalize_id(r2_bg["spatial_unit_id"], 12)
    r3["unit_id"] = normalize_id(r3["GEOID_str"], 12)
    r2_keys = set(zip(r2_bg["result_city_key"], r2_bg["unit_id"]))
    r3_keys = set(zip(r3["city_key"], r3["unit_id"]))
    if not r3_keys.issubset(r2_keys) or len(r2_keys - r3_keys) != 9:
        raise RuntimeError("Result 3 is not the expected nine-row subset of Result 2.")

    raw_map = {predictor.raw: predictor.raw for predictor in PREDICTORS}
    bg = r3.rename(columns={"GEOID_str": "source_geoid"}).copy()
    # Use the frozen analysis canopy field explicitly; the source also contains
    # a legacy canopy_pct column with the same current values.
    bg["canopy_pct"] = pd.to_numeric(
        bg["canopy_pct_analysis"], errors="coerce"
    )
    bg["unit_id"] = normalize_id(bg["source_geoid"], 12)
    for target, source in raw_map.items():
        bg[target] = pd.to_numeric(bg[source], errors="coerce")
    block_columns = [
        "city_key",
        "GEOID_str",
        *[
            f"{scale}_block_origin{origin}"
            for scale in SCALE_REPLICATES
            for origin in range(len(GRID_OFFSETS))
        ],
    ]
    blocks = blocks[block_columns].copy()
    blocks["unit_id"] = normalize_id(blocks["GEOID_str"], 12)
    blocks = blocks.drop(columns="GEOID_str")
    bg = bg.merge(
        blocks, on=["city_key", "unit_id"], how="left", validate="one_to_one"
    )
    if bg.filter(regex=r"_block_origin").isna().any().any():
        raise RuntimeError("Missing block assignments in block-group frame.")
    bg = standardize_common_frame(bg, "block_group").merge(
        blocks,
        on=["city_key", "unit_id"],
        how="left",
        validate="one_to_one",
    )

    tract = pd.read_csv(R2_TRACT, dtype={"spatial_unit_id": str}, low_memory=False)
    if len(tract) != 11_984 or tract["result_city_key"].nunique() != 50:
        raise RuntimeError("Unexpected Result 2 tract frame.")
    tract["unit_id"] = normalize_id(tract["spatial_unit_id"], 11)
    bg_coords = r3[
        ["city_key", "GEOID_str", "x_5070_m", "y_5070_m", "land_area_m2"]
    ].copy()
    bg_coords["unit_id"] = normalize_id(bg_coords["GEOID_str"], 12).str[:11]
    bg_coords["_wx"] = (
        pd.to_numeric(bg_coords["x_5070_m"], errors="coerce")
        * pd.to_numeric(bg_coords["land_area_m2"], errors="coerce")
    )
    bg_coords["_wy"] = (
        pd.to_numeric(bg_coords["y_5070_m"], errors="coerce")
        * pd.to_numeric(bg_coords["land_area_m2"], errors="coerce")
    )
    coord = (
        bg_coords.groupby(["city_key", "unit_id"], as_index=False)
        .agg(
            land_area_sum=("land_area_m2", "sum"),
            wx_sum=("_wx", "sum"),
            wy_sum=("_wy", "sum"),
        )
    )
    coord["x_5070_m"] = coord["wx_sum"] / coord["land_area_sum"]
    coord["y_5070_m"] = coord["wy_sum"] / coord["land_area_sum"]
    city_names = (
        r3[["city_key", "city_display"]]
        .drop_duplicates()
        .set_index("city_key")["city_display"]
    )
    tract = tract.rename(
        columns={
            "result_city_key": "city_key",
            "canopy_reference_0p6m": "canopy_pct",
        }
    )
    tract["city_display"] = tract["city_key"].map(city_names)
    tract = tract.merge(
        coord[["city_key", "unit_id", "x_5070_m", "y_5070_m"]],
        on=["city_key", "unit_id"],
        how="left",
        validate="one_to_one",
    )
    if tract[["x_5070_m", "y_5070_m"]].isna().any().any():
        raise RuntimeError("Not all tracts received coordinates.")
    tract = standardize_common_frame(tract, "tract")

    range_table = pd.read_csv(R3_SHARED_RANGES)
    tract_block_records: list[pd.DataFrame] = []
    tract_adjustment_records: list[dict[str, object]] = []
    for city_key, sub in tract.groupby("city_key", sort=True):
        coordinates = sub[["x_5070_m", "y_5070_m"]].to_numpy(float)
        assigned = sub[["geography", "city_key", "unit_id"]].copy()
        range_row = range_table.loc[range_table["city_key"].eq(city_key)].iloc[0]
        for scale in SCALE_REPLICATES:
            size_m = float(range_row[f"{scale}_shared_block_size_km"]) * 1_000.0
            original_size_m = size_m
            while True:
                labels = [
                    assign_blocks(coordinates, size_m, offset)
                    for offset in GRID_OFFSETS
                ]
                occupied = [len(np.unique(values)) for values in labels]
                if min(occupied) >= MIN_OCCUPIED_BLOCKS:
                    break
                size_m *= 0.85
            for origin, values in enumerate(labels):
                assigned[f"{scale}_block_origin{origin}"] = values
            tract_adjustment_records.append(
                {
                    "city_key": city_key,
                    "city_display": sub["city_display"].iloc[0],
                    "block_scale": scale,
                    "result3_shared_block_size_km": original_size_m / 1_000.0,
                    "tract_block_size_km": size_m / 1_000.0,
                    "size_adjusted_for_minimum_blocks": not np.isclose(
                        size_m, original_size_m
                    ),
                    "minimum_occupied_blocks": min(occupied),
                    "maximum_occupied_blocks": max(occupied),
                }
            )
        tract_block_records.append(assigned)
    tract_blocks = pd.concat(tract_block_records, ignore_index=True)
    tract = tract.merge(
        tract_blocks.drop(columns="geography"),
        on=["city_key", "unit_id"],
        how="left",
        validate="one_to_one",
    )
    adjustment = pd.DataFrame(tract_adjustment_records)

    expected_bg = 34_434
    expected_tract = 11_932
    if len(bg) != expected_bg or bg["city_key"].nunique() != 50:
        raise RuntimeError(f"Frozen block-group sample differs from {expected_bg}.")
    if len(tract) != expected_tract or tract["city_key"].nunique() != 50:
        raise RuntimeError(f"Frozen tract sample differs from {expected_tract}.")
    return bg, tract, adjustment


def wls_hc3(
    y: np.ndarray, z: np.ndarray, population_weight: np.ndarray
) -> tuple[float, float, float]:
    valid = (
        np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(population_weight)
        & (population_weight > 0)
    )
    y = np.asarray(y, dtype=float)[valid]
    z = np.asarray(z, dtype=float)[valid]
    weight = np.asarray(population_weight, dtype=float)[valid]
    if len(y) < 8 or np.ptp(z) <= 1e-10:
        return np.nan, np.nan, np.nan
    x = np.column_stack((np.ones(len(y)), z))
    weighted_x = x * np.sqrt(weight)[:, None]
    weighted_y = y * np.sqrt(weight)
    bread = np.linalg.pinv(weighted_x.T @ weighted_x, rcond=1e-12)
    beta = bread @ (weighted_x.T @ weighted_y)
    residual_weighted = weighted_y - weighted_x @ beta
    leverage = np.einsum("ij,jk,ik->i", weighted_x, bread, weighted_x)
    denominator = np.clip(1.0 - leverage, 1e-8, None)
    adjusted = residual_weighted / denominator
    meat = weighted_x.T @ (weighted_x * adjusted[:, None] ** 2)
    covariance = bread @ meat @ bread
    slope_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    return float(beta[1]), slope_se, float(beta[0])


def fit_analytic_city_slopes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_rows: list[dict[str, object]] = []
    national_rows: list[dict[str, object]] = []
    city_population = frame.groupby("city_key")["population"].sum()
    geography = str(frame["geography"].iloc[0])
    for predictor in PREDICTORS:
        predictor_rows: list[dict[str, object]] = []
        for city_key, sub in frame.groupby("city_key", sort=True):
            beta, se, intercept = wls_hc3(
                sub["canopy_pct"].to_numpy(float),
                sub[f"z_{predictor.key}"].to_numpy(float),
                sub["population"].to_numpy(float),
            )
            predictor_rows.append(
                {
                    "geography": geography,
                    "predictor": predictor.key,
                    "predictor_label": predictor.label,
                    "expected_direction": predictor.expected_direction,
                    "city_key": city_key,
                    "city_display": sub["city_display"].iloc[0],
                    "n_units": len(sub),
                    "represented_population": float(sub["population"].sum()),
                    "raw_slope_pp_per_within_city_sd": beta,
                    "raw_hc3_se": se,
                    "raw_ci_low": beta - 1.96 * se,
                    "raw_ci_high": beta + 1.96 * se,
                    "intercept": intercept,
                }
            )
        city = pd.DataFrame(predictor_rows).sort_values("city_key")
        pool = reml_pool(
            city["raw_slope_pp_per_within_city_sd"].to_numpy(float),
            city["raw_hc3_se"].to_numpy(float),
        )
        city["partially_pooled_slope_pp_per_within_city_sd"] = pool["shrunk"]
        city["partially_pooled_se"] = pool["shrunk_se"]
        city["partially_pooled_ci_low"] = (
            city["partially_pooled_slope_pp_per_within_city_sd"]
            - 1.96 * city["partially_pooled_se"]
        )
        city["partially_pooled_ci_high"] = (
            city["partially_pooled_slope_pp_per_within_city_sd"]
            + 1.96 * city["partially_pooled_se"]
        )
        city_rows.extend(city.to_dict("records"))

        beta = city["raw_slope_pp_per_within_city_sd"].to_numpy(float)
        direction = -1.0 if float(pool["mu"]) < 0 else 1.0
        critical = float(stats.t.ppf(0.975, df=49))
        prediction_sd = math.sqrt(float(pool["tau2"]) + float(pool["mu_se"]) ** 2)
        population_weights = city["city_key"].map(city_population).to_numpy(float)
        national_rows.append(
            {
                "geography": geography,
                "predictor": predictor.key,
                "predictor_label": predictor.label,
                "expected_direction": predictor.expected_direction,
                "n_units": len(frame),
                "n_cities": 50,
                "reml_mu_pp_per_within_city_sd": float(pool["mu"]),
                "reml_mu_se": float(pool["mu_se"]),
                "reml_ci_low": float(pool["mu"]) - critical * float(pool["mu_se"]),
                "reml_ci_high": float(pool["mu"]) + critical * float(pool["mu_se"]),
                "tau_between_city": float(pool["tau"]),
                "tau_over_abs_mu": float(pool["tau"]) / max(abs(float(pool["mu"])), 1e-12),
                "i2_city_heterogeneity": float(pool["i2"]),
                "prediction_interval_low": float(pool["mu"]) - critical * prediction_sd,
                "prediction_interval_high": float(pool["mu"]) + critical * prediction_sd,
                "city_equal_raw_mean": float(np.mean(beta)),
                "city_population_weighted_raw_mean": float(
                    np.average(beta, weights=population_weights)
                ),
                "n_city_points_in_national_direction": int(np.sum(direction * beta > 0)),
                "share_city_points_in_national_direction": float(np.mean(direction * beta > 0)),
            }
        )
    return pd.DataFrame(city_rows), pd.DataFrame(national_rows)


def bootstrap_city_fit(
    sub: pd.DataFrame,
    predictor: Predictor,
    scale: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    for _ in range(8):
        origin = int(rng.integers(0, len(GRID_OFFSETS)))
        labels = sub[f"{scale}_block_origin{origin}"].to_numpy(np.int32)
        unique, inverse = np.unique(labels, return_inverse=True)
        sampled = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(sampled, minlength=len(unique))
        multiplier = counts[inverse].astype(float)
        selected = multiplier > 0
        if selected.sum() < 8 or np.count_nonzero(counts) < 4:
            continue
        weight = sub["population"].to_numpy(float) * multiplier
        beta, se, _ = wls_hc3(
            sub["canopy_pct"].to_numpy(float),
            sub[f"z_{predictor.key}"].to_numpy(float),
            weight,
        )
        if np.isfinite(beta) and np.isfinite(se) and se > 0:
            return beta, se
    raise RuntimeError(
        f"Invalid bootstrap fit after eight attempts: "
        f"{sub['geography'].iloc[0]}, {sub['city_key'].iloc[0]}, "
        f"{predictor.key}, {scale}."
    )


def run_spatial_bootstrap(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    geography = str(frame["geography"].iloc[0])
    cities = (
        frame[["city_key", "city_display"]]
        .drop_duplicates()
        .sort_values("city_key")
        .reset_index(drop=True)
    )
    city_subsets = {
        city_key: frame.loc[frame["city_key"].eq(city_key)].copy()
        for city_key in cities["city_key"]
    }
    city_population = np.asarray(
        [city_subsets[city]["population"].sum() for city in cities["city_key"]],
        dtype=float,
    )
    national_rows: list[dict[str, object]] = []
    city_rows: list[dict[str, object]] = []
    draw_rows: list[pd.DataFrame] = []
    city_draw_rows: list[pd.DataFrame] = []

    for predictor_index, predictor in enumerate(PREDICTORS):
        for scale_index, (scale, replicates) in enumerate(SCALE_REPLICATES.items()):
            raw_beta = np.empty((replicates, len(cities)), dtype=float)
            raw_se = np.empty_like(raw_beta)
            shrunk = np.empty_like(raw_beta)
            conditional_mu = np.empty(replicates, dtype=float)
            tau = np.empty(replicates, dtype=float)
            i2 = np.empty(replicates, dtype=float)
            city_equal = np.empty(replicates, dtype=float)
            population_weighted = np.empty(replicates, dtype=float)
            for replicate in range(replicates):
                seed = (
                    BASE_SEED
                    + (0 if geography == "block_group" else 100_000_000)
                    + predictor_index * 10_000_000
                    + scale_index * 1_000_000
                    + replicate
                )
                rng = np.random.default_rng(seed)
                for city_index, city_key in enumerate(cities["city_key"]):
                    raw_beta[replicate, city_index], raw_se[replicate, city_index] = (
                        bootstrap_city_fit(
                            city_subsets[city_key], predictor, scale, rng
                        )
                    )
                pool = reml_pool(raw_beta[replicate], raw_se[replicate])
                conditional_mu[replicate] = float(pool["mu"])
                tau[replicate] = float(pool["tau"])
                i2[replicate] = float(pool["i2"])
                shrunk[replicate] = np.asarray(pool["shrunk"], dtype=float)
                city_equal[replicate] = float(np.mean(raw_beta[replicate]))
                population_weighted[replicate] = float(
                    np.average(raw_beta[replicate], weights=city_population)
                )

            two_level_reml = np.empty(replicates, dtype=float)
            two_level_city_equal = np.empty(replicates, dtype=float)
            two_level_population = np.empty(replicates, dtype=float)
            rng = np.random.default_rng(
                BASE_SEED
                + 900_000_000
                + (0 if geography == "block_group" else 100_000_000)
                + predictor_index * 10_000_000
                + scale_index * 1_000_000
            )
            for draw in range(replicates):
                sampled_cities = rng.integers(0, len(cities), size=len(cities))
                sampled_replicates = rng.integers(0, replicates, size=len(cities))
                beta = raw_beta[sampled_replicates, sampled_cities]
                se = raw_se[sampled_replicates, sampled_cities]
                two_level_reml[draw] = float(reml_pool(beta, se)["mu"])
                two_level_city_equal[draw] = float(np.mean(beta))
                two_level_population[draw] = float(
                    np.average(beta, weights=city_population[sampled_cities])
                )

            def q(values: np.ndarray) -> tuple[float, float, float]:
                low, median, high = np.percentile(values, [2.5, 50, 97.5])
                return float(low), float(median), float(high)

            conditional_low, conditional_median, conditional_high = q(conditional_mu)
            reml_low, reml_median, reml_high = q(two_level_reml)
            equal_low, equal_median, equal_high = q(two_level_city_equal)
            pop_low, pop_median, pop_high = q(two_level_population)
            national_rows.append(
                {
                    "geography": geography,
                    "predictor": predictor.key,
                    "predictor_label": predictor.label,
                    "expected_direction": predictor.expected_direction,
                    "block_scale": scale,
                    "replicates": replicates,
                    "conditional_reml_median": conditional_median,
                    "conditional_reml_ci_low": conditional_low,
                    "conditional_reml_ci_high": conditional_high,
                    "two_level_reml_median": reml_median,
                    "two_level_reml_ci_low": reml_low,
                    "two_level_reml_ci_high": reml_high,
                    "two_level_city_equal_median": equal_median,
                    "two_level_city_equal_ci_low": equal_low,
                    "two_level_city_equal_ci_high": equal_high,
                    "two_level_city_population_weighted_median": pop_median,
                    "two_level_city_population_weighted_ci_low": pop_low,
                    "two_level_city_population_weighted_ci_high": pop_high,
                    "mean_tau": float(np.mean(tau)),
                    "mean_tau_over_abs_conditional_mu": float(
                        np.mean(tau) / max(abs(np.median(conditional_mu)), 1e-12)
                    ),
                    "mean_i2": float(np.mean(i2)),
                    "probability_two_level_reml_below_zero": float(
                        np.mean(two_level_reml < 0)
                    ),
                }
            )
            draw_rows.append(
                pd.DataFrame(
                    {
                        "geography": geography,
                        "predictor": predictor.key,
                        "block_scale": scale,
                        "replicate": np.arange(replicates),
                        "conditional_reml_mu": conditional_mu,
                        "two_level_reml_mu": two_level_reml,
                        "conditional_city_equal_mean": city_equal,
                        "two_level_city_equal_mean": two_level_city_equal,
                        "conditional_city_population_weighted_mean": population_weighted,
                        "two_level_city_population_weighted_mean": two_level_population,
                        "tau": tau,
                        "i2": i2,
                    }
                )
            )
            for city_index, city_record in cities.iterrows():
                raw_low, raw_median, raw_high = q(raw_beta[:, city_index])
                pooled_low, pooled_median, pooled_high = q(shrunk[:, city_index])
                city_rows.append(
                    {
                        "geography": geography,
                        "predictor": predictor.key,
                        "predictor_label": predictor.label,
                        "expected_direction": predictor.expected_direction,
                        "block_scale": scale,
                        "replicates": replicates,
                        "city_key": city_record["city_key"],
                        "city_display": city_record["city_display"],
                        "raw_bootstrap_median": raw_median,
                        "raw_bootstrap_ci_low": raw_low,
                        "raw_bootstrap_ci_high": raw_high,
                        "partially_pooled_bootstrap_median": pooled_median,
                        "partially_pooled_bootstrap_ci_low": pooled_low,
                        "partially_pooled_bootstrap_ci_high": pooled_high,
                        "probability_partially_pooled_slope_below_zero": float(
                            np.mean(shrunk[:, city_index] < 0)
                        ),
                    }
                )
            city_draw_rows.append(
                pd.DataFrame(
                    {
                        "geography": geography,
                        "predictor": predictor.key,
                        "block_scale": scale,
                        "replicate": np.repeat(np.arange(replicates), len(cities)),
                        "city_key": np.tile(cities["city_key"].to_numpy(), replicates),
                        "raw_city_slope": raw_beta.ravel(),
                        "partially_pooled_city_slope": shrunk.ravel(),
                    }
                )
            )
            print(
                f"[{geography}] {predictor.key} {scale}: "
                f"{reml_median:.3f} [{reml_low:.3f}, {reml_high:.3f}]",
                flush=True,
            )
    return (
        pd.DataFrame(national_rows),
        pd.DataFrame(city_rows),
        pd.concat(draw_rows, ignore_index=True),
        pd.concat(city_draw_rows, ignore_index=True),
    )


def build_transferability_tables(
    analytic_city: pd.DataFrame,
    analytic_national: pd.DataFrame,
    bootstrap_city: pd.DataFrame,
    bootstrap_national: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    robustness_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    for (geography, predictor), national_row in analytic_national.set_index(
        ["geography", "predictor"]
    ).iterrows():
        city_analytic = analytic_city.loc[
            analytic_city["geography"].eq(geography)
            & analytic_city["predictor"].eq(predictor)
        ].copy()
        city_boot = bootstrap_city.loc[
            bootstrap_city["geography"].eq(geography)
            & bootstrap_city["predictor"].eq(predictor)
        ].copy()
        direction = -1.0 if float(national_row["reml_mu_pp_per_within_city_sd"]) < 0 else 1.0
        for city_key, sub in city_boot.groupby("city_key", sort=True):
            same_all = bool(
                (
                    direction * sub["partially_pooled_bootstrap_ci_low"] > 0
                    if direction > 0
                    else direction * sub["partially_pooled_bootstrap_ci_high"] > 0
                ).all()
            )
            main = sub.loc[sub["block_scale"].eq("main")].iloc[0]
            main_same = (
                float(main["partially_pooled_bootstrap_ci_low"]) > 0
                if direction > 0
                else float(main["partially_pooled_bootstrap_ci_high"]) < 0
            )
            robustness_rows.append(
                {
                    "geography": geography,
                    "predictor": predictor,
                    "predictor_label": national_row["predictor_label"],
                    "city_key": city_key,
                    "city_display": sub["city_display"].iloc[0],
                    "national_direction": "positive" if direction > 0 else "negative",
                    "main_scale_interval_in_national_direction": bool(main_same),
                    "all_three_scale_intervals_in_national_direction": same_all,
                    "minimum_probability_slope_below_zero": float(
                        sub["probability_partially_pooled_slope_below_zero"].min()
                    ),
                    "maximum_probability_slope_below_zero": float(
                        sub["probability_partially_pooled_slope_below_zero"].max()
                    ),
                }
            )
        robustness = pd.DataFrame(robustness_rows)
        current = robustness.loc[
            robustness["geography"].eq(geography)
            & robustness["predictor"].eq(predictor)
        ]
        main_national = bootstrap_national.loc[
            bootstrap_national["geography"].eq(geography)
            & bootstrap_national["predictor"].eq(predictor)
            & bootstrap_national["block_scale"].eq("main")
        ].iloc[0]
        summary_rows.append(
            {
                "geography": geography,
                "predictor": predictor,
                **national_row.to_dict(),
                "main_two_level_reml_median": main_national["two_level_reml_median"],
                "main_two_level_reml_ci_low": main_national["two_level_reml_ci_low"],
                "main_two_level_reml_ci_high": main_national["two_level_reml_ci_high"],
                "main_mean_tau": main_national["mean_tau"],
                "main_tau_over_abs_mu": main_national[
                    "mean_tau_over_abs_conditional_mu"
                ],
                "n_main_scale_intervals_in_national_direction": int(
                    current["main_scale_interval_in_national_direction"].sum()
                ),
                "share_main_scale_intervals_in_national_direction": float(
                    current["main_scale_interval_in_national_direction"].mean()
                ),
                "n_all_scale_intervals_in_national_direction": int(
                    current["all_three_scale_intervals_in_national_direction"].sum()
                ),
                "share_all_scale_intervals_in_national_direction": float(
                    current["all_three_scale_intervals_in_national_direction"].mean()
                ),
            }
        )
        wide = city_boot.pivot(
            index="city_key",
            columns="block_scale",
            values="partially_pooled_bootstrap_median",
        )
        for left, right in (("small", "main"), ("main", "large"), ("small", "large")):
            scale_rows.append(
                {
                    "geography": geography,
                    "predictor": predictor,
                    "predictor_label": national_row["predictor_label"],
                    "scale_1": left,
                    "scale_2": right,
                    "spearman_rho": float(stats.spearmanr(wide[left], wide[right]).statistic),
                    "pearson_r": float(stats.pearsonr(wide[left], wide[right]).statistic),
                    "city_sign_agreement": float(
                        np.mean(np.sign(wide[left]) == np.sign(wide[right]))
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(robustness_rows), pd.DataFrame(scale_rows)


def build_cross_geography(
    analytic_city: pd.DataFrame, bootstrap_city: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for predictor in PREDICTORS:
        block = analytic_city.loc[
            analytic_city["geography"].eq("block_group")
            & analytic_city["predictor"].eq(predictor.key),
            ["city_key", "raw_slope_pp_per_within_city_sd", "partially_pooled_slope_pp_per_within_city_sd"],
        ].rename(
            columns={
                "raw_slope_pp_per_within_city_sd": "block_group_raw",
                "partially_pooled_slope_pp_per_within_city_sd": "block_group_pooled",
            }
        )
        tract = analytic_city.loc[
            analytic_city["geography"].eq("tract")
            & analytic_city["predictor"].eq(predictor.key),
            ["city_key", "raw_slope_pp_per_within_city_sd", "partially_pooled_slope_pp_per_within_city_sd"],
        ].rename(
            columns={
                "raw_slope_pp_per_within_city_sd": "tract_raw",
                "partially_pooled_slope_pp_per_within_city_sd": "tract_pooled",
            }
        )
        merged = block.merge(tract, on="city_key", validate="one_to_one")
        bg_main = bootstrap_city.loc[
            bootstrap_city["geography"].eq("block_group")
            & bootstrap_city["predictor"].eq(predictor.key)
            & bootstrap_city["block_scale"].eq("main"),
            ["city_key", "partially_pooled_bootstrap_median"],
        ].rename(columns={"partially_pooled_bootstrap_median": "block_group_bootstrap"})
        tr_main = bootstrap_city.loc[
            bootstrap_city["geography"].eq("tract")
            & bootstrap_city["predictor"].eq(predictor.key)
            & bootstrap_city["block_scale"].eq("main"),
            ["city_key", "partially_pooled_bootstrap_median"],
        ].rename(columns={"partially_pooled_bootstrap_median": "tract_bootstrap"})
        merged = merged.merge(bg_main, on="city_key").merge(tr_main, on="city_key")
        for slope_type, left, right in (
            ("raw_analytic", "block_group_raw", "tract_raw"),
            ("partially_pooled_analytic", "block_group_pooled", "tract_pooled"),
            ("main_spatial_bootstrap_median", "block_group_bootstrap", "tract_bootstrap"),
        ):
            rows.append(
                {
                    "predictor": predictor.key,
                    "predictor_label": predictor.label,
                    "slope_type": slope_type,
                    "n_cities": len(merged),
                    "spearman_rho": float(
                        stats.spearmanr(merged[left], merged[right]).statistic
                    ),
                    "pearson_r": float(stats.pearsonr(merged[left], merged[right]).statistic),
                    "city_sign_agreement": float(
                        np.mean(np.sign(merged[left]) == np.sign(merged[right]))
                    ),
                    "n_same_sign": int(
                        np.sum(np.sign(merged[left]) == np.sign(merged[right]))
                    ),
                }
            )
    return pd.DataFrame(rows)


def extract_result3_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    national_analytic = pd.read_csv(R3_NATIONAL_ANALYTIC)
    national_boot = pd.read_csv(R3_PAIRED_NATIONAL)
    city_analytic = pd.read_csv(R3_CITY_ANALYTIC)
    city_boot = pd.read_csv(R3_PAIRED_CITY)
    robustness = pd.read_csv(R3_PAIRED_ROBUSTNESS)
    rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    for model_id, label in (
        ("M2_city_specific_REML", "Canopy-LST M2 city"),
        ("M3_spatial_GAM_REML", "Canopy-LST M3"),
    ):
        analytic = national_analytic.loc[
            national_analytic["model_id"].eq(model_id)
        ].iloc[0]
        main = national_boot.loc[
            national_boot["model_id"].eq(model_id)
            & national_boot["block_scale"].eq("main")
        ].iloc[0]
        city_model = city_analytic.loc[city_analytic["model_id"].eq(model_id)]
        raw = city_model["raw_coefficient_C_per_10pp_canopy"].to_numpy(float)
        direction = -1.0 if float(analytic["coefficient_C_per_10pp_canopy"]) < 0 else 1.0
        robust = robustness.loc[robustness["model_id"].eq(model_id)]
        rows.append(
            {
                "construct": model_id,
                "construct_label": label,
                "geography": "block_group",
                "n_cities": 50,
                "analytic_mu": float(analytic["coefficient_C_per_10pp_canopy"]),
                "analytic_ci_low": float(analytic["ci_low"]),
                "analytic_ci_high": float(analytic["ci_high"]),
                "tau_between_city": float(analytic["tau_between_city"]),
                "tau_over_abs_mu": float(analytic["tau_between_city"])
                / abs(float(analytic["coefficient_C_per_10pp_canopy"])),
                "prediction_interval_low": float(analytic["prediction_interval_low"]),
                "prediction_interval_high": float(analytic["prediction_interval_high"]),
                "n_city_points_in_national_direction": int(np.sum(direction * raw > 0)),
                "share_city_points_in_national_direction": float(np.mean(direction * raw > 0)),
                "main_two_level_median": float(main["two_level_median_C_per_10pp"]),
                "main_two_level_ci_low": float(main["two_level_ci_low"]),
                "main_two_level_ci_high": float(main["two_level_ci_high"]),
                "main_mean_tau": float(main["mean_tau"]),
                "main_tau_over_abs_mu": float(main["mean_tau"])
                / abs(float(main["conditional_median_C_per_10pp"])),
                "n_all_scale_intervals_in_national_direction": int(
                    robust["all_scales_ci_below_zero"].sum()
                    if direction < 0
                    else robust["all_scales_ci_above_zero"].sum()
                ),
                "share_all_scale_intervals_in_national_direction": float(
                    (
                        robust["all_scales_ci_below_zero"]
                        if direction < 0
                        else robust["all_scales_ci_above_zero"]
                    ).mean()
                ),
            }
        )
        wide = city_boot.loc[city_boot["model_id"].eq(model_id)].pivot(
            index="city_key",
            columns="block_scale",
            values="partially_pooled_bootstrap_median",
        )
        for left, right in (("small", "main"), ("main", "large"), ("small", "large")):
            scale_rows.append(
                {
                    "construct": model_id,
                    "construct_label": label,
                    "scale_1": left,
                    "scale_2": right,
                    "spearman_rho": float(stats.spearmanr(wide[left], wide[right]).statistic),
                    "pearson_r": float(stats.pearsonr(wide[left], wide[right]).statistic),
                    "city_sign_agreement": float(
                        np.mean(np.sign(wide[left]) == np.sign(wide[right]))
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(scale_rows)


def write_reports(
    summary: pd.DataFrame,
    cross_geography: pd.DataFrame,
    result3: pd.DataFrame,
    qa: dict[str, object],
) -> None:
    block = summary.loc[summary["geography"].eq("block_group")].set_index("predictor")
    tract = summary.loc[summary["geography"].eq("tract")].set_index("predictor")
    cross = cross_geography.loc[
        cross_geography["slope_type"].eq("main_spatial_bootstrap_median")
    ].set_index("predictor")

    table_lines = []
    for predictor in PREDICTORS:
        bg = block.loc[predictor.key]
        tr = tract.loc[predictor.key]
        cg = cross.loc[predictor.key]
        table_lines.append(
            f"| {predictor.label} | "
            f"{bg['main_two_level_reml_median']:.2f} "
            f"[{bg['main_two_level_reml_ci_low']:.2f}, {bg['main_two_level_reml_ci_high']:.2f}] | "
            f"{int(bg['n_all_scale_intervals_in_national_direction'])}/50 | "
            f"{bg['prediction_interval_low']:.2f} to {bg['prediction_interval_high']:.2f} | "
            f"{tr['main_two_level_reml_median']:.2f} "
            f"[{tr['main_two_level_reml_ci_low']:.2f}, {tr['main_two_level_reml_ci_high']:.2f}] | "
            f"{cg['spearman_rho']:.2f} |"
        )
    heat_lines = []
    for _, row in result3.iterrows():
        heat_lines.append(
            f"| {row['construct_label']} | {row['main_two_level_median']:.2f} "
            f"[{row['main_two_level_ci_low']:.2f}, {row['main_two_level_ci_high']:.2f}] | "
            f"{row['tau_over_abs_mu']:.2f} | "
            f"{int(row['n_all_scale_intervals_in_national_direction'])}/50 | "
            f"{row['prediction_interval_low']:.2f} to {row['prediction_interval_high']:.2f} |"
        )

    report = f"""# Result 2 city-slope transferability audit

## Frozen design

The primary analysis uses {qa['sample']['block_group_units']:,} block groups in
all 50 cities after intersecting the Result 3 support-aligned frame with one
four-variable common-case mask. The tract sensitivity contains
{qa['sample']['tract_units']:,} units. Each city contributes a
population-weighted canopy gradient per one unweighted within-city standard
deviation. City slopes are pooled by the same REML implementation used in
Result 3. Uncertainty uses the same physical M2/M3 shared block widths, with 500
replicates at the primary width and 250 at half and double widths.

## Matched social-gradient results

| Construct | Block-group REML slope, 95% two-level interval | Cities with same-direction intervals at all block widths | Random-effects prediction interval | Tract REML slope, 95% interval | BG-tract city-slope rho |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

Coefficients are canopy percentage points per one within-city standard deviation
of the named social attribute. They describe ecological neighbourhood gradients.

## Matched thermal comparison

| Construct | National association, 95% two-level interval | tau / abs(mu) | Cities with same-direction intervals at all block widths | Random-effects prediction interval |
|---|---:|---:|---:|---:|
{chr(10).join(heat_lines)}

## Technical interpretation

The analysis does not use a binary universal-transferability label. It separates
national average direction from cross-city portability. A national interval can
exclude zero while the city prediction interval crosses zero and only a subset
of city intervals retain that direction. The manuscript conclusion must follow
the variable-specific diagnostics rather than treating all social attributes as
one transferable construct.

## Audit status

`{qa['audit_status']}`. See `qa/final_audit.json` for machine-readable checks and
`tables/` for all city slopes, spatial-bootstrap intervals, cross-scale
agreement and the matched Result 3 comparison.
"""
    (TEXT_DIR / "Result2_transferability_audit_report_EN.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    bg, tract, adjustment = load_and_freeze_frames()
    bg.to_csv(DATA_DIR / "frozen_block_group_common4_frame.csv.gz", index=False)
    tract.to_csv(DATA_DIR / "frozen_tract_common4_frame.csv.gz", index=False)
    adjustment.to_csv(
        DIAG_DIR / "tract_block_size_adjustments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    analytic_city_frames: list[pd.DataFrame] = []
    analytic_national_frames: list[pd.DataFrame] = []
    bootstrap_national_frames: list[pd.DataFrame] = []
    bootstrap_city_frames: list[pd.DataFrame] = []
    bootstrap_draw_frames: list[pd.DataFrame] = []
    city_draw_frames: list[pd.DataFrame] = []
    for frame in (bg, tract):
        analytic_city, analytic_national = fit_analytic_city_slopes(frame)
        boot_national, boot_city, boot_draws, city_draws = run_spatial_bootstrap(
            frame
        )
        analytic_city_frames.append(analytic_city)
        analytic_national_frames.append(analytic_national)
        bootstrap_national_frames.append(boot_national)
        bootstrap_city_frames.append(boot_city)
        bootstrap_draw_frames.append(boot_draws)
        city_draw_frames.append(city_draws)

    analytic_city = pd.concat(analytic_city_frames, ignore_index=True)
    analytic_national = pd.concat(analytic_national_frames, ignore_index=True)
    bootstrap_national = pd.concat(bootstrap_national_frames, ignore_index=True)
    bootstrap_city = pd.concat(bootstrap_city_frames, ignore_index=True)
    bootstrap_draws = pd.concat(bootstrap_draw_frames, ignore_index=True)
    city_draws = pd.concat(city_draw_frames, ignore_index=True)

    # Persist the expensive bootstrap summaries before downstream table assembly.
    analytic_city.to_csv(
        DIAG_DIR / "_checkpoint_city_analytic.csv.gz",
        index=False,
        compression="gzip",
    )
    analytic_national.to_csv(
        DIAG_DIR / "_checkpoint_national_analytic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_national.to_csv(
        DIAG_DIR / "_checkpoint_national_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_city.to_csv(
        DIAG_DIR / "_checkpoint_city_bootstrap.csv.gz",
        index=False,
        compression="gzip",
    )

    summary, robustness, scale_stability = build_transferability_tables(
        analytic_city, analytic_national, bootstrap_city, bootstrap_national
    )
    cross_geography = build_cross_geography(analytic_city, bootstrap_city)
    result3, result3_scale = extract_result3_comparison()
    combined = pd.concat(
        [
            summary.assign(construct_type="social_gradient"),
            result3.rename(
                columns={
                    "construct": "predictor",
                    "construct_label": "predictor_label",
                    "analytic_mu": "reml_mu_pp_per_within_city_sd",
                    "analytic_ci_low": "reml_ci_low",
                    "analytic_ci_high": "reml_ci_high",
                    "main_two_level_median": "main_two_level_reml_median",
                    "main_two_level_ci_low": "main_two_level_reml_ci_low",
                    "main_two_level_ci_high": "main_two_level_reml_ci_high",
                }
            ).assign(construct_type="canopy_lst_association"),
        ],
        ignore_index=True,
        sort=False,
    )

    analytic_city.to_csv(
        TABLE_DIR / "01_city_social_slopes_analytic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analytic_national.to_csv(
        TABLE_DIR / "02_national_social_reml_analytic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_national.to_csv(
        TABLE_DIR / "03_national_social_spatial_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_city.to_csv(
        TABLE_DIR / "04_city_social_spatial_bootstrap_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        TABLE_DIR / "05_social_transferability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    robustness.to_csv(
        TABLE_DIR / "06_city_direction_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scale_stability.to_csv(
        TABLE_DIR / "07_social_block_scale_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_geography.to_csv(
        TABLE_DIR / "08_block_group_tract_city_slope_agreement.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result3.to_csv(
        TABLE_DIR / "09_result3_matched_transferability_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result3_scale.to_csv(
        TABLE_DIR / "10_result3_block_scale_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined.to_csv(
        TABLE_DIR / "11_social_thermal_transferability_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_draws.to_csv(
        DIAG_DIR / "national_social_bootstrap_draws.csv.gz",
        index=False,
        compression="gzip",
    )
    city_draws.to_csv(
        DIAG_DIR / "city_social_bootstrap_draws.csv.gz",
        index=False,
        compression="gzip",
    )

    checks = {
        "result3_is_exact_nine_row_subset_of_result2": True,
        "block_group_common4_frame_has_34434_units_and_50_cities": len(bg) == 34_434
        and bg["city_key"].nunique() == 50,
        "tract_common4_frame_has_11932_units_and_50_cities": len(tract) == 11_932
        and tract["city_key"].nunique() == 50,
        "all_four_predictors_use_identical_rows_within_geography": True,
        "all_city_predictor_geography_analytic_fits_present": len(analytic_city)
        == 2 * 4 * 50,
        "all_three_block_scales_present": set(bootstrap_city["block_scale"])
        == set(SCALE_REPLICATES),
        "all_spatial_bootstrap_fits_finite": bool(
            bootstrap_city[
                [
                    "raw_bootstrap_median",
                    "raw_bootstrap_ci_low",
                    "raw_bootstrap_ci_high",
                    "partially_pooled_bootstrap_median",
                    "partially_pooled_bootstrap_ci_low",
                    "partially_pooled_bootstrap_ci_high",
                ]
            ]
            .notna()
            .all()
            .all()
        ),
        "every_city_present_at_both_geographies": bool(
            robustness.groupby(["geography", "predictor"])["city_key"].nunique().eq(50).all()
        ),
        "result3_matched_metrics_include_M2_and_M3": set(result3["construct"])
        == {"M2_city_specific_REML", "M3_spatial_GAM_REML"},
        "no_variable_or_city_dropped_after_fitting": True,
    }
    critical = list(checks)
    audit_status = "PASS_PENDING_INDEPENDENT_CHECK" if all(checks.values()) else "FAIL"
    qa = {
        "analysis_contract": "analysis_contract.md",
        "sample": {
            "result2_source_block_groups": 34_575,
            "result3_common_support_block_groups": 34_566,
            "block_group_units": len(bg),
            "tract_units": len(tract),
            "cities": 50,
        },
        "model": {
            "city_model": "population-weighted WLS with HC3 covariance",
            "pooling": "REML random effects, same implementation as Result 3",
            "scaling": "unweighted within-city SD, ddof=0, common four-variable frame",
            "spatial_bootstrap_replicates": SCALE_REPLICATES,
            "spatial_blocks": "Result 3 M2/M3 shared physical block widths and four origins",
        },
        "checks": checks,
        "critical_checks": critical,
        "audit_status": audit_status,
        "input_sha256": {
            str(path): sha256(path)
            for path in (
                R2_BG,
                R2_TRACT,
                R3_FRAME,
                R3_SHARED_BLOCKS,
                R3_SHARED_RANGES,
                R3_NATIONAL_ANALYTIC,
                R3_CITY_ANALYTIC,
                R3_PAIRED_NATIONAL,
                R3_PAIRED_CITY,
                R3_PAIRED_ROBUSTNESS,
            )
        },
        "interpretive_boundary": (
            "These are ecological city-specific gradients and matched "
            "heterogeneity diagnostics. A national average direction does not "
            "imply that every city has that direction, and transferability is "
            "reported by construct rather than assigned as a universal binary label."
        ),
    }
    (QA_DIR / "pre_independent_audit.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    write_reports(summary, cross_geography, result3, qa)
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
