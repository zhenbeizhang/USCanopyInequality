from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


MODULE_SCRIPT = Path(__file__).with_name("run_support_aligned_models.py")
spec = importlib.util.spec_from_file_location("support_result3", MODULE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_SCRIPT}")
support = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = support
spec.loader.exec_module(support)

OUT_ROOT = support.OUT_ROOT / "spatial_block_bootstrap"
DATA_DIR = OUT_ROOT / "data"
TABLE_DIR = OUT_ROOT / "tables"
DIAG_DIR = OUT_ROOT / "diagnostics"
QA_DIR = OUT_ROOT / "qa"
TEXT_DIR = OUT_ROOT / "text"
CHECKPOINT_DIR = OUT_ROOT / "checkpoints"

MODEL_FRAME = support.DATA_DIR / "support_aligned_model_frame_34566.csv"
PREDICTIONS = support.DIAG_DIR / "in_sample_predictions_and_residuals.csv"
SELECTION = support.QA_DIR / "nested_spatial_cv_selection.json"

BLOCK_ASSIGNMENTS = DATA_DIR / "spatial_block_assignments.csv"
BOOTSTRAP_FRAME = DATA_DIR / "bootstrap_model_frame.csv"
CITY_ORDER = DATA_DIR / "city_order.csv"
RANGE_TABLE = TABLE_DIR / "city_residual_range_and_block_sizes.csv"

DISTANCE_BIN_EDGES_KM = np.asarray([0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50], dtype=float)
MAX_RANGE_SAMPLE = 1_200
MIN_PAIRS_PER_BIN = 150
MIN_OCCUPIED_BLOCKS = 8
MIN_BLOCK_SIZE_M = 750.0
MAX_BLOCK_SIZE_M = 15_000.0
GRID_OFFSETS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
SCALE_MULTIPLIERS = {"small": 0.5, "main": 1.0, "large": 2.0}
BASE_SEED = 20260721

MODEL_COLUMNS = [
    "support_row",
    "analysis_record_id",
    "city_key",
    "city_display",
    "GEOID_str",
    "heat_anomaly_C",
    "population_model_weight",
    "x_5070_m",
    "y_5070_m",
    "x_city_0_1",
    "y_city_0_1",
    *support.GAM_LINEAR_COLUMNS,
]

_WORKER_FRAME: pd.DataFrame | None = None
_WORKER_BLOCKS: pd.DataFrame | None = None
_WORKER_CITY_ORDER: list[str] | None = None
_WORKER_CITY_ROWS: dict[str, np.ndarray] | None = None
_WORKER_N_SPLINES: int | None = None
_WORKER_LAMBDA: float | None = None


def ensure_dirs() -> None:
    for path in (DATA_DIR, TABLE_DIR, DIAG_DIR, QA_DIR, TEXT_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def empirical_residual_correlation(
    city_key: str, coords_m: np.ndarray, residual: np.ndarray
) -> tuple[list[dict[str, object]], float, str]:
    rng = np.random.default_rng(BASE_SEED + sum(ord(character) for character in city_key))
    valid = np.isfinite(coords_m).all(axis=1) & np.isfinite(residual)
    coords_m = coords_m[valid]
    residual = residual[valid]
    if len(residual) > MAX_RANGE_SAMPLE:
        selected = np.sort(rng.choice(len(residual), MAX_RANGE_SAMPLE, replace=False))
        coords_m = coords_m[selected]
        residual = residual[selected]

    z = (residual - residual.mean()) / max(residual.std(ddof=1), 1e-9)
    distances_km = pdist(coords_m) / 1_000.0
    upper_i, upper_j = np.triu_indices(len(z), k=1)
    products = z[upper_i] * z[upper_j]
    bin_index = np.digitize(distances_km, DISTANCE_BIN_EDGES_KM, right=False) - 1

    rows: list[dict[str, object]] = []
    midpoints: list[float] = []
    correlations: list[float] = []
    for index in range(len(DISTANCE_BIN_EDGES_KM) - 1):
        chosen = bin_index == index
        count = int(chosen.sum())
        low = float(DISTANCE_BIN_EDGES_KM[index])
        high = float(DISTANCE_BIN_EDGES_KM[index + 1])
        midpoint = (low + high) / 2.0
        correlation = float(np.mean(products[chosen])) if count else np.nan
        rows.append(
            {
                "city_key": city_key,
                "distance_low_km": low,
                "distance_high_km": high,
                "distance_midpoint_km": midpoint,
                "residual_pair_correlation": correlation,
                "pair_count": count,
                "sampled_points": len(z),
            }
        )
        if count >= MIN_PAIRS_PER_BIN and np.isfinite(correlation):
            midpoints.append(midpoint)
            correlations.append(correlation)

    if not correlations:
        return rows, 5_000.0, "fallback_no_valid_distance_bins"
    x = np.asarray(midpoints, dtype=float)
    y = np.asarray(correlations, dtype=float)
    if len(y) >= 3:
        smooth = y.copy()
        smooth[1:-1] = (y[:-2] + 2 * y[1:-1] + y[2:]) / 4.0
    else:
        smooth = y
    start = int(np.nanargmax(smooth[: min(3, len(smooth))]))
    crossing: int | None = None
    for index in range(start + 1, len(smooth)):
        persistent = index == len(smooth) - 1 or smooth[index + 1] <= 0.10
        if smooth[index] <= 0.10 and persistent:
            crossing = index
            break
    if crossing is None:
        positive = np.flatnonzero(smooth > 0.10)
        if len(positive):
            range_km = float(min(x[positive[-1]], 15.0))
            method = "last_valid_bin_above_0.10_capped_at_15km"
        else:
            range_km = float(max(x[0], 1.0))
            method = "first_valid_bin_at_or_below_0.10"
    elif crossing == 0:
        range_km = float(max(x[0], 1.0))
        method = "first_valid_bin_at_or_below_0.10"
    else:
        x0, x1 = x[crossing - 1], x[crossing]
        y0, y1 = smooth[crossing - 1], smooth[crossing]
        range_km = float(x0 + (0.10 - y0) * (x1 - x0) / (y1 - y0)) if y0 > 0.10 and y1 != y0 else float(x1)
        method = "interpolated_first_persistent_crossing_below_0.10"
    return rows, float(np.clip(range_km * 1_000.0, MIN_BLOCK_SIZE_M, MAX_BLOCK_SIZE_M)), method


def assign_blocks(coords: np.ndarray, block_size_m: float, offset: tuple[float, float]) -> np.ndarray:
    shifted = coords - coords.min(axis=0) + np.asarray(offset) * block_size_m
    grid = np.floor(shifted / block_size_m).astype(np.int64)
    _, labels = np.unique(grid, axis=0, return_inverse=True)
    return labels.astype(np.int32)


def adjust_size_for_block_count(coords: np.ndarray, proposed_m: float) -> tuple[float, int, int]:
    size = float(np.clip(proposed_m, MIN_BLOCK_SIZE_M, MAX_BLOCK_SIZE_M))
    while True:
        counts = [len(np.unique(assign_blocks(coords, size, offset))) for offset in GRID_OFFSETS]
        if min(counts) >= MIN_OCCUPIED_BLOCKS or size <= MIN_BLOCK_SIZE_M + 1:
            return size, min(counts), max(counts)
        size = max(MIN_BLOCK_SIZE_M, size * 0.85)


def prepare() -> None:
    ensure_dirs()
    frame = pd.read_csv(MODEL_FRAME, dtype={"GEOID_str": str}, low_memory=False)
    if len(frame) != 34_566 or frame["city_key"].nunique() != 50:
        raise RuntimeError("Support-aligned frame must contain 34,566 block groups in 50 cities.")
    prediction = pd.read_csv(PREDICTIONS, dtype={"GEOID_str": str}, low_memory=False)
    prediction = prediction.loc[prediction["model_id"].eq("M3_spatial_GAM_REML")].copy()
    residual = prediction[["support_row", "residual"]].drop_duplicates("support_row")
    frame = frame.merge(residual, on="support_row", how="left", validate="one_to_one")
    if frame["residual"].isna().any():
        raise RuntimeError("M3 residuals did not align with all support rows.")

    correlation_rows: list[dict[str, object]] = []
    range_rows: list[dict[str, object]] = []
    block_frame = frame[["support_row", "analysis_record_id", "city_key", "GEOID_str"]].copy()
    for number, (city_key, sub) in enumerate(frame.groupby("city_key", sort=True), start=1):
        coords = sub[["x_5070_m", "y_5070_m"]].to_numpy(float)
        rows, range_m, method = empirical_residual_correlation(
            city_key, coords, sub["residual"].to_numpy(float)
        )
        correlation_rows.extend(rows)
        record: dict[str, object] = {
            "city_key": city_key,
            "city_display": sub["city_display"].iloc[0],
            "n_block_groups": len(sub),
            "estimated_residual_correlation_range_km": range_m / 1_000.0,
            "range_method": method,
            "range_definition": "first persistent binned residual-pair correlation at or below 0.10",
        }
        positions = sub.index.to_numpy()
        for scale, multiplier in SCALE_MULTIPLIERS.items():
            size, minimum, maximum = adjust_size_for_block_count(coords, range_m * multiplier)
            record[f"{scale}_block_size_km"] = size / 1_000.0
            record[f"{scale}_minimum_occupied_blocks_across_origins"] = minimum
            record[f"{scale}_maximum_occupied_blocks_across_origins"] = maximum
            for origin_index, offset in enumerate(GRID_OFFSETS):
                block_frame.loc[positions, f"{scale}_block_origin{origin_index}"] = assign_blocks(coords, size, offset)
        range_rows.append(record)
        print(f"[Residual range {number:02d}/50] {city_key}: {range_m / 1000:.2f} km", flush=True)

    pd.DataFrame(correlation_rows).to_csv(
        DIAG_DIR / "residual_pair_correlation_by_distance_city.csv", index=False, encoding="utf-8-sig"
    )
    ranges = pd.DataFrame(range_rows)
    ranges.to_csv(RANGE_TABLE, index=False, encoding="utf-8-sig")
    for column in [name for name in block_frame if "_block_origin" in name]:
        block_frame[column] = block_frame[column].astype(np.int32)
    block_frame.to_csv(BLOCK_ASSIGNMENTS, index=False, encoding="utf-8-sig")
    frame[MODEL_COLUMNS].to_csv(BOOTSTRAP_FRAME, index=False, encoding="utf-8-sig")
    cities = frame[["city_key", "city_display"]].drop_duplicates().sort_values("city_key").reset_index(drop=True)
    cities.insert(0, "city_index", np.arange(len(cities), dtype=np.int16))
    cities.to_csv(CITY_ORDER, index=False, encoding="utf-8-sig")

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    qa = {
        "rows": len(frame),
        "cities": frame["city_key"].nunique(),
        "m3_n_splines_each_dimension": selection["n_splines_each_dimension"],
        "m3_lambda": selection["lambda"],
        "range_min_km": float(ranges["estimated_residual_correlation_range_km"].min()),
        "range_median_km": float(ranges["estimated_residual_correlation_range_km"].median()),
        "range_max_km": float(ranges["estimated_residual_correlation_range_km"].max()),
        "all_scales_minimum_8_occupied_blocks": bool(
            ranges[[f"{scale}_minimum_occupied_blocks_across_origins" for scale in SCALE_MULTIPLIERS]]
            .ge(MIN_OCCUPIED_BLOCKS)
            .all()
            .all()
        ),
    }
    (QA_DIR / "bootstrap_input_validation.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def init_worker() -> None:
    global _WORKER_FRAME, _WORKER_BLOCKS, _WORKER_CITY_ORDER, _WORKER_CITY_ROWS
    global _WORKER_N_SPLINES, _WORKER_LAMBDA
    _WORKER_FRAME = pd.read_csv(BOOTSTRAP_FRAME, dtype={"GEOID_str": str}, low_memory=False)
    _WORKER_BLOCKS = pd.read_csv(BLOCK_ASSIGNMENTS, dtype={"GEOID_str": str}, low_memory=False)
    cities = pd.read_csv(CITY_ORDER).sort_values("city_index")
    _WORKER_CITY_ORDER = cities["city_key"].tolist()
    _WORKER_CITY_ROWS = {
        city: np.flatnonzero(_WORKER_FRAME["city_key"].to_numpy(str) == city)
        for city in _WORKER_CITY_ORDER
    }
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    _WORKER_N_SPLINES = int(selection["n_splines_each_dimension"])
    _WORKER_LAMBDA = float(selection["lambda"])


def bootstrap_one_city(city_key: str, scale: str, rng: np.random.Generator) -> tuple[float, float]:
    assert _WORKER_FRAME is not None and _WORKER_BLOCKS is not None and _WORKER_CITY_ROWS is not None
    assert _WORKER_N_SPLINES is not None and _WORKER_LAMBDA is not None
    row_positions = _WORKER_CITY_ROWS[city_key]
    for _ in range(8):
        origin = int(rng.integers(0, len(GRID_OFFSETS)))
        labels = _WORKER_BLOCKS.iloc[row_positions][f"{scale}_block_origin{origin}"].to_numpy(np.int32)
        unique, inverse = np.unique(labels, return_inverse=True)
        sampled = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(sampled, minlength=len(unique))
        multiplier = counts[inverse]
        selected = multiplier > 0
        if selected.sum() < max(45, int(0.20 * len(row_positions))) or np.count_nonzero(counts) < 4:
            continue
        sub = _WORKER_FRAME.iloc[row_positions[selected]].copy()
        sub["population_model_weight"] = (
            sub["population_model_weight"].to_numpy(float) * multiplier[selected]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, _, _, coefficient = support.fit_one_city_gam(
                    sub, _WORKER_N_SPLINES, _WORKER_LAMBDA
                )
            except Exception:
                continue
        beta, se = map(float, coefficient)
        if np.isfinite(beta) and np.isfinite(se) and se > 0:
            return beta, se
    raise RuntimeError(f"Unable to fit a valid block-bootstrap GAM for {city_key} at {scale} scale.")


def bootstrap_replicate(task: tuple[str, int]) -> dict[str, object]:
    scale, replicate = task
    scale_index = list(SCALE_MULTIPLIERS).index(scale)
    for attempt in range(4):
        seed = BASE_SEED + scale_index * 10_000_000 + replicate * 10_000 + attempt * 1_000_000
        rng = np.random.default_rng(seed)
        try:
            raw = np.asarray(
                [bootstrap_one_city(city, scale, rng) for city in _WORKER_CITY_ORDER or []],
                dtype=float,
            )
            pool = support.reml_pool(raw[:, 0], raw[:, 1])
            return {
                "scale": scale,
                "replicate": replicate,
                "seed": seed,
                "national_mu": float(pool["mu"]),
                "national_model_se": float(pool["mu_se"]),
                "tau": float(pool["tau"]),
                "i2": float(pool["i2"]),
                "raw_beta": raw[:, 0],
                "raw_model_se": raw[:, 1],
                "shrunk_beta": np.asarray(pool["shrunk"], dtype=float),
            }
        except Exception:
            if attempt == 3:
                raise
    raise RuntimeError("Unreachable bootstrap failure.")


def checkpoint_path(scale: str, replicate: int) -> Path:
    path = CHECKPOINT_DIR / scale
    path.mkdir(parents=True, exist_ok=True)
    return path / f"replicate_{replicate:04d}.npz"


def save_checkpoint(result: dict[str, object]) -> None:
    np.savez_compressed(
        checkpoint_path(str(result["scale"]), int(result["replicate"])),
        scale=str(result["scale"]),
        replicate=int(result["replicate"]),
        seed=int(result["seed"]),
        national_mu=float(result["national_mu"]),
        national_model_se=float(result["national_model_se"]),
        tau=float(result["tau"]),
        i2=float(result["i2"]),
        raw_beta=np.asarray(result["raw_beta"]),
        raw_model_se=np.asarray(result["raw_model_se"]),
        shrunk_beta=np.asarray(result["shrunk_beta"]),
    )


def run_bootstrap(scale: str, replicates: int, workers: int) -> None:
    ensure_dirs()
    missing = [replicate for replicate in range(replicates) if not checkpoint_path(scale, replicate).exists()]
    if not missing:
        print(f"All {replicates} {scale} checkpoints already exist.")
        return
    print(f"Running {len(missing)} {scale} replicates with {workers} workers.", flush=True)
    failures: list[dict[str, object]] = []
    completed = replicates - len(missing)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        futures = {
            executor.submit(bootstrap_replicate, (scale, replicate)): replicate
            for replicate in missing
        }
        for future in as_completed(futures):
            replicate = futures[future]
            try:
                save_checkpoint(future.result())
                completed += 1
                if completed % 10 == 0 or completed == replicates:
                    print(f"[{scale}] {completed}/{replicates}", flush=True)
            except Exception as error:
                failures.append({"scale": scale, "replicate": replicate, "error": repr(error)})
    if failures:
        pd.DataFrame(failures).to_csv(
            QA_DIR / f"bootstrap_failures_{scale}.csv", index=False, encoding="utf-8-sig"
        )
        raise RuntimeError(f"{len(failures)} bootstrap replicates failed for {scale} scale.")


def load_checkpoint(scale: str, replicate: int) -> dict[str, np.ndarray]:
    with np.load(checkpoint_path(scale, replicate), allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def interval(values: np.ndarray) -> tuple[float, float, float]:
    low, median, high = np.percentile(values, [2.5, 50, 97.5])
    return float(low), float(median), float(high)


def summarize(main_replicates: int, sensitivity_replicates: int) -> None:
    counts = {"small": sensitivity_replicates, "main": main_replicates, "large": sensitivity_replicates}
    cities = pd.read_csv(CITY_ORDER).sort_values("city_index").reset_index(drop=True)
    analytic = pd.read_csv(support.TABLE_DIR / "city_canopy_lst_coefficients_hierarchical_models.csv")
    analytic = analytic.loc[analytic["model_id"].eq("M3_spatial_GAM_REML")].set_index("city_key")
    national_rows: list[dict[str, object]] = []
    national_draws: list[pd.DataFrame] = []
    city_draws: list[pd.DataFrame] = []
    city_rows: list[dict[str, object]] = []

    for scale, count in counts.items():
        missing = [replicate for replicate in range(count) if not checkpoint_path(scale, replicate).exists()]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} {scale} checkpoints.")
        results = [load_checkpoint(scale, replicate) for replicate in range(count)]
        national = np.asarray([float(result["national_mu"]) for result in results])
        raw = np.vstack([result["raw_beta"] for result in results])
        raw_se = np.vstack([result["raw_model_se"] for result in results])
        shrunk = np.vstack([result["shrunk_beta"] for result in results])
        tau = np.asarray([float(result["tau"]) for result in results])
        i2 = np.asarray([float(result["i2"]) for result in results])

        conditional_low, conditional_median, conditional_high = interval(national)
        two_level = np.empty(count, dtype=float)
        rng = np.random.default_rng(BASE_SEED + 90_000_000 + list(SCALE_MULTIPLIERS).index(scale) * 1_000_000)
        for draw in range(count):
            sampled_cities = rng.integers(0, len(cities), size=len(cities))
            sampled_replicates = rng.integers(0, count, size=len(cities))
            beta = raw[sampled_replicates, sampled_cities]
            se = raw_se[sampled_replicates, sampled_cities]
            two_level[draw] = float(support.reml_pool(beta, se)["mu"])
        two_low, two_median, two_high = interval(two_level)
        national_rows.append(
            {
                "block_scale": scale,
                "replicates": count,
                "conditional_50_city_median_C_per_10pp": conditional_median,
                "conditional_50_city_ci_low_C_per_10pp": conditional_low,
                "conditional_50_city_ci_high_C_per_10pp": conditional_high,
                "conditional_probability_slope_below_zero": float(np.mean(national < 0)),
                "two_level_city_and_spatial_median_C_per_10pp": two_median,
                "two_level_city_and_spatial_ci_low_C_per_10pp": two_low,
                "two_level_city_and_spatial_ci_high_C_per_10pp": two_high,
                "two_level_probability_slope_below_zero": float(np.mean(two_level < 0)),
                "mean_between_city_tau": float(tau.mean()),
                "mean_i2": float(i2.mean()),
                "primary_inference": "two-level percentile interval resampling cities and city-specific spatial blocks",
            }
        )
        national_draws.append(
            pd.DataFrame(
                {
                    "block_scale": scale,
                    "replicate": np.arange(count),
                    "conditional_observed_50_city_mu_C_per_10pp": national,
                    "two_level_city_and_spatial_mu_C_per_10pp": two_level,
                }
            )
        )
        for replicate in range(count):
            city_draws.append(
                pd.DataFrame(
                    {
                        "block_scale": scale,
                        "replicate": replicate,
                        "city_key": cities["city_key"],
                        "raw_city_slope_C_per_10pp": raw[replicate],
                        "partially_pooled_city_slope_C_per_10pp": shrunk[replicate],
                    }
                )
            )
        for city_index, record in cities.iterrows():
            raw_low, raw_median, raw_high = interval(raw[:, city_index])
            pooled_low, pooled_median, pooled_high = interval(shrunk[:, city_index])
            city_key = record["city_key"]
            city_rows.append(
                {
                    "block_scale": scale,
                    "replicates": count,
                    "city_key": city_key,
                    "city_display": record["city_display"],
                    "analytic_M3_partially_pooled_slope_C_per_10pp": float(
                        analytic.loc[city_key, "coefficient_C_per_10pp_canopy"]
                    ),
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

    national_table = pd.DataFrame(national_rows)
    national_table.to_csv(
        TABLE_DIR / "national_spatial_block_bootstrap_inference.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(national_draws, ignore_index=True).to_csv(
        DIAG_DIR / "national_spatial_block_bootstrap_draws.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(city_draws, ignore_index=True).to_csv(
        DIAG_DIR / "city_slope_bootstrap_draws.csv", index=False, encoding="utf-8-sig"
    )
    city_table = pd.DataFrame(city_rows)
    city_table.to_csv(
        TABLE_DIR / "city_spatial_block_bootstrap_intervals.csv", index=False, encoding="utf-8-sig"
    )

    robustness_rows: list[dict[str, object]] = []
    for city_key, sub in city_table.groupby("city_key", sort=True):
        robustness_rows.append(
            {
                "city_key": city_key,
                "city_display": sub["city_display"].iloc[0],
                "all_scales_ci_below_zero": bool((sub["partially_pooled_bootstrap_ci_high"] < 0).all()),
                "all_scales_ci_above_zero": bool((sub["partially_pooled_bootstrap_ci_low"] > 0).all()),
                "minimum_probability_slope_below_zero_across_scales": float(
                    sub["probability_partially_pooled_slope_below_zero"].min()
                ),
                "maximum_probability_slope_below_zero_across_scales": float(
                    sub["probability_partially_pooled_slope_below_zero"].max()
                ),
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(TABLE_DIR / "city_slope_block_scale_robustness.csv", index=False, encoding="utf-8-sig")

    main = national_table.loc[national_table["block_scale"].eq("main")].iloc[0]
    report = f"""# Support-aligned spatial block-bootstrap inference

The analysis resampled spatial blocks within 34,566 fully contained block groups in all 50 cities. The primary block width was estimated separately for each city from the distance at which M3 residual-pair correlation persistently declined to 0.10 or less. The M3 specification and hyperparameters were fixed before bootstrap inference using nested spatial cross-validation.

At the primary block scale, the conditional 50-city median association was {main['conditional_50_city_median_C_per_10pp']:.3f} C per 10 percentage points of canopy (95% interval {main['conditional_50_city_ci_low_C_per_10pp']:.3f} to {main['conditional_50_city_ci_high_C_per_10pp']:.3f}). The two-level interval, which additionally resampled cities, was {main['two_level_city_and_spatial_ci_low_C_per_10pp']:.3f} to {main['two_level_city_and_spatial_ci_high_C_per_10pp']:.3f}; {100 * main['two_level_probability_slope_below_zero']:.1f}% of draws were below zero. Half- and double-width block scales are reported as sensitivity analyses.

These intervals quantify uncertainty from residual spatial dependence and between-city heterogeneity. They do not convert the cross-sectional LST association into a causal cooling effect, air-temperature response, personal exposure or health effect.
"""
    (TEXT_DIR / "spatial_block_bootstrap_report.md").write_text(report, encoding="utf-8")
    qa = {
        "main_replicates": main_replicates,
        "sensitivity_replicates_each_scale": sensitivity_replicates,
        "all_national_intervals_below_zero": bool(
            (national_table["two_level_city_and_spatial_ci_high_C_per_10pp"] < 0).all()
        ),
        "cities_ci_below_zero_all_scales": int(robustness["all_scales_ci_below_zero"].sum()),
        "cities_ci_above_zero_all_scales": int(robustness["all_scales_ci_above_zero"].sum()),
    }
    (QA_DIR / "bootstrap_final_validation.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Support-aligned spatial block bootstrap.")
    parser.add_argument("--stage", choices=("prepare", "bootstrap", "summarize"), required=True)
    parser.add_argument("--scale", choices=tuple(SCALE_MULTIPLIERS), default="main")
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--main-replicates", type=int, default=500)
    parser.add_argument("--sensitivity-replicates", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "bootstrap":
        run_bootstrap(args.scale, args.replicates, args.workers)
    else:
        summarize(args.main_replicates, args.sensitivity_replicates)


if __name__ == "__main__":
    main()
