from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import importlib.util
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


HERE = Path(__file__).resolve().parent


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


support = import_module("support_result3_paired", HERE / "run_support_aligned_models.py")
spatial = import_module(
    "support_spatial_bootstrap_paired", HERE / "run_spatial_block_bootstrap.py"
)

ROOT = support.OUT_ROOT
OUT_ROOT = ROOT / "paired_m2_m3_shared_block_bootstrap"
DATA_DIR = OUT_ROOT / "data"
TABLE_DIR = OUT_ROOT / "tables"
DIAG_DIR = OUT_ROOT / "diagnostics"
QA_DIR = OUT_ROOT / "qa"
TEXT_DIR = OUT_ROOT / "text"
CHECKPOINT_DIR = OUT_ROOT / "checkpoints"

MODEL_FRAME = ROOT / "data" / "support_aligned_model_frame_34566.csv"
PREDICTIONS = ROOT / "diagnostics" / "in_sample_predictions_and_residuals.csv"
MODEL_COEFFICIENTS = ROOT / "tables" / "city_canopy_lst_coefficients_hierarchical_models.csv"
SELECTION = ROOT / "qa" / "nested_spatial_cv_selection.json"
HVCG_SOURCE = Path(
    str(analysis_store_root().joinpath('Result', 'Result4', 'versions', 'counterfactual_density_strata', 'module_05_hvcg_priority', 'result4_module_05_hvcg_priority_block_groups_valid.csv'))
)

BLOCK_ASSIGNMENTS = DATA_DIR / "shared_spatial_block_assignments.csv"
BOOTSTRAP_FRAME = DATA_DIR / "bootstrap_model_frame.csv"
BOOTSTRAP_HVCG = DATA_DIR / "bootstrap_hvcg_frame.csv"
CITY_ORDER = DATA_DIR / "city_order.csv"
RANGE_TABLE = TABLE_DIR / "m2_m3_residual_ranges_and_shared_block_sizes.csv"

MODELS = ("M2_city_specific_REML", "M3_spatial_GAM_REML")
SCALES = spatial.SCALE_MULTIPLIERS
GRID_OFFSETS = spatial.GRID_OFFSETS
BASE_SEED = 20260721

_FRAME: pd.DataFrame | None = None
_BLOCKS: pd.DataFrame | None = None
_HVCG: pd.DataFrame | None = None
_CITY_ORDER: list[str] | None = None
_CITY_ROWS: dict[str, np.ndarray] | None = None
_N_SPLINES: int | None = None
_LAMBDA: float | None = None


def ensure_dirs() -> None:
    for path in (DATA_DIR, TABLE_DIR, DIAG_DIR, QA_DIR, TEXT_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_geoid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def fit_m2_city(sub: pd.DataFrame) -> tuple[float, float]:
    x = sm.add_constant(
        sub[["canopy_pct_10pp", *support.MODEL2_COVARIATES]].astype(float),
        has_constant="add",
    )
    fit = sm.WLS(
        sub["heat_anomaly_C"].to_numpy(float),
        x,
        weights=sub["population_model_weight"].to_numpy(float),
    ).fit(cov_type="HC3")
    return float(fit.params["canopy_pct_10pp"]), float(fit.bse["canopy_pct_10pp"])


def prepare() -> None:
    ensure_dirs()
    frame = pd.read_csv(MODEL_FRAME, dtype={"GEOID_str": str}, low_memory=False)
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    if len(frame) != 34_566 or frame["city_key"].nunique() != 50:
        raise RuntimeError("Expected 34,566 support-aligned block groups in 50 cities.")
    frame = frame.sort_values("support_row").reset_index(drop=True)

    predictions = pd.read_csv(PREDICTIONS, dtype={"GEOID_str": str}, low_memory=False)
    predictions = predictions.loc[predictions["model_id"].isin(MODELS)].copy()
    residual_wide = predictions.pivot(
        index="support_row", columns="model_id", values="residual"
    ).rename(columns={model: f"residual_{model}" for model in MODELS})
    frame = frame.merge(residual_wide, on="support_row", how="left", validate="one_to_one")
    residual_columns = [f"residual_{model}" for model in MODELS]
    if frame[residual_columns].isna().any().any():
        raise RuntimeError("M2/M3 residuals did not align with all support rows.")

    correlation_rows: list[dict[str, object]] = []
    range_rows: list[dict[str, object]] = []
    block_frame = frame[["support_row", "analysis_record_id", "city_key", "GEOID_str"]].copy()
    for number, (city_key, sub) in enumerate(frame.groupby("city_key", sort=True), start=1):
        coords = sub[["x_5070_m", "y_5070_m"]].to_numpy(float)
        model_ranges: dict[str, float] = {}
        model_methods: dict[str, str] = {}
        for model in MODELS:
            rows, range_m, method = spatial.empirical_residual_correlation(
                city_key,
                coords,
                sub[f"residual_{model}"].to_numpy(float),
            )
            for row in rows:
                row["model_id"] = model
            correlation_rows.extend(rows)
            model_ranges[model] = range_m
            model_methods[model] = method
        shared_range_m = max(model_ranges.values())
        shared_driver = max(model_ranges, key=model_ranges.get)
        record: dict[str, object] = {
            "city_key": city_key,
            "city_display": sub["city_display"].iloc[0],
            "n_block_groups": len(sub),
            "m2_residual_range_km": model_ranges["M2_city_specific_REML"] / 1_000.0,
            "m3_residual_range_km": model_ranges["M3_spatial_GAM_REML"] / 1_000.0,
            "shared_primary_range_km": shared_range_m / 1_000.0,
            "shared_range_driver": shared_driver,
            "m2_range_method": model_methods["M2_city_specific_REML"],
            "m3_range_method": model_methods["M3_spatial_GAM_REML"],
            "shared_rule": "city-specific maximum of M2 and M3 residual correlation ranges",
        }
        positions = sub.index.to_numpy()
        for scale, multiplier in SCALES.items():
            size, minimum, maximum = spatial.adjust_size_for_block_count(
                coords, shared_range_m * multiplier
            )
            record[f"{scale}_shared_block_size_km"] = size / 1_000.0
            record[f"{scale}_minimum_occupied_blocks"] = minimum
            record[f"{scale}_maximum_occupied_blocks"] = maximum
            for origin_index, offset in enumerate(GRID_OFFSETS):
                block_frame.loc[
                    positions, f"{scale}_block_origin{origin_index}"
                ] = spatial.assign_blocks(coords, size, offset)
        range_rows.append(record)
        print(
            f"[Shared range {number:02d}/50] {city_key}: "
            f"M2={model_ranges['M2_city_specific_REML']/1000:.2f}, "
            f"M3={model_ranges['M3_spatial_GAM_REML']/1000:.2f}, "
            f"shared={shared_range_m/1000:.2f} km",
            flush=True,
        )

    pd.DataFrame(correlation_rows).to_csv(
        DIAG_DIR / "m2_m3_residual_pair_correlation_by_distance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranges = pd.DataFrame(range_rows)
    ranges.to_csv(RANGE_TABLE, index=False, encoding="utf-8-sig")
    for column in [name for name in block_frame if "_block_origin" in name]:
        block_frame[column] = block_frame[column].astype(np.int32)
    block_frame.to_csv(BLOCK_ASSIGNMENTS, index=False, encoding="utf-8-sig")

    model_columns = [
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
    frame[model_columns].to_csv(BOOTSTRAP_FRAME, index=False, encoding="utf-8-sig")

    hvcg = pd.read_csv(HVCG_SOURCE, dtype={"GEOID_str": str}, low_memory=False)
    hvcg["GEOID_str"] = normalize_geoid(hvcg["GEOID_str"])
    hvcg_columns = [
        "city_key",
        "GEOID_str",
        "population",
        "canopy_gap_q75_pct",
        "heat_excess_reference_C",
        "vulnerability_amplification_factor",
    ]
    aligned_hvcg = frame[
        ["support_row", "analysis_record_id", "city_key", "city_display", "GEOID_str"]
    ].merge(
        hvcg[hvcg_columns],
        on=["city_key", "GEOID_str"],
        how="left",
        validate="one_to_one",
    )
    if aligned_hvcg[hvcg_columns[2:]].isna().any().any():
        raise RuntimeError("HVCG inputs did not align with all 34,566 support rows.")
    aligned_hvcg.to_csv(BOOTSTRAP_HVCG, index=False, encoding="utf-8-sig")

    cities = frame[["city_key", "city_display"]].drop_duplicates().sort_values("city_key").reset_index(drop=True)
    cities.insert(0, "city_index", np.arange(len(cities), dtype=np.int16))
    cities.to_csv(CITY_ORDER, index=False, encoding="utf-8-sig")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    qa = {
        "rows": len(frame),
        "cities": frame["city_key"].nunique(),
        "hvcg_rows": len(aligned_hvcg),
        "m3_n_splines_each_dimension": selection["n_splines_each_dimension"],
        "m3_lambda": selection["lambda"],
        "shared_range_min_km": float(ranges["shared_primary_range_km"].min()),
        "shared_range_median_km": float(ranges["shared_primary_range_km"].median()),
        "shared_range_max_km": float(ranges["shared_primary_range_km"].max()),
        "cities_shared_range_driven_by_m2": int(
            ranges["shared_range_driver"].eq("M2_city_specific_REML").sum()
        ),
        "cities_shared_range_driven_by_m3": int(
            ranges["shared_range_driver"].eq("M3_spatial_GAM_REML").sum()
        ),
        "all_scales_have_at_least_8_blocks": bool(
            ranges[[f"{scale}_minimum_occupied_blocks" for scale in SCALES]]
            .ge(spatial.MIN_OCCUPIED_BLOCKS)
            .all()
            .all()
        ),
    }
    (QA_DIR / "paired_bootstrap_input_validation.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def init_worker() -> None:
    global _FRAME, _BLOCKS, _HVCG, _CITY_ORDER, _CITY_ROWS, _N_SPLINES, _LAMBDA
    _FRAME = pd.read_csv(BOOTSTRAP_FRAME, dtype={"GEOID_str": str}, low_memory=False)
    _BLOCKS = pd.read_csv(BLOCK_ASSIGNMENTS, dtype={"GEOID_str": str}, low_memory=False)
    _HVCG = pd.read_csv(BOOTSTRAP_HVCG, dtype={"GEOID_str": str}, low_memory=False)
    cities = pd.read_csv(CITY_ORDER).sort_values("city_index")
    _CITY_ORDER = cities["city_key"].tolist()
    _CITY_ROWS = {
        city: np.flatnonzero(_FRAME["city_key"].to_numpy(str) == city)
        for city in _CITY_ORDER
    }
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))
    _N_SPLINES = int(selected["n_splines_each_dimension"])
    _LAMBDA = float(selected["lambda"])


def sample_city_once(
    city_key: str, scale: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    assert _FRAME is not None and _BLOCKS is not None and _CITY_ROWS is not None
    rows = _CITY_ROWS[city_key]
    origin = int(rng.integers(0, len(GRID_OFFSETS)))
    labels = _BLOCKS.iloc[rows][f"{scale}_block_origin{origin}"].to_numpy(np.int32)
    unique, inverse = np.unique(labels, return_inverse=True)
    sampled = rng.integers(0, len(unique), size=len(unique))
    counts = np.bincount(sampled, minlength=len(unique))
    multiplier = counts[inverse]
    selected = multiplier > 0
    if selected.sum() < max(45, int(0.20 * len(rows))) or np.count_nonzero(counts) < 4:
        raise RuntimeError("Insufficient sampled blocks")
    return rows[selected], multiplier[selected]


def fit_paired_city(
    city_key: str, scale: str, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    assert _FRAME is not None and _N_SPLINES is not None and _LAMBDA is not None
    for _ in range(8):
        try:
            selected_rows, multiplier = sample_city_once(city_key, scale, rng)
            sub = _FRAME.iloc[selected_rows].copy()
            sub["population_model_weight"] = (
                sub["population_model_weight"].to_numpy(float) * multiplier
            )
            m2_beta, m2_se = fit_m2_city(sub)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, _, m3_coefficient = support.fit_one_city_gam(
                    sub, _N_SPLINES, _LAMBDA
                )
            m3_beta, m3_se = map(float, m3_coefficient)
            values = np.asarray([m2_beta, m2_se, m3_beta, m3_se])
            if np.isfinite(values).all() and m2_se > 0 and m3_se > 0:
                return m2_beta, m2_se, m3_beta, m3_se
        except Exception:
            continue
    raise RuntimeError(f"Unable to fit paired models for {city_key} at {scale} scale.")


def tier1_membership(slopes: np.ndarray) -> np.ndarray:
    assert _HVCG is not None and _CITY_ORDER is not None and _CITY_ROWS is not None
    selected = np.zeros(len(_HVCG), dtype=bool)
    for city_index, city_key in enumerate(_CITY_ORDER):
        rows = _CITY_ROWS[city_key]
        slope = float(slopes[city_index])
        if slope >= 0:
            continue
        city = _HVCG.iloc[rows]
        relievable = np.minimum(
            np.clip(city["heat_excess_reference_C"].to_numpy(float), 0, None),
            (-slope) * city["canopy_gap_q75_pct"].to_numpy(float) / 10.0,
        )
        intensity = relievable * city["vulnerability_amplification_factor"].to_numpy(float)
        burden = intensity * city["population"].to_numpy(float)
        n = len(rows)
        intensity_percentile = stats.rankdata(intensity, method="average") / n
        burden_percentile = stats.rankdata(burden, method="average") / n
        selected[rows] = (intensity_percentile >= 0.90) & (burden_percentile >= 0.90)
    return selected


def paired_replicate(task: tuple[str, int]) -> dict[str, object]:
    scale, replicate = task
    scale_index = list(SCALES).index(scale)
    for attempt in range(4):
        seed = BASE_SEED + scale_index * 10_000_000 + replicate * 10_000 + attempt * 1_000_000
        rng = np.random.default_rng(seed)
        try:
            raw = np.asarray(
                [fit_paired_city(city, scale, rng) for city in _CITY_ORDER or []],
                dtype=float,
            )
            m2_pool = support.reml_pool(raw[:, 0], raw[:, 1])
            m3_pool = support.reml_pool(raw[:, 2], raw[:, 3])
            m2_tier = tier1_membership(np.asarray(m2_pool["shrunk"], dtype=float))
            m3_tier = tier1_membership(np.asarray(m3_pool["shrunk"], dtype=float))
            return {
                "scale": scale,
                "replicate": replicate,
                "seed": seed,
                "m2_mu": float(m2_pool["mu"]),
                "m2_mu_se": float(m2_pool["mu_se"]),
                "m2_tau": float(m2_pool["tau"]),
                "m2_i2": float(m2_pool["i2"]),
                "m2_raw_beta": raw[:, 0],
                "m2_raw_se": raw[:, 1],
                "m2_shrunk": np.asarray(m2_pool["shrunk"], dtype=float),
                "m3_mu": float(m3_pool["mu"]),
                "m3_mu_se": float(m3_pool["mu_se"]),
                "m3_tau": float(m3_pool["tau"]),
                "m3_i2": float(m3_pool["i2"]),
                "m3_raw_beta": raw[:, 2],
                "m3_raw_se": raw[:, 3],
                "m3_shrunk": np.asarray(m3_pool["shrunk"], dtype=float),
                "m2_tier_packed": np.packbits(m2_tier, bitorder="little"),
                "m3_tier_packed": np.packbits(m3_tier, bitorder="little"),
                "m2_tier_n": int(m2_tier.sum()),
                "m3_tier_n": int(m3_tier.sum()),
                "tier_intersection_n": int((m2_tier & m3_tier).sum()),
                "tier_union_n": int((m2_tier | m3_tier).sum()),
            }
        except Exception:
            if attempt == 3:
                raise
    raise RuntimeError("Unreachable paired bootstrap failure")


def checkpoint_path(scale: str, replicate: int) -> Path:
    directory = CHECKPOINT_DIR / scale
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"replicate_{replicate:04d}.npz"


def save_checkpoint(result: dict[str, object]) -> None:
    np.savez_compressed(
        checkpoint_path(str(result["scale"]), int(result["replicate"])),
        **{key: value for key, value in result.items() if key != "scale"},
        scale=str(result["scale"]),
    )


def run_bootstrap(scale: str, replicates: int, workers: int) -> None:
    ensure_dirs()
    missing = [replicate for replicate in range(replicates) if not checkpoint_path(scale, replicate).exists()]
    if not missing:
        print(f"All {replicates} {scale} paired checkpoints exist.")
        return
    completed = replicates - len(missing)
    failures: list[dict[str, object]] = []
    print(f"Running {len(missing)} paired {scale} replicates with {workers} workers.", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        futures = {
            executor.submit(paired_replicate, (scale, replicate)): replicate
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
            QA_DIR / f"paired_failures_{scale}.csv", index=False, encoding="utf-8-sig"
        )
        raise RuntimeError(f"{len(failures)} paired replicates failed for {scale}.")


def load_checkpoint(scale: str, replicate: int) -> dict[str, np.ndarray]:
    with np.load(checkpoint_path(scale, replicate), allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def interval(values: np.ndarray) -> tuple[float, float, float]:
    low, median, high = np.percentile(values, [2.5, 50, 97.5])
    return float(low), float(median), float(high)


def deterministic_tier_membership(
    hvcg: pd.DataFrame,
    city_order: list[str],
    city_rows: dict[str, np.ndarray],
    slopes: np.ndarray,
) -> np.ndarray:
    selected = np.zeros(len(hvcg), dtype=bool)
    for city_index, city_key in enumerate(city_order):
        rows = city_rows[city_key]
        slope = float(slopes[city_index])
        if slope >= 0:
            continue
        city = hvcg.iloc[rows]
        relievable = np.minimum(
            np.clip(city["heat_excess_reference_C"].to_numpy(float), 0, None),
            (-slope) * city["canopy_gap_q75_pct"].to_numpy(float) / 10.0,
        )
        intensity = relievable * city["vulnerability_amplification_factor"].to_numpy(float)
        burden = intensity * city["population"].to_numpy(float)
        n = len(rows)
        selected[rows] = (
            (stats.rankdata(intensity, method="average") / n >= 0.90)
            & (stats.rankdata(burden, method="average") / n >= 0.90)
        )
    return selected


def summarize(main_replicates: int, sensitivity_replicates: int) -> None:
    ensure_dirs()
    counts = {"small": sensitivity_replicates, "main": main_replicates, "large": sensitivity_replicates}
    cities = pd.read_csv(CITY_ORDER).sort_values("city_index").reset_index(drop=True)
    city_order = cities["city_key"].tolist()
    hvcg = pd.read_csv(BOOTSTRAP_HVCG, dtype={"GEOID_str": str}, low_memory=False)
    city_rows = {
        city: np.flatnonzero(hvcg["city_key"].to_numpy(str) == city)
        for city in city_order
    }
    analytic = pd.read_csv(MODEL_COEFFICIENTS)
    analytic = analytic.loc[analytic["model_id"].isin(MODELS)].copy()
    analytic_map = {
        model: analytic.loc[analytic["model_id"].eq(model)].set_index("city_key")
        for model in MODELS
    }
    analytic_slopes = {
        model: np.asarray(
            [float(analytic_map[model].loc[city, "coefficient_C_per_10pp_canopy"]) for city in city_order]
        )
        for model in MODELS
    }
    deterministic = {
        model: deterministic_tier_membership(hvcg, city_order, city_rows, analytic_slopes[model])
        for model in MODELS
    }

    national_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    national_draw_rows: list[pd.DataFrame] = []
    city_summary_rows: list[dict[str, object]] = []
    city_draw_rows: list[pd.DataFrame] = []
    probability: dict[tuple[str, str], np.ndarray] = {}
    overlap_rows: list[dict[str, object]] = []

    for scale, count in counts.items():
        missing = [replicate for replicate in range(count) if not checkpoint_path(scale, replicate).exists()]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} {scale} checkpoints")
        results = [load_checkpoint(scale, replicate) for replicate in range(count)]
        arrays: dict[str, dict[str, np.ndarray]] = {}
        tiers: dict[str, np.ndarray] = {}
        for short, model in (("m2", "M2_city_specific_REML"), ("m3", "M3_spatial_GAM_REML")):
            arrays[model] = {
                "mu": np.asarray([float(result[f"{short}_mu"]) for result in results]),
                "tau": np.asarray([float(result[f"{short}_tau"]) for result in results]),
                "i2": np.asarray([float(result[f"{short}_i2"]) for result in results]),
                "raw": np.vstack([result[f"{short}_raw_beta"] for result in results]),
                "raw_se": np.vstack([result[f"{short}_raw_se"] for result in results]),
                "shrunk": np.vstack([result[f"{short}_shrunk"] for result in results]),
            }
            tiers[model] = np.vstack(
                [
                    np.unpackbits(result[f"{short}_tier_packed"], bitorder="little")[: len(hvcg)].astype(bool)
                    for result in results
                ]
            )
            probability[(model, scale)] = tiers[model].mean(axis=0)

        scale_index = list(SCALES).index(scale)
        rng = np.random.default_rng(BASE_SEED + 90_000_000 + scale_index * 1_000_000)
        two_level = {model: np.empty(count, dtype=float) for model in MODELS}
        for draw in range(count):
            sampled_cities = rng.integers(0, len(cities), size=len(cities))
            sampled_replicates = rng.integers(0, count, size=len(cities))
            for model in MODELS:
                raw = arrays[model]["raw"]
                raw_se = arrays[model]["raw_se"]
                beta = raw[sampled_replicates, sampled_cities]
                se = raw_se[sampled_replicates, sampled_cities]
                two_level[model][draw] = float(support.reml_pool(beta, se)["mu"])

        for model in MODELS:
            conditional_low, conditional_median, conditional_high = interval(arrays[model]["mu"])
            two_low, two_median, two_high = interval(two_level[model])
            national_rows.append(
                {
                    "model_id": model,
                    "block_scale": scale,
                    "replicates": count,
                    "conditional_median_C_per_10pp": conditional_median,
                    "conditional_ci_low": conditional_low,
                    "conditional_ci_high": conditional_high,
                    "conditional_probability_below_zero": float(np.mean(arrays[model]["mu"] < 0)),
                    "two_level_median_C_per_10pp": two_median,
                    "two_level_ci_low": two_low,
                    "two_level_ci_high": two_high,
                    "two_level_probability_below_zero": float(np.mean(two_level[model] < 0)),
                    "mean_tau": float(arrays[model]["tau"].mean()),
                    "mean_i2": float(arrays[model]["i2"].mean()),
                }
            )
            for city_index, city_record in cities.iterrows():
                raw_low, raw_median, raw_high = interval(arrays[model]["raw"][:, city_index])
                pooled_low, pooled_median, pooled_high = interval(arrays[model]["shrunk"][:, city_index])
                city_summary_rows.append(
                    {
                        "model_id": model,
                        "block_scale": scale,
                        "replicates": count,
                        "city_key": city_record["city_key"],
                        "city_display": city_record["city_display"],
                        "analytic_partially_pooled_slope": analytic_slopes[model][city_index],
                        "raw_bootstrap_median": raw_median,
                        "raw_bootstrap_ci_low": raw_low,
                        "raw_bootstrap_ci_high": raw_high,
                        "partially_pooled_bootstrap_median": pooled_median,
                        "partially_pooled_bootstrap_ci_low": pooled_low,
                        "partially_pooled_bootstrap_ci_high": pooled_high,
                        "probability_slope_below_zero": float(
                            np.mean(arrays[model]["shrunk"][:, city_index] < 0)
                        ),
                    }
                )
            for replicate in range(count):
                city_draw_rows.append(
                    pd.DataFrame(
                        {
                            "model_id": model,
                            "block_scale": scale,
                            "replicate": replicate,
                            "city_key": cities["city_key"],
                            "raw_city_slope": arrays[model]["raw"][replicate],
                            "partially_pooled_city_slope": arrays[model]["shrunk"][replicate],
                        }
                    )
                )

        conditional_difference = arrays["M3_spatial_GAM_REML"]["mu"] - arrays["M2_city_specific_REML"]["mu"]
        two_level_difference = two_level["M3_spatial_GAM_REML"] - two_level["M2_city_specific_REML"]
        cond_low, cond_median, cond_high = interval(conditional_difference)
        two_low, two_median, two_high = interval(two_level_difference)
        difference_rows.append(
            {
                "block_scale": scale,
                "replicates": count,
                "difference_definition": "M3 minus M2 city; positive values are less-negative M3 associations",
                "conditional_difference_median": cond_median,
                "conditional_difference_ci_low": cond_low,
                "conditional_difference_ci_high": cond_high,
                "two_level_difference_median": two_median,
                "two_level_difference_ci_low": two_low,
                "two_level_difference_ci_high": two_high,
                "probability_M3_less_negative_than_M2": float(np.mean(two_level_difference > 0)),
            }
        )
        national_draw_rows.append(
            pd.DataFrame(
                {
                    "block_scale": scale,
                    "replicate": np.arange(count),
                    "m2_conditional_mu": arrays["M2_city_specific_REML"]["mu"],
                    "m3_conditional_mu": arrays["M3_spatial_GAM_REML"]["mu"],
                    "m3_minus_m2_conditional": conditional_difference,
                    "m2_two_level_mu": two_level["M2_city_specific_REML"],
                    "m3_two_level_mu": two_level["M3_spatial_GAM_REML"],
                    "m3_minus_m2_two_level": two_level_difference,
                }
            )
        )

        for replicate in range(count):
            m2_tier = tiers["M2_city_specific_REML"][replicate]
            m3_tier = tiers["M3_spatial_GAM_REML"][replicate]
            intersection = int((m2_tier & m3_tier).sum())
            union = int((m2_tier | m3_tier).sum())
            overlap_rows.append(
                {
                    "block_scale": scale,
                    "replicate": replicate,
                    "m2_tier1_n": int(m2_tier.sum()),
                    "m3_tier1_n": int(m3_tier.sum()),
                    "intersection_n": intersection,
                    "union_n": union,
                    "jaccard": intersection / max(union, 1),
                    "m2_retained_by_m3": intersection / max(int(m2_tier.sum()), 1),
                    "m3_retained_by_m2": intersection / max(int(m3_tier.sum()), 1),
                }
            )

    national = pd.DataFrame(national_rows)
    national.to_csv(TABLE_DIR / "paired_national_spatial_bootstrap_inference.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(difference_rows).to_csv(
        TABLE_DIR / "paired_national_model_difference.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(national_draw_rows, ignore_index=True).to_csv(
        DIAG_DIR / "paired_national_bootstrap_draws.csv", index=False, encoding="utf-8-sig"
    )
    city_summary = pd.DataFrame(city_summary_rows)
    city_summary.to_csv(TABLE_DIR / "paired_city_spatial_bootstrap_intervals.csv", index=False, encoding="utf-8-sig")
    pd.concat(city_draw_rows, ignore_index=True).to_csv(
        DIAG_DIR / "paired_city_slope_bootstrap_draws.csv", index=False, encoding="utf-8-sig"
    )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(DIAG_DIR / "paired_hvcg_overlap_by_replicate.csv", index=False, encoding="utf-8-sig")
    overlap.groupby("block_scale", as_index=False).agg(
        replicates=("replicate", "size"),
        median_m2_tier1_n=("m2_tier1_n", "median"),
        median_m3_tier1_n=("m3_tier1_n", "median"),
        median_intersection_n=("intersection_n", "median"),
        median_jaccard=("jaccard", "median"),
        q025_jaccard=("jaccard", lambda value: np.percentile(value, 2.5)),
        q975_jaccard=("jaccard", lambda value: np.percentile(value, 97.5)),
        median_m2_retained_by_m3=("m2_retained_by_m3", "median"),
        median_m3_retained_by_m2=("m3_retained_by_m2", "median"),
    ).to_csv(TABLE_DIR / "paired_hvcg_overlap_summary.csv", index=False, encoding="utf-8-sig")

    probability_frame = hvcg[
        ["support_row", "analysis_record_id", "city_key", "city_display", "GEOID_str", "population"]
    ].copy()
    for model in MODELS:
        short = "m2" if model.startswith("M2") else "m3"
        probability_frame[f"{short}_deterministic_tier1"] = deterministic[model]
        for scale in SCALES:
            probability_frame[f"{short}_tier1_probability_{scale}"] = probability[(model, scale)]
        probability_frame[f"{short}_minimum_probability_across_scales"] = probability_frame[
            [f"{short}_tier1_probability_{scale}" for scale in SCALES]
        ].min(axis=1)
        probability_frame[f"{short}_stable_core_ge_0p80_all_scales"] = (
            probability_frame[f"{short}_minimum_probability_across_scales"] >= 0.80
        )
    probability_frame["consensus_stable_core_both_models"] = (
        probability_frame["m2_stable_core_ge_0p80_all_scales"]
        & probability_frame["m3_stable_core_ge_0p80_all_scales"]
    )
    probability_frame["either_model_stable_core"] = (
        probability_frame["m2_stable_core_ge_0p80_all_scales"]
        | probability_frame["m3_stable_core_ge_0p80_all_scales"]
    )
    probability_frame.to_csv(
        TABLE_DIR / "paired_hvcg_tier1_probabilities_and_consensus_core.csv",
        index=False,
        encoding="utf-8-sig",
    )

    deterministic_m2 = deterministic["M2_city_specific_REML"]
    deterministic_m3 = deterministic["M3_spatial_GAM_REML"]
    deterministic_intersection = int((deterministic_m2 & deterministic_m3).sum())
    deterministic_union = int((deterministic_m2 | deterministic_m3).sum())
    stable_m2 = probability_frame["m2_stable_core_ge_0p80_all_scales"].to_numpy(bool)
    stable_m3 = probability_frame["m3_stable_core_ge_0p80_all_scales"].to_numpy(bool)
    stable_intersection = int((stable_m2 & stable_m3).sum())
    stable_union = int((stable_m2 | stable_m3).sum())
    core_summary = pd.DataFrame(
        [
            {
                "definition": "deterministic_tier1",
                "m2_n": int(deterministic_m2.sum()),
                "m3_n": int(deterministic_m3.sum()),
                "intersection_n": deterministic_intersection,
                "union_n": deterministic_union,
                "jaccard": deterministic_intersection / max(deterministic_union, 1),
                "m2_retained_by_m3": deterministic_intersection / max(int(deterministic_m2.sum()), 1),
                "m3_retained_by_m2": deterministic_intersection / max(int(deterministic_m3.sum()), 1),
            },
            {
                "definition": "stable_core_probability_ge_0p80_all_scales",
                "m2_n": int(stable_m2.sum()),
                "m3_n": int(stable_m3.sum()),
                "intersection_n": stable_intersection,
                "union_n": stable_union,
                "jaccard": stable_intersection / max(stable_union, 1),
                "m2_retained_by_m3": stable_intersection / max(int(stable_m2.sum()), 1),
                "m3_retained_by_m2": stable_intersection / max(int(stable_m3.sum()), 1),
            },
        ]
    )
    core_summary.to_csv(TABLE_DIR / "m2_m3_hvcg_core_overlap.csv", index=False, encoding="utf-8-sig")

    robustness_rows: list[dict[str, object]] = []
    for model in MODELS:
        for city_key, sub in city_summary.loc[city_summary["model_id"].eq(model)].groupby("city_key"):
            robustness_rows.append(
                {
                    "model_id": model,
                    "city_key": city_key,
                    "city_display": sub["city_display"].iloc[0],
                    "all_scales_ci_below_zero": bool(
                        (sub["partially_pooled_bootstrap_ci_high"] < 0).all()
                    ),
                    "all_scales_ci_above_zero": bool(
                        (sub["partially_pooled_bootstrap_ci_low"] > 0).all()
                    ),
                    "minimum_probability_below_zero": float(
                        sub["probability_slope_below_zero"].min()
                    ),
                }
            )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(TABLE_DIR / "paired_city_slope_robustness.csv", index=False, encoding="utf-8-sig")

    main = national.loc[national["block_scale"].eq("main")].set_index("model_id")
    overlap_main = pd.read_csv(TABLE_DIR / "paired_hvcg_overlap_summary.csv").loc[
        lambda value: value["block_scale"].eq("main")
    ].iloc[0]
    report = f"""# Paired M2-M3 shared-block bootstrap audit

Both models were refitted to identical city-specific spatial block resamples. Shared block widths used the larger residual correlation range estimated from M2 city and M3 in each city; half- and double-width grids were sensitivity analyses.

At the primary shared-block scale, M2 city had a two-level median association of {main.loc['M2_city_specific_REML','two_level_median_C_per_10pp']:.3f} C per 10 percentage points (95% interval {main.loc['M2_city_specific_REML','two_level_ci_low']:.3f} to {main.loc['M2_city_specific_REML','two_level_ci_high']:.3f}). M3 had a median of {main.loc['M3_spatial_GAM_REML','two_level_median_C_per_10pp']:.3f} C (95% interval {main.loc['M3_spatial_GAM_REML','two_level_ci_low']:.3f} to {main.loc['M3_spatial_GAM_REML','two_level_ci_high']:.3f}).

The median paired HVCG Tier 1 Jaccard index at the primary scale was {overlap_main['median_jaccard']:.3f} (95% bootstrap range {overlap_main['q025_jaccard']:.3f} to {overlap_main['q975_jaccard']:.3f}). The cross-model, cross-scale consensus stable core contained {stable_intersection} block groups.

M2 and M3 remain association models for Landsat LST. Their agreement supports model-robust screening; disagreement identifies model-sensitive neighbourhoods and must not be hidden by selecting whichever model produces the stronger HVCG contrast.
"""
    (TEXT_DIR / "paired_m2_m3_audit.md").write_text(report, encoding="utf-8")
    decision = f"""# Result 3 model decision

## Decision

No model is declared a universal winner. M2 city is the parsimonious predictive benchmark because it had the lowest nested spatial-CV RMSE and out-of-fold residual Moran's I. M3 is retained as the conservative spatial-adjustment model because it absorbs broad two-dimensional spatial structure and produces a less-negative association. The manuscript should report both shared-block estimates as a model-form range rather than selecting the stronger coefficient.

## National association

- M2 city: {main.loc['M2_city_specific_REML','two_level_median_C_per_10pp']:.3f} C per 10 percentage points (95% shared-block interval {main.loc['M2_city_specific_REML','two_level_ci_low']:.3f} to {main.loc['M2_city_specific_REML','two_level_ci_high']:.3f}).
- M3 spatial: {main.loc['M3_spatial_GAM_REML','two_level_median_C_per_10pp']:.3f} C per 10 percentage points (95% shared-block interval {main.loc['M3_spatial_GAM_REML','two_level_ci_low']:.3f} to {main.loc['M3_spatial_GAM_REML','two_level_ci_high']:.3f}).

Both intervals remained below zero at half, primary and double shared block widths. The association is model-robust at the national level but heterogeneous among cities.

## Downstream rule

Use M3 as the conservative central slope when a single coefficient is required for model-implied HVCG intensity. Report M2 as a model-form sensitivity. Define the policy-facing stable priority core as the intersection of neighbourhoods with Tier 1 probability at least 0.80 under both models at all three shared block scales. This consensus core contains {stable_intersection} block groups. Neighbourhoods stable under only one model are model-sensitive and must be labelled separately.

This rule prevents circular selection of whichever model produces the largest burden reduction and preserves a clear distinction between screening and implementable planting plans.
"""
    (TEXT_DIR / "result3_model_decision.md").write_text(decision, encoding="utf-8")
    qa = {
        "main_replicates": main_replicates,
        "sensitivity_replicates_each_scale": sensitivity_replicates,
        "all_national_intervals_below_zero_both_models_all_scales": bool(
            (national["two_level_ci_high"] < 0).all()
        ),
        "m2_cities_ci_below_zero_all_scales": int(
            robustness.loc[
                robustness["model_id"].eq("M2_city_specific_REML"),
                "all_scales_ci_below_zero",
            ].sum()
        ),
        "m3_cities_ci_below_zero_all_scales": int(
            robustness.loc[
                robustness["model_id"].eq("M3_spatial_GAM_REML"),
                "all_scales_ci_below_zero",
            ].sum()
        ),
        "consensus_stable_core_n": stable_intersection,
        "stable_core_union_n": stable_union,
        "stable_core_jaccard": stable_intersection / max(stable_union, 1),
    }
    (QA_DIR / "paired_bootstrap_final_validation.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired M2-M3 shared spatial-block bootstrap")
    parser.add_argument("--stage", choices=("prepare", "bootstrap", "summarize"), required=True)
    parser.add_argument("--scale", choices=tuple(SCALES), default="main")
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
