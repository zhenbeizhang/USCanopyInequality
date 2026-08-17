from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = analysis_store_root()
R4 = PROJECT / "Result" / "Result4" / "versions" / "joint_context_boundary_spatial_audit"
R5 = PROJECT / "Result" / "Result5"
LEGACY = R5 / "versions" / "step2_multicriteria_pareto"
OUT = R5 / "versions" / "weighting_heterogeneity_audit"
FRAME_PATH = R4 / "data" / "fully_contained_primary_GHV27.csv.gz"
BLOCK_PATH = (
    PROJECT
    / "Result"
    / "Result3"
    / "Result3_support_aligned"
    / "spatial_block_bootstrap"
    / "data"
    / "spatial_block_assignments.csv"
)
LEGACY_CITY = LEGACY / "tables" / "01_city_rule_outcomes.csv"

PRIMARY_BUDGET = 0.10
SEED = 20260804
EXPECTED_ROWS = 34_566
EXPECTED_CITIES = 50
METRICS = (
    "population_weighted_overlap",
    "equal_neighbourhood_overlap",
    "overlap_difference_equal_minus_population",
    "population_heat_retained_by_social",
    "population_social_retained_by_heat",
    "equal_heat_retained_by_social",
    "equal_social_retained_by_heat",
    "heat_selection_overlap_between_weightings",
    "social_selection_overlap_between_weightings",
)


@dataclass(frozen=True)
class CityBootstrap:
    city_key: str
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-replicates", type=int, default=500)
    parser.add_argument("--outer-draws", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def ensure_dirs() -> None:
    for name in ("tables", "bootstrap_city", "diagnostics", "qa", "methods", "text", "source_data"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_geoid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def hard_allocate(capacity: np.ndarray, budget: float, score: np.ndarray) -> np.ndarray:
    allocation = np.zeros_like(capacity, dtype=float)
    order = np.lexsort((np.arange(len(capacity), dtype=np.int64), -np.where(np.isfinite(score), score, -np.inf)))
    ordered_capacity = capacity[order]
    cumulative = np.cumsum(ordered_capacity)
    exhausted_at = int(np.searchsorted(cumulative, budget, side="right"))
    if exhausted_at > 0:
        allocation[order[:exhausted_at]] = ordered_capacity[:exhausted_at]
        allocated = float(cumulative[exhausted_at - 1])
    else:
        allocated = 0.0
    if exhausted_at < len(order):
        allocation[order[exhausted_at]] = max(0.0, float(budget) - allocated)
    remaining = float(budget) - float(allocation.sum())
    if remaining > max(1e-5, 1e-10 * max(budget, 1.0)):
        raise RuntimeError(f"Budget not exhausted: {remaining}")
    return allocation


def continuous_overlap(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.maximum(a, b).sum())
    return float(np.minimum(a, b).sum() / denominator) if denominator > 0 else float("nan")


def quantities(
    population: np.ndarray,
    land: np.ndarray,
    heat: np.ndarray,
    social: np.ndarray,
    allocation: np.ndarray,
) -> tuple[float, float, float, float]:
    delta = 100.0 * allocation / land
    return (
        float(np.sum(population * heat * delta)),
        float(np.sum(population * social * delta)),
        float(np.sum(heat * delta)),
        float(np.sum(social * delta)),
    )


def evaluate_arrays(
    population: np.ndarray,
    land: np.ndarray,
    gap: np.ndarray,
    heat: np.ndarray,
    social: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    capacity = land * gap / 100.0
    budget = PRIMARY_BUDGET * float(capacity.sum())
    if budget <= 0:
        raise RuntimeError("Non-positive analytical budget")

    scores = {
        "population_heat": 100.0 * population * heat / land,
        "population_social": 100.0 * population * social / land,
        "equal_heat": 100.0 * heat / land,
        "equal_social": 100.0 * social / land,
    }
    allocations = {name: hard_allocate(capacity, budget, score) for name, score in scores.items()}
    q = {name: quantities(population, land, heat, social, allocation) for name, allocation in allocations.items()}

    pop_heat_max = q["population_heat"][0]
    pop_social_max = q["population_social"][1]
    equal_heat_max = q["equal_heat"][2]
    equal_social_max = q["equal_social"][3]
    if min(pop_heat_max, pop_social_max, equal_heat_max, equal_social_max) <= 0:
        raise RuntimeError("Non-positive objective maximum")

    values = np.asarray(
        [
            continuous_overlap(allocations["population_heat"], allocations["population_social"]),
            continuous_overlap(allocations["equal_heat"], allocations["equal_social"]),
            0.0,
            q["population_social"][0] / pop_heat_max,
            q["population_heat"][1] / pop_social_max,
            q["equal_social"][2] / equal_heat_max,
            q["equal_heat"][3] / equal_social_max,
            continuous_overlap(allocations["population_heat"], allocations["equal_heat"]),
            continuous_overlap(allocations["population_social"], allocations["equal_social"]),
        ],
        dtype=float,
    )
    values[2] = values[1] - values[0]
    return values, allocations


def weighted_correlation(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[valid]
    y = y[valid]
    w = w[valid]
    if len(x) < 3:
        return float("nan")
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    covariance = np.average((x - x_mean) * (y - y_mean), weights=w)
    variance_x = np.average((x - x_mean) ** 2, weights=w)
    variance_y = np.average((y - y_mean) ** 2, weights=w)
    if variance_x <= 0 or variance_y <= 0:
        return float("nan")
    return float(covariance / math.sqrt(variance_x * variance_y))


def weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return weighted_correlation(stats.rankdata(x, method="average"), stats.rankdata(y, method="average"), w)


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(FRAME_PATH, dtype={"GEOID_str": str}, low_memory=False)
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    blocks = pd.read_csv(BLOCK_PATH, dtype={"GEOID_str": str}, low_memory=False)
    blocks["GEOID_str"] = normalize_geoid(blocks["GEOID_str"])
    block_columns = ["city_key", "GEOID_str", *[f"main_block_origin{i}" for i in range(4)]]
    frame = frame.merge(blocks[block_columns], on=["city_key", "GEOID_str"], validate="one_to_one")
    frame = frame.sort_values(["city_key", "GEOID_str"]).reset_index(drop=True)
    required = [
        "population",
        "land_area_analysis_m2",
        "G",
        "H",
        "V",
        "context_id",
        *[f"main_block_origin{i}" for i in range(4)],
    ]
    if len(frame) != EXPECTED_ROWS or frame["city_key"].nunique() != EXPECTED_CITIES:
        raise RuntimeError("Frozen Result 5 frame mismatch")
    if frame[required].isna().any().any():
        raise RuntimeError("Missing values in Result 5 audit frame")
    if (frame["population"] <= 0).any() or (frame["land_area_analysis_m2"] <= 0).any():
        raise RuntimeError("Non-positive population or land area")
    return frame


def arrays(frame: pd.DataFrame, index: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    if index is None:
        index = np.arange(len(frame))
    return (
        frame["population"].to_numpy(float)[index],
        frame["land_area_analysis_m2"].to_numpy(float)[index],
        frame["G"].to_numpy(float)[index],
        frame["H"].to_numpy(float)[index],
        frame["V"].to_numpy(float)[index],
    )


def prepare_block_groups(frame: pd.DataFrame) -> tuple[tuple[np.ndarray, ...], ...]:
    prepared = []
    for origin in range(4):
        labels = frame[f"main_block_origin{origin}"].to_numpy(np.int64)
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        split_at = np.flatnonzero(np.diff(sorted_labels)) + 1
        prepared.append(tuple(np.split(order, split_at)))
    return tuple(prepared)


def block_resample(groups_by_origin: tuple[tuple[np.ndarray, ...], ...], rng: np.random.Generator) -> np.ndarray:
    groups = groups_by_origin[int(rng.integers(0, len(groups_by_origin)))]
    sampled = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[int(index)] for index in sampled]).astype(np.int64)


def city_bootstrap(payload: tuple[str, pd.DataFrame, int, int]) -> CityBootstrap:
    city_key, frame, replicates, seed = payload
    rng = np.random.default_rng(seed)
    base_arrays = arrays(frame)
    groups_by_origin = prepare_block_groups(frame)
    result = np.empty((replicates, len(METRICS)), dtype=np.float32)
    for replicate in range(replicates):
        index = block_resample(groups_by_origin, rng)
        result[replicate] = evaluate_arrays(*(values[index] for values in base_arrays))[0]
    return CityBootstrap(city_key, result)


def city_descriptors(frame: pd.DataFrame, values: np.ndarray, allocations: dict[str, np.ndarray]) -> dict[str, object]:
    population, land, gap, heat, social = arrays(frame)
    capacity = land * gap / 100.0
    positive = capacity > 0
    weights = capacity[positive]
    shares = weights / weights.sum()
    population_score = population / land
    heat_score = population * heat / land
    social_score = population * social / land
    return {
        "city_key": frame["city_key"].iloc[0],
        "city_display": frame["city_display"].iloc[0],
        "n_block_groups": len(frame),
        "positive_capacity_units": int(positive.sum()),
        "capacity_km2": float(capacity.sum() / 1e6),
        "budget_km2": float(PRIMARY_BUDGET * capacity.sum() / 1e6),
        "capacity_hhi": float(np.sum(shares**2)),
        "joint_context_capacity_share": float(capacity[frame["context_id"].eq("A").to_numpy()].sum() / capacity.sum()),
        "capacity_weighted_heat_social_spearman": weighted_spearman(heat[positive], social[positive], weights),
        "capacity_weighted_population_score_spearman_with_heat": weighted_spearman(population_score[positive], heat[positive], weights),
        "capacity_weighted_population_score_spearman_with_social": weighted_spearman(population_score[positive], social[positive], weights),
        "capacity_weighted_marginal_score_spearman": weighted_spearman(heat_score[positive], social_score[positive], weights),
        **{metric: float(values[index]) for index, metric in enumerate(METRICS)},
    }


def central(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, city in frame.groupby("city_key", sort=True):
        values, allocation = evaluate_arrays(*arrays(city))
        rows.append(city_descriptors(city, values, allocation))
    return pd.DataFrame(rows)


def two_level_draws(results: dict[str, CityBootstrap], draws: int) -> np.ndarray:
    keys = sorted(results)
    n_cities = len(keys)
    n_replicates = next(iter(results.values())).values.shape[0]
    rng = np.random.default_rng(SEED + 999_999)
    out = np.empty((draws, len(METRICS)), dtype=np.float32)
    for draw in range(draws):
        cities = rng.integers(0, n_cities, size=n_cities)
        replicates = rng.integers(0, n_replicates, size=n_cities)
        out[draw] = np.nanmean(
            np.stack([results[keys[int(city)]].values[int(rep)] for city, rep in zip(cities, replicates, strict=True)]),
            axis=0,
        )
    return out


def correlation_audit(city: pd.DataFrame, draws: int = 20_000) -> pd.DataFrame:
    predictors = (
        "capacity_weighted_marginal_score_spearman",
        "capacity_weighted_heat_social_spearman",
        "joint_context_capacity_share",
        "capacity_hhi",
    )
    outcome = city["population_weighted_overlap"].to_numpy(float)
    rng = np.random.default_rng(SEED + 12345)
    bootstrap_index = rng.integers(0, len(city), size=(draws, len(city)))
    ranked_outcome = stats.rankdata(outcome, method="average")
    sampled_outcome = ranked_outcome[bootstrap_index]
    sampled_outcome = sampled_outcome - sampled_outcome.mean(axis=1, keepdims=True)
    sampled_outcome_ss = np.sum(sampled_outcome**2, axis=1)
    rows = []
    for predictor in predictors:
        x = city[predictor].to_numpy(float)
        point = float(stats.spearmanr(x, outcome).statistic)
        ranked_x = stats.rankdata(x, method="average")
        sampled_x = ranked_x[bootstrap_index]
        sampled_x = sampled_x - sampled_x.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(sampled_x**2, axis=1) * sampled_outcome_ss)
        boot = np.divide(
            np.sum(sampled_x * sampled_outcome, axis=1),
            denominator,
            out=np.full(draws, np.nan, dtype=float),
            where=denominator > 0,
        )
        rows.append(
            {
                "outcome": "population_weighted_overlap",
                "predictor": predictor,
                "spearman_rho": point,
                "city_bootstrap_ci_low": float(np.nanquantile(boot, 0.025)),
                "city_bootstrap_ci_high": float(np.nanquantile(boot, 0.975)),
                "interpretation": "descriptive city-level association; not a causal mechanism estimate",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    frame = load_frame()
    city = central(frame)

    legacy = pd.read_csv(LEGACY_CITY)
    legacy_diag = (
        legacy.loc[legacy["rule"].isin(["heat_context_oracle", "social_condition_oracle"])]
        .pivot(index="city_key", columns="rule", values=["heat_context_efficiency", "social_condition_efficiency"])
    )
    max_reproduction_error = max(
        float(np.max(np.abs(city.set_index("city_key")["population_heat_retained_by_social"] - legacy_diag[("heat_context_efficiency", "social_condition_oracle")]))),
        float(np.max(np.abs(city.set_index("city_key")["population_social_retained_by_heat"] - legacy_diag[("social_condition_efficiency", "heat_context_oracle")]))),
    )
    if max_reproduction_error > 1e-10:
        raise RuntimeError(f"Primary Result 5 reproduction failed: {max_reproduction_error}")

    payloads = [
        (city_key, sub.reset_index(drop=True), args.city_replicates, SEED + 10_000 * index)
        for index, (city_key, sub) in enumerate(frame.groupby("city_key", sort=True))
    ]
    results = {}
    if args.workers == 1:
        for payload in payloads:
            result = city_bootstrap(payload)
            results[result.city_key] = result
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(city_bootstrap, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                results[result.city_key] = result
                np.savez_compressed(OUT / "bootstrap_city" / f"{result.city_key}.npz", values=result.values)
                print(f"completed {result.city_key}", flush=True)

    draws = two_level_draws(results, args.outer_draws)
    central_mean = city[list(METRICS)].mean()
    interval_rows = []
    for index, metric in enumerate(METRICS):
        interval_rows.append(
            {
                "metric": metric,
                "equal_city_deterministic_point": float(central_mean[metric]),
                "two_level_bootstrap_median": float(np.nanmedian(draws[:, index])),
                "two_level_ci_low": float(np.nanquantile(draws[:, index], 0.025)),
                "two_level_ci_high": float(np.nanquantile(draws[:, index], 0.975)),
            }
        )
    interval = pd.DataFrame(interval_rows)
    correlations = correlation_audit(city)

    city.to_csv(OUT / "tables" / "result5_weighting_city_diagnostics.csv", index=False, encoding="utf-8-sig")
    interval.to_csv(OUT / "tables" / "result5_weighting_two_level_intervals.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(OUT / "tables" / "result5_city_alignment_correlations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(draws, columns=METRICS).to_csv(
        OUT / "diagnostics" / "result5_weighting_two_level_draws.csv.gz", index=False, compression="gzip"
    )

    row = interval.set_index("metric")
    audit = {
        "status": "PASS",
        "analysis_units": len(frame),
        "cities": int(frame["city_key"].nunique()),
        "city_spatial_block_replicates": args.city_replicates,
        "two_level_outer_draws": args.outer_draws,
        "primary_reproduction_max_abs_error": max_reproduction_error,
        "population_weighted_overlap": float(row.loc["population_weighted_overlap", "equal_city_deterministic_point"]),
        "equal_neighbourhood_overlap": float(row.loc["equal_neighbourhood_overlap", "equal_city_deterministic_point"]),
        "overlap_difference_equal_minus_population": float(row.loc["overlap_difference_equal_minus_population", "equal_city_deterministic_point"]),
        "overlap_difference_interval": [
            float(row.loc["overlap_difference_equal_minus_population", "two_level_ci_low"]),
            float(row.loc["overlap_difference_equal_minus_population", "two_level_ci_high"]),
        ],
        "interpretation": (
            "The equal-neighbourhood sensitivity isolates the shared population multiplier while holding G, H, V, capacity, budget and ranking algorithm fixed. "
            "City descriptors are exploratory explanations of heterogeneity, not causal mechanism estimates."
        ),
        "input_hashes": {
            str(FRAME_PATH): sha256(FRAME_PATH),
            str(BLOCK_PATH): sha256(BLOCK_PATH),
            str(LEGACY_CITY): sha256(LEGACY_CITY),
        },
    }
    (OUT / "qa" / "result5_weighting_heterogeneity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "text" / "result5_interpretation.txt").write_text(
        f"With population in both objectives, the heat- and social-oracle allocations overlapped by {audit['population_weighted_overlap']:.3f} on average across cities. "
        f"Removing the common population multiplier while retaining the same G, H, V, capacity and budgets changed overlap to {audit['equal_neighbourhood_overlap']:.3f}; "
        f"the paired change was {audit['overlap_difference_equal_minus_population']:+.3f} "
        f"(95% two-level interval {audit['overlap_difference_interval'][0]:+.3f} to {audit['overlap_difference_interval'][1]:+.3f}). "
        "The comparison is a diagnostic allocation experiment, not a parcel-level planting plan or an estimate of realized cooling.\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OUT / "methods" / Path(__file__).name)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
