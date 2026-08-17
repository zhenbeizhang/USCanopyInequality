from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_ROOT = Path(str(analysis_store_root().joinpath('Result')))
TYPOLOGY_INPUT = (
    RESULT_ROOT
    / "Result4_5_neighbourhood_typology"
    / "data"
    / "block_group_GHV_primary_classification.csv.gz"
)
THERMAL_INPUT = (
    RESULT_ROOT
    / "Result3"
    / "Result3_support_aligned"
    / "data"
    / "support_aligned_model_frame_34566.csv"
)
RAW_ACS_INPUT = (
    RESULT_ROOT
    / "Result4"
    / "versions"
    / "construct_validity_audit"
    / "tables"
    / "result4_hvcg_construct_audited_block_groups.csv"
)
BLOCK_INPUT = (
    RESULT_ROOT
    / "Result3"
    / "Result3_support_aligned"
    / "spatial_block_bootstrap"
    / "data"
    / "spatial_block_assignments.csv"
)
BLOCK_RANGE_INPUT = (
    RESULT_ROOT
    / "Result3"
    / "Result3_support_aligned"
    / "spatial_block_bootstrap"
    / "tables"
    / "city_residual_range_and_block_sizes.csv"
)
OUT = RESULT_ROOT / "Result5" / "versions" / "step2_multicriteria_pareto"

PRIMARY_BUDGET = 0.10
PARETO_LAMBDAS = np.linspace(0.0, 1.0, 21)
EXPECTED_ROWS = 34_566
EXPECTED_CITIES = 50
BASE_SEED = 37_020_001

SIMPLE_RULES = (
    "uniform",
    "canopy_deficit_first",
    "surface_heat_only",
    "social_condition_only",
)
ORACLE_RULES = (
    "population_oracle",
    "heat_context_oracle",
    "social_condition_oracle",
)
PARETO_RULES = tuple(f"pareto_{lam:.2f}" for lam in PARETO_LAMBDAS)
ALL_RULES = SIMPLE_RULES + ORACLE_RULES + PARETO_RULES
GROUP_RULES = (
    "uniform",
    "heat_context_oracle",
    "social_condition_oracle",
    "pareto_0.50",
)
GROUP_COLUMNS = (
    "people_of_color_pct",
    "hispanic_or_latino_pct",
    "black_alone_pct",
    "asian_alone_pct",
    "white_alone_pct",
    "age_65_plus_pct",
)

V_COLUMNS = (
    "V_indicator_poverty_rate_pct",
    "V_indicator_no_vehicle_households_pct",
    "V_indicator_renter_occupied_pct",
    "V_indicator_age_65_plus_pct",
)


@dataclass(frozen=True)
class CityBootstrapResult:
    city_key: str
    metric_array: np.ndarray
    group_array: np.ndarray
    overlap_array: np.ndarray
    row_count_array: np.ndarray
    budget_array: np.ndarray


def normalize_geoid(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def density_strata(values: np.ndarray, n_strata: int = 5) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("Non-finite log population density")
    # Match the frozen Result 4 implementation exactly. In particular, qcut
    # keeps identical density values together rather than splitting bootstrap
    # duplicates across rank bins.
    strata = pd.qcut(
        pd.Series(values), q=n_strata, labels=False, duplicates="drop"
    ).to_numpy()
    if np.isnan(strata).any():
        raise ValueError("Unable to construct density strata")
    return strata.astype(np.int8)


def recompute_gap(canopy: np.ndarray, log_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    strata = density_strata(log_density, 5)
    city_target = float(np.quantile(canopy, 0.75))
    target = np.empty_like(canopy, dtype=float)
    for stratum in np.unique(strata):
        mask = strata == stratum
        values = canopy[mask]
        target[mask] = float(np.quantile(values, 0.75)) if len(values) >= 10 else city_target
    return np.maximum(target - canopy, 0.0), strata


def recompute_social_condition(values: np.ndarray) -> np.ndarray:
    filled = values.astype(float, copy=True)
    for column in range(filled.shape[1]):
        vector = filled[:, column]
        if np.isnan(vector).any():
            median = float(np.nanmedian(vector))
            if not np.isfinite(median):
                median = 0.5
            vector[np.isnan(vector)] = median
            filled[:, column] = vector
    economic = filled[:, 0]
    housing_mobility = filled[:, 1:3].mean(axis=1)
    age = filled[:, 3]
    return np.column_stack((economic, housing_mobility, age)).mean(axis=1)


def hard_allocate(capacity: np.ndarray, budget: float, score: np.ndarray) -> np.ndarray:
    allocation = np.zeros_like(capacity, dtype=float)
    tie = np.arange(len(capacity), dtype=np.int64)
    valid_score = np.where(np.isfinite(score), score, -np.inf)
    order = np.lexsort((tie, -valid_score))
    remaining = float(budget)
    for index in order:
        if remaining <= 1e-8:
            break
        take = min(float(capacity[index]), remaining)
        allocation[index] = take
        remaining -= take
    tolerance = max(1e-5, 1e-10 * max(budget, 1.0))
    if remaining > tolerance:
        raise RuntimeError(f"Budget not exhausted: {remaining}")
    return allocation


def allocation_quantities(
    population: np.ndarray,
    land: np.ndarray,
    heat: np.ndarray,
    social: np.ndarray,
    allocation: np.ndarray,
) -> tuple[float, float, float]:
    delta_canopy = 100.0 * allocation / land
    return (
        float(np.sum(population * heat * delta_canopy)),
        float(np.sum(population * social * delta_canopy)),
        float(np.sum(population * delta_canopy)),
    )


def group_distribution_ratio(
    population: np.ndarray,
    land: np.ndarray,
    groups_pct: np.ndarray,
    allocation: np.ndarray,
) -> np.ndarray:
    delta_canopy = 100.0 * allocation / land
    group_population = population[:, None] * groups_pct / 100.0
    population_share = group_population.sum(axis=0) / population.sum()
    total_increment = float(np.sum(population * delta_canopy))
    increment_share = np.sum(group_population * delta_canopy[:, None], axis=0) / total_increment
    return np.divide(
        increment_share,
        population_share,
        out=np.full_like(population_share, np.nan, dtype=float),
        where=population_share > 0,
    )


def continuous_overlap(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.maximum(first, second).sum())
    return float(np.minimum(first, second).sum() / denominator) if denominator > 0 else np.nan


def evaluate_city_arrays(
    population: np.ndarray,
    land: np.ndarray,
    canopy: np.ndarray,
    log_density: np.ndarray,
    temperature: np.ndarray,
    v_indicators: np.ndarray,
    group_pct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    gap, _ = recompute_gap(canopy, log_density)
    reference_temperature = float(np.average(temperature, weights=population))
    heat = np.maximum(temperature - reference_temperature, 0.0)
    social = recompute_social_condition(v_indicators)
    capacity = land * gap / 100.0
    budget = PRIMARY_BUDGET * float(capacity.sum())
    if budget <= 0:
        raise RuntimeError("Non-positive city budget")

    marginal_heat = 100.0 * population * heat / land
    marginal_social = 100.0 * population * social / land
    marginal_population = 100.0 * population / land

    allocations: dict[str, np.ndarray] = {
        "uniform": capacity * PRIMARY_BUDGET,
        "canopy_deficit_first": hard_allocate(capacity, budget, gap),
        "surface_heat_only": hard_allocate(capacity, budget, heat),
        "social_condition_only": hard_allocate(capacity, budget, social),
        "population_oracle": hard_allocate(capacity, budget, marginal_population),
        "heat_context_oracle": hard_allocate(capacity, budget, marginal_heat),
        "social_condition_oracle": hard_allocate(capacity, budget, marginal_social),
    }

    heat_oracle = allocation_quantities(
        population, land, heat, social, allocations["heat_context_oracle"]
    )[0]
    social_oracle = allocation_quantities(
        population, land, heat, social, allocations["social_condition_oracle"]
    )[1]
    population_oracle = allocation_quantities(
        population, land, heat, social, allocations["population_oracle"]
    )[2]
    if min(heat_oracle, social_oracle, population_oracle) <= 0:
        raise RuntimeError("Non-positive oracle quantity")

    for lam in PARETO_LAMBDAS:
        score = lam * marginal_heat / heat_oracle + (1.0 - lam) * marginal_social / social_oracle
        allocations[f"pareto_{lam:.2f}"] = hard_allocate(capacity, budget, score)

    metrics = np.empty((len(ALL_RULES), 6), dtype=float)
    for rule_index, rule in enumerate(ALL_RULES):
        quantities = allocation_quantities(population, land, heat, social, allocations[rule])
        metrics[rule_index, :] = (
            quantities[0] / heat_oracle,
            quantities[1] / social_oracle,
            quantities[2] / population_oracle,
            quantities[0],
            quantities[1],
            quantities[2],
        )

    group_ratios = np.empty((len(GROUP_RULES), len(GROUP_COLUMNS)), dtype=float)
    for rule_index, rule in enumerate(GROUP_RULES):
        group_ratios[rule_index, :] = group_distribution_ratio(
            population, land, group_pct, allocations[rule]
        )

    overlap = continuous_overlap(
        allocations["heat_context_oracle"], allocations["social_condition_oracle"]
    )
    allocated_area = np.asarray([allocations[rule].sum() for rule in ALL_RULES])
    return metrics, group_ratios, overlap, budget, allocated_area


def load_frame() -> pd.DataFrame:
    typology_columns = [
        "city_key",
        "city_display",
        "GEOID_str",
        "population",
        "land_area_m2",
        "canopy_pct_analysis",
        "people_of_color_pct",
        "hispanic_or_latino_pct",
        "black_alone_pct",
        "asian_alone_pct",
        *V_COLUMNS,
    ]
    frame = pd.read_csv(
        TYPOLOGY_INPUT,
        usecols=typology_columns,
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])

    thermal = pd.read_csv(
        THERMAL_INPUT,
        usecols=[
            "city_key",
            "GEOID_str",
            "mean_LST_C",
            "log_population_density",
            "x_5070_m",
            "y_5070_m",
        ],
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    thermal["GEOID_str"] = normalize_geoid(thermal["GEOID_str"])

    raw_acs = pd.read_csv(
        RAW_ACS_INPUT,
        usecols=["city_key", "GEOID_str", "white_alone_pct", "age_65_plus_pct"],
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    raw_acs["GEOID_str"] = normalize_geoid(raw_acs["GEOID_str"])

    blocks = pd.read_csv(BLOCK_INPUT, dtype={"GEOID_str": str}, low_memory=False)
    blocks["GEOID_str"] = normalize_geoid(blocks["GEOID_str"])
    block_columns = ["city_key", "GEOID_str"] + [f"main_block_origin{i}" for i in range(4)]
    blocks = blocks[block_columns]

    frame = frame.merge(thermal, on=["city_key", "GEOID_str"], how="left", validate="one_to_one")
    frame = frame.merge(raw_acs, on=["city_key", "GEOID_str"], how="left", validate="one_to_one")
    frame = frame.merge(blocks, on=["city_key", "GEOID_str"], how="left", validate="one_to_one")

    required = [
        "population",
        "land_area_m2",
        "canopy_pct_analysis",
        "mean_LST_C",
        "log_population_density",
        *GROUP_COLUMNS,
        *[f"main_block_origin{i}" for i in range(4)],
    ]
    missing = frame[required].isna().sum()
    if int(missing.sum()) > 0:
        raise RuntimeError(f"Missing required inputs:\n{missing[missing > 0]}")
    if len(frame) != EXPECTED_ROWS or frame["city_key"].nunique() != EXPECTED_CITIES:
        raise RuntimeError("Frozen development frame mismatch")
    if (frame["population"] <= 0).any() or (frame["land_area_m2"] <= 0).any():
        raise RuntimeError("Non-positive population or land area")
    return frame.sort_values(["city_key", "GEOID_str"]).reset_index(drop=True)


def city_arrays(city_frame: pd.DataFrame, indices: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    if indices is None:
        indices = np.arange(len(city_frame), dtype=int)
    return (
        city_frame["population"].to_numpy(float)[indices],
        city_frame["land_area_m2"].to_numpy(float)[indices],
        city_frame["canopy_pct_analysis"].to_numpy(float)[indices],
        city_frame["log_population_density"].to_numpy(float)[indices],
        city_frame["mean_LST_C"].to_numpy(float)[indices],
        city_frame[list(V_COLUMNS)].to_numpy(float)[indices, :],
        city_frame[list(GROUP_COLUMNS)].to_numpy(float)[indices, :],
    )


def block_resample_indices(city_frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    origin = int(rng.integers(0, 4))
    labels = city_frame[f"main_block_origin{origin}"].to_numpy(np.int64)
    unique = np.unique(labels)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    counts = pd.Series(sampled).value_counts(sort=False).to_dict()
    pieces = [np.tile(np.flatnonzero(labels == label), count) for label, count in counts.items()]
    return np.concatenate(pieces).astype(np.int64, copy=False)


def bootstrap_city(payload: tuple[str, pd.DataFrame, int, int]) -> CityBootstrapResult:
    city_key, city_frame, n_replicates, seed = payload
    rng = np.random.default_rng(seed)
    metrics = np.empty((n_replicates, len(ALL_RULES), 6), dtype=np.float32)
    groups = np.empty((n_replicates, len(GROUP_RULES), len(GROUP_COLUMNS)), dtype=np.float32)
    overlaps = np.empty(n_replicates, dtype=np.float32)
    row_counts = np.empty(n_replicates, dtype=np.int32)
    budgets = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        indices = block_resample_indices(city_frame, rng)
        evaluated = evaluate_city_arrays(*city_arrays(city_frame, indices))
        metrics[replicate] = evaluated[0]
        groups[replicate] = evaluated[1]
        overlaps[replicate] = evaluated[2]
        budgets[replicate] = evaluated[3]
        row_counts[replicate] = len(indices)
    return CityBootstrapResult(city_key, metrics, groups, overlaps, row_counts, budgets)


def central_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    metric_rows: list[dict] = []
    group_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for city_key, city_frame in frame.groupby("city_key", sort=True):
        evaluated = evaluate_city_arrays(*city_arrays(city_frame))
        metrics, groups, overlap, budget, allocated = evaluated
        arrays[city_key] = metrics
        display = city_frame["city_display"].iloc[0]
        for rule_index, rule in enumerate(ALL_RULES):
            metric_rows.append(
                {
                    "city_key": city_key,
                    "city_display": display,
                    "rule": rule,
                    "budget_fraction": PRIMARY_BUDGET,
                    "budget_m2": budget,
                    "allocated_m2": allocated[rule_index],
                    "heat_context_efficiency": metrics[rule_index, 0],
                    "social_condition_efficiency": metrics[rule_index, 1],
                    "population_efficiency": metrics[rule_index, 2],
                    "heat_context_quantity": metrics[rule_index, 3],
                    "social_condition_quantity": metrics[rule_index, 4],
                    "population_quantity": metrics[rule_index, 5],
                }
            )
        for rule_index, rule in enumerate(GROUP_RULES):
            for group_index, group in enumerate(GROUP_COLUMNS):
                group_rows.append(
                    {
                        "city_key": city_key,
                        "city_display": display,
                        "rule": rule,
                        "group": group,
                        "allocation_share_to_population_share": groups[rule_index, group_index],
                    }
                )
        heat_oracle = metrics[ALL_RULES.index("heat_context_oracle")]
        social_oracle = metrics[ALL_RULES.index("social_condition_oracle")]
        diagnostic_rows.append(
            {
                "city_key": city_key,
                "city_display": display,
                "heat_oracle_social_efficiency": heat_oracle[1],
                "social_oracle_heat_efficiency": social_oracle[0],
                "heat_social_oracle_overlap": overlap,
                "budget_km2": budget / 1e6,
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(diagnostic_rows),
        arrays,
    )


def save_city_bootstrap(result: CityBootstrapResult, directory: Path) -> Path:
    path = directory / f"{result.city_key}.npz"
    np.savez_compressed(
        path,
        metrics=result.metric_array,
        groups=result.group_array,
        overlap=result.overlap_array,
        row_counts=result.row_count_array,
        budgets=result.budget_array,
    )
    return path


def run_city_bootstraps(
    frame: pd.DataFrame,
    n_replicates: int,
    workers: int,
    directory: Path,
) -> dict[str, CityBootstrapResult]:
    directory.mkdir(parents=True, exist_ok=True)
    payloads = []
    for city_index, (city_key, city_frame) in enumerate(frame.groupby("city_key", sort=True)):
        payloads.append(
            (
                city_key,
                city_frame.reset_index(drop=True),
                n_replicates,
                BASE_SEED + 10_000 * city_index,
            )
        )
    results: dict[str, CityBootstrapResult] = {}
    if workers == 1:
        for payload in payloads:
            result = bootstrap_city(payload)
            save_city_bootstrap(result, directory)
            results[result.city_key] = result
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(bootstrap_city, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                save_city_bootstrap(result, directory)
                results[result.city_key] = result
                print(f"Completed {result.city_key}", flush=True)
    return results


def two_level_draws(
    city_results: dict[str, CityBootstrapResult],
    n_draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    city_keys = sorted(city_results)
    n_cities = len(city_keys)
    n_city_replicates = next(iter(city_results.values())).metric_array.shape[0]
    metric_draws = np.empty((n_draws, len(ALL_RULES), 3), dtype=np.float32)
    group_draws = np.empty((n_draws, len(GROUP_RULES), len(GROUP_COLUMNS)), dtype=np.float32)
    overlap_draws = np.empty(n_draws, dtype=np.float32)
    rng = np.random.default_rng(BASE_SEED + 999_999)
    for draw in range(n_draws):
        sampled_cities = rng.integers(0, n_cities, size=n_cities)
        sampled_replicates = rng.integers(0, n_city_replicates, size=n_cities)
        metric_stack = []
        group_stack = []
        overlap_stack = []
        for city_position, replicate in zip(sampled_cities, sampled_replicates, strict=True):
            result = city_results[city_keys[int(city_position)]]
            metric_stack.append(result.metric_array[int(replicate), :, :3])
            group_stack.append(result.group_array[int(replicate)])
            overlap_stack.append(result.overlap_array[int(replicate)])
        metric_draws[draw] = np.nanmean(np.stack(metric_stack), axis=0)
        group_draws[draw] = np.nanmean(np.stack(group_stack), axis=0)
        overlap_draws[draw] = float(np.nanmean(overlap_stack))
    return metric_draws, group_draws, overlap_draws


def interval_table(
    central_metrics: pd.DataFrame,
    metric_draws: np.ndarray,
) -> pd.DataFrame:
    central = central_metrics.groupby("rule", sort=False)[
        ["heat_context_efficiency", "social_condition_efficiency", "population_efficiency"]
    ].mean()
    rows = []
    metric_names = (
        "heat_context_efficiency",
        "social_condition_efficiency",
        "population_efficiency",
    )
    for rule_index, rule in enumerate(ALL_RULES):
        for metric_index, metric in enumerate(metric_names):
            values = metric_draws[:, rule_index, metric_index]
            rows.append(
                {
                    "rule": rule,
                    "metric": metric,
                    "equal_city_central": central.loc[rule, metric],
                    "two_level_median": float(np.nanmedian(values)),
                    "two_level_ci_low": float(np.nanquantile(values, 0.025)),
                    "two_level_ci_high": float(np.nanquantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def group_interval_table(central_groups: pd.DataFrame, group_draws: np.ndarray) -> pd.DataFrame:
    central = central_groups.groupby(["rule", "group"], sort=False)[
        "allocation_share_to_population_share"
    ].mean()
    rows = []
    for rule_index, rule in enumerate(GROUP_RULES):
        for group_index, group in enumerate(GROUP_COLUMNS):
            values = group_draws[:, rule_index, group_index]
            rows.append(
                {
                    "rule": rule,
                    "group": group,
                    "equal_city_central": central.loc[(rule, group)],
                    "two_level_median": float(np.nanmedian(values)),
                    "two_level_ci_low": float(np.nanquantile(values, 0.025)),
                    "two_level_ci_high": float(np.nanquantile(values, 0.975)),
                    "interval_excludes_parity": bool(
                        np.nanquantile(values, 0.025) > 1.0 or np.nanquantile(values, 0.975) < 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def monte_carlo_endpoint_audit(metric_draws: np.ndarray, batches: int = 10) -> dict[str, float]:
    if len(metric_draws) < batches * 20:
        return {"batches": batches, "max_ci_endpoint_sd": math.nan}
    chunks = np.array_split(metric_draws, batches)
    endpoints = []
    for chunk in chunks:
        endpoints.append(
            np.stack(
                (
                    np.nanquantile(chunk, 0.025, axis=0),
                    np.nanquantile(chunk, 0.975, axis=0),
                )
            )
        )
    endpoint_sd = np.nanstd(np.stack(endpoints), axis=0, ddof=1)
    return {
        "batches": batches,
        "max_ci_endpoint_sd": float(np.nanmax(endpoint_sd)),
        "median_ci_endpoint_sd": float(np.nanmedian(endpoint_sd)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Result 5 multiobjective allocation analysis")
    parser.add_argument("--city-replicates", type=int, default=1000)
    parser.add_argument("--outer-draws", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    for subdir in (
        "bootstrap_city",
        "data",
        "figures",
        "methods",
        "qa",
        "source_data",
        "tables",
        "text",
    ):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)

    frame = load_frame()
    central_metrics, central_groups, city_diagnostics, _ = central_analysis(frame)
    central_metrics.to_csv(OUT / "tables" / "01_city_rule_outcomes.csv", index=False)
    central_groups.to_csv(OUT / "tables" / "02_city_group_distribution_ratios.csv", index=False)
    city_diagnostics.to_csv(OUT / "tables" / "03_city_tradeoff_diagnostics.csv", index=False)

    city_results = run_city_bootstraps(
        frame,
        n_replicates=args.city_replicates,
        workers=args.workers,
        directory=OUT / "bootstrap_city",
    )
    metric_draws, group_draws, overlap_draws = two_level_draws(city_results, args.outer_draws)
    np.savez_compressed(
        OUT / "data" / "two_level_draws.npz",
        metrics=metric_draws,
        groups=group_draws,
        overlap=overlap_draws,
        rules=np.asarray(ALL_RULES),
        group_rules=np.asarray(GROUP_RULES),
        group_columns=np.asarray(GROUP_COLUMNS),
    )

    intervals = interval_table(central_metrics, metric_draws)
    group_intervals = group_interval_table(central_groups, group_draws)
    intervals.to_csv(OUT / "tables" / "04_two_level_rule_intervals.csv", index=False)
    group_intervals.to_csv(OUT / "tables" / "05_two_level_group_intervals.csv", index=False)

    city_interval_rows = []
    for city_key, result in sorted(city_results.items()):
        for rule_index, rule in enumerate(ALL_RULES):
            for metric_index, metric in enumerate(
                ("heat_context_efficiency", "social_condition_efficiency", "population_efficiency")
            ):
                values = result.metric_array[:, rule_index, metric_index]
                city_interval_rows.append(
                    {
                        "city_key": city_key,
                        "rule": rule,
                        "metric": metric,
                        "median": float(np.nanmedian(values)),
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                    }
                )
    pd.DataFrame(city_interval_rows).to_csv(
        OUT / "tables" / "06_city_spatial_block_intervals.csv", index=False
    )

    pareto = intervals.loc[intervals["rule"].str.startswith("pareto_")].copy()
    pareto["lambda_heat"] = pareto["rule"].str.split("_").str[1].astype(float)
    pareto.to_csv(OUT / "source_data" / "Fig5a_pareto_frontier.csv", index=False)
    city_diagnostics.to_csv(OUT / "source_data" / "Fig5b_city_opportunity_cost.csv", index=False)
    group_intervals.to_csv(OUT / "source_data" / "Fig5c_group_distribution.csv", index=False)

    budget_error = np.max(
        np.abs(central_metrics["allocated_m2"] - central_metrics["budget_m2"])
        / central_metrics["budget_m2"]
    )
    endpoint_zero = central_metrics.loc[central_metrics["rule"].eq("pareto_0.00")].sort_values("city_key")
    social_oracle = central_metrics.loc[
        central_metrics["rule"].eq("social_condition_oracle")
    ].sort_values("city_key")
    endpoint_one = central_metrics.loc[central_metrics["rule"].eq("pareto_1.00")].sort_values("city_key")
    heat_oracle = central_metrics.loc[central_metrics["rule"].eq("heat_context_oracle")].sort_values("city_key")

    qa = {
        "status": "PASS",
        "input_rows": len(frame),
        "cities": int(frame["city_key"].nunique()),
        "city_spatial_block_replicates": args.city_replicates,
        "two_level_outer_draws": args.outer_draws,
        "primary_budget_fraction": PRIMARY_BUDGET,
        "primary_scope": "positive density-peer q75 canopy deficit",
        "primary_heat_reference": "population-weighted mean LST among analysed residential units",
        "canopy_lst_coefficient_used": False,
        "max_central_relative_budget_error": float(budget_error),
        "pareto_zero_matches_social_oracle": bool(
            np.allclose(
                endpoint_zero[["heat_context_efficiency", "social_condition_efficiency"]],
                social_oracle[["heat_context_efficiency", "social_condition_efficiency"]],
            )
        ),
        "pareto_one_matches_heat_oracle": bool(
            np.allclose(
                endpoint_one[["heat_context_efficiency", "social_condition_efficiency"]],
                heat_oracle[["heat_context_efficiency", "social_condition_efficiency"]],
            )
        ),
        "median_heat_social_oracle_overlap": float(city_diagnostics["heat_social_oracle_overlap"].median()),
        "monte_carlo_endpoint_audit": monte_carlo_endpoint_audit(metric_draws),
        "input_hashes": {
            str(path): sha256(path)
            for path in (TYPOLOGY_INPUT, THERMAL_INPUT, RAW_ACS_INPUT, BLOCK_INPUT, BLOCK_RANGE_INPUT)
        },
    }
    qa["status"] = "PASS" if all(
        (
            qa["cities"] == EXPECTED_CITIES,
            qa["input_rows"] == EXPECTED_ROWS,
            qa["max_central_relative_budget_error"] < 1e-10,
            qa["pareto_zero_matches_social_oracle"],
            qa["pareto_one_matches_heat_oracle"],
            not qa["canopy_lst_coefficient_used"],
        )
    ) else "FAIL"
    (OUT / "qa" / "step2_analysis_audit.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, indent=2), flush=True)
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
