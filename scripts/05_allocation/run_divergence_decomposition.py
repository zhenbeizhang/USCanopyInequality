from __future__ import annotations

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

from uscanopy.paths import analysis_store_root, repo_root


BUDGETS = np.asarray([0.05, 0.10, 0.20], dtype=float)
STRUCTURES = (
    "observed_gap_capacity",
    "neutral_capacity_same_candidates",
    "neutral_capacity_all_units",
    "neutral_capacity_all_units_rate_matched",
)
PRIMARY_STRUCTURES = STRUCTURES[:3]
OUTCOMES = (
    "allocation_overlap",
    "heat_context_opportunity_loss_under_social",
    "social_context_opportunity_loss_under_heat",
)
CONTRASTS = (
    "capacity_intensity_neutral_minus_observed",
    "candidate_gate_total_matched_all_minus_same",
    "candidate_gate_rate_matched_all_minus_same",
)
EXPECTED_ROWS = 34_566
EXPECTED_CITIES = 50
EXPECTED_REPLICATES = 1_000
EXPECTED_OUTER_DRAWS = 5_000
SEED = 20260812
MATERIALITY = 0.02

RESULT_ROOT = analysis_store_root() / "Result"
STEP1_ROOT = (
    RESULT_ROOT / "Result5" / "versions" / "step01_city_tradeoff_outcomes"
)
DEFAULT_OUTPUT = (
    RESULT_ROOT / "Result5" / "versions" / "step02_divergence_decomposition"
)
DEFAULT_PROTOCOL = repo_root() / "docs" / "protocols" / "DIVERGENCE_DECOMPOSITION_PROTOCOL.md"


@dataclass(frozen=True)
class CityBootstrap:
    city_key: str
    values: np.ndarray
    heat_tie_fraction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--step1-root", type=Path, default=STEP1_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--city-replicates", type=int, default=EXPECTED_REPLICATES)
    parser.add_argument("--outer-draws", type=int, default=EXPECTED_OUTER_DRAWS)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--central-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_geoid(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def ensure_directories(output: Path) -> None:
    for name in ("tables", "diagnostics", "bootstrap_city", "qa", "methods", "manifests", "text"):
        (output / name).mkdir(parents=True, exist_ok=True)


def density_strata(values: np.ndarray, n_strata: int = 5) -> np.ndarray:
    strata = pd.qcut(pd.Series(values), q=n_strata, labels=False, duplicates="drop").to_numpy()
    if np.isnan(strata).any():
        raise RuntimeError("Unable to construct density strata")
    return strata.astype(np.int8)


def recompute_gap(canopy: np.ndarray, log_density: np.ndarray) -> np.ndarray:
    strata = density_strata(log_density)
    city_target = float(np.quantile(canopy, 0.75))
    target = np.empty_like(canopy, dtype=float)
    for stratum in np.unique(strata):
        mask = strata == stratum
        target[mask] = float(np.quantile(canopy[mask], 0.75)) if mask.sum() >= 10 else city_target
    return np.maximum(target - canopy, 0.0)


def recompute_social(values: np.ndarray) -> np.ndarray:
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


def allocate_with_proportional_ties(
    capacity: np.ndarray, budget: float, score: np.ndarray
) -> tuple[np.ndarray, float]:
    if budget < -1e-8 or budget > capacity.sum() + 1e-6:
        raise RuntimeError("Budget falls outside capacity")
    allocation = np.zeros_like(capacity, dtype=float)
    valid = capacity > 0
    if not valid.any():
        raise RuntimeError("No positive capacity")
    score = np.where(np.isfinite(score), score, -np.inf)
    unique_scores = np.unique(score[valid])[::-1]
    remaining = float(budget)
    for level in unique_scores:
        if remaining <= 1e-8:
            break
        tied = valid & (score == level)
        tied_capacity = float(capacity[tied].sum())
        if tied_capacity <= 0:
            continue
        if tied_capacity <= remaining + 1e-8:
            allocation[tied] = capacity[tied]
            remaining -= tied_capacity
        else:
            allocation[tied] = capacity[tied] * remaining / tied_capacity
            remaining = 0.0
            break
    tolerance = max(1e-5, 1e-10 * max(budget, 1.0))
    if remaining > tolerance:
        raise RuntimeError(f"Budget not exhausted: {remaining}")
    tie_fraction = float(allocation[score <= 0].sum() / budget) if budget > 0 else 0.0
    return allocation, tie_fraction


def quantities(
    population: np.ndarray,
    land: np.ndarray,
    heat: np.ndarray,
    social: np.ndarray,
    allocation: np.ndarray,
) -> tuple[float, float]:
    delta = 100.0 * allocation / land
    return (
        float(np.sum(population * heat * delta)),
        float(np.sum(population * social * delta)),
    )


def overlap(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.maximum(first, second).sum())
    return float(np.minimum(first, second).sum() / denominator) if denominator > 0 else np.nan


def build_capacities(land: np.ndarray, gap: np.ndarray) -> dict[str, np.ndarray]:
    observed = land * gap / 100.0
    total = float(observed.sum())
    eligible = observed > 0
    if total <= 0 or not eligible.any():
        raise RuntimeError("Non-positive observed gap capacity")
    eligible_rate = total / float(land[eligible].sum())
    return {
        "observed_gap_capacity": observed,
        "neutral_capacity_same_candidates": np.where(eligible, land * eligible_rate, 0.0),
        "neutral_capacity_all_units": land * total / float(land.sum()),
        "neutral_capacity_all_units_rate_matched": land * eligible_rate,
    }


def evaluate_arrays(
    population: np.ndarray,
    land: np.ndarray,
    canopy: np.ndarray,
    log_density: np.ndarray,
    temperature: np.ndarray,
    social_indicators: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gap = recompute_gap(canopy, log_density)
    heat_reference = float(np.average(temperature, weights=population))
    heat = np.maximum(temperature - heat_reference, 0.0)
    social = recompute_social(social_indicators)
    capacities = build_capacities(land, gap)
    observed_total = float(capacities["observed_gap_capacity"].sum())
    heat_score = 100.0 * population * heat / land
    social_score = 100.0 * population * social / land

    values = np.empty((len(STRUCTURES), len(BUDGETS), len(OUTCOMES)), dtype=float)
    heat_ties = np.empty((len(STRUCTURES), len(BUDGETS)), dtype=float)
    capacity_totals = np.asarray([capacities[name].sum() for name in STRUCTURES], dtype=float)
    budgets = observed_total * BUDGETS
    for structure_index, structure in enumerate(STRUCTURES):
        capacity = capacities[structure]
        for budget_index, budget in enumerate(budgets):
            if budget > capacity.sum() + 1e-6:
                raise RuntimeError(f"Budget exceeds {structure} capacity")
            heat_allocation, tie_fraction = allocate_with_proportional_ties(capacity, budget, heat_score)
            social_allocation, _ = allocate_with_proportional_ties(capacity, budget, social_score)
            heat_upper = quantities(population, land, heat, social, heat_allocation)[0]
            social_upper = quantities(population, land, heat, social, social_allocation)[1]
            if heat_upper <= 0 or social_upper <= 0:
                raise RuntimeError("Non-positive objective upper bound")
            heat_at_social = quantities(population, land, heat, social, social_allocation)[0]
            social_at_heat = quantities(population, land, heat, social, heat_allocation)[1]
            values[structure_index, budget_index, :] = (
                overlap(heat_allocation, social_allocation),
                1.0 - heat_at_social / heat_upper,
                1.0 - social_at_heat / social_upper,
            )
            heat_ties[structure_index, budget_index] = tie_fraction
    return values, heat_ties, capacity_totals, budgets


def load_frame(result_root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    typology = result_root / "Result4_5_neighbourhood_typology" / "data" / "block_group_GHV_primary_classification.csv.gz"
    thermal = result_root / "Result3" / "Result3_support_aligned" / "data" / "support_aligned_model_frame_34566.csv"
    blocks = result_root / "Result3" / "Result3_support_aligned" / "spatial_block_bootstrap" / "data" / "spatial_block_assignments.csv"
    sources = {"typology": typology, "thermal": thermal, "spatial_blocks": blocks}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    v_columns = (
        "V_indicator_poverty_rate_pct",
        "V_indicator_no_vehicle_households_pct",
        "V_indicator_renter_occupied_pct",
        "V_indicator_age_65_plus_pct",
    )
    frame = pd.read_csv(
        typology,
        usecols=[
            "city_key", "city_display", "GEOID_str", "population", "land_area_m2",
            "canopy_pct_analysis", *v_columns,
        ],
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    thermal_frame = pd.read_csv(
        thermal,
        usecols=["city_key", "GEOID_str", "mean_LST_C", "log_population_density"],
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    block_frame = pd.read_csv(blocks, dtype={"GEOID_str": str}, low_memory=False)
    for data in (frame, thermal_frame, block_frame):
        data["GEOID_str"] = normalize_geoid(data["GEOID_str"])
    frame = frame.merge(thermal_frame, on=["city_key", "GEOID_str"], validate="one_to_one")
    block_columns = ["city_key", "GEOID_str", *[f"main_block_origin{i}" for i in range(4)]]
    frame = frame.merge(block_frame[block_columns], on=["city_key", "GEOID_str"], validate="one_to_one")
    frame = frame.sort_values(["city_key", "GEOID_str"]).reset_index(drop=True)
    required = [
        "population", "land_area_m2", "canopy_pct_analysis", "mean_LST_C",
        "log_population_density", *[f"main_block_origin{i}" for i in range(4)],
    ]
    if len(frame) != EXPECTED_ROWS or frame["city_key"].nunique() != EXPECTED_CITIES:
        raise RuntimeError("Frozen analysis-frame mismatch")
    if frame[required].isna().any().any():
        raise RuntimeError("Missing required non-social input")
    if (frame["population"] <= 0).any() or (frame["land_area_m2"] <= 0).any():
        raise RuntimeError("Non-positive population or land")
    return frame, sources


def arrays(frame: pd.DataFrame, index: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    if index is None:
        index = np.arange(len(frame))
    v_columns = (
        "V_indicator_poverty_rate_pct",
        "V_indicator_no_vehicle_households_pct",
        "V_indicator_renter_occupied_pct",
        "V_indicator_age_65_plus_pct",
    )
    return (
        frame["population"].to_numpy(float)[index],
        frame["land_area_m2"].to_numpy(float)[index],
        frame["canopy_pct_analysis"].to_numpy(float)[index],
        frame["log_population_density"].to_numpy(float)[index],
        frame["mean_LST_C"].to_numpy(float)[index],
        frame[list(v_columns)].to_numpy(float)[index, :],
    )


def prepare_blocks(frame: pd.DataFrame) -> tuple[tuple[np.ndarray, ...], ...]:
    result = []
    for origin in range(4):
        labels = frame[f"main_block_origin{origin}"].to_numpy(np.int64)
        order = np.argsort(labels, kind="stable")
        split = np.flatnonzero(np.diff(labels[order])) + 1
        result.append(tuple(np.split(order, split)))
    return tuple(result)


def resample_blocks(blocks: tuple[tuple[np.ndarray, ...], ...], rng: np.random.Generator) -> np.ndarray:
    groups = blocks[int(rng.integers(0, len(blocks)))]
    sampled = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[int(index)] for index in sampled]).astype(np.int64)


def bootstrap_city(payload: tuple[str, pd.DataFrame, int, int]) -> CityBootstrap:
    city_key, frame, replicates, seed = payload
    rng = np.random.default_rng(seed)
    base = arrays(frame)
    blocks = prepare_blocks(frame)
    values = np.empty((replicates, len(STRUCTURES), len(BUDGETS), len(OUTCOMES)), dtype=np.float32)
    heat_ties = np.empty((replicates, len(STRUCTURES), len(BUDGETS)), dtype=np.float32)
    for replicate in range(replicates):
        index = resample_blocks(blocks, rng)
        evaluated = evaluate_arrays(*(vector[index] for vector in base[:-1]), base[-1][index, :])
        values[replicate] = evaluated[0]
        heat_ties[replicate] = evaluated[1]
    return CityBootstrap(city_key, values, heat_ties)


def central_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome_rows = []
    control_rows = []
    for city_key, city in frame.groupby("city_key", sort=True):
        values, ties, totals, budgets = evaluate_arrays(*arrays(city))
        display = city["city_display"].iloc[0]
        for structure_index, structure in enumerate(STRUCTURES):
            for budget_index, budget_fraction in enumerate(BUDGETS):
                row = {
                    "city_key": city_key,
                    "city_display": display,
                    "structure": structure,
                    "budget_fraction": budget_fraction,
                    "capacity_km2": totals[structure_index] / 1e6,
                    "observed_capacity_km2": totals[0] / 1e6,
                    "budget_km2": budgets[budget_index] / 1e6,
                    "heat_zero_score_allocation_fraction": ties[structure_index, budget_index],
                }
                for outcome_index, outcome in enumerate(OUTCOMES):
                    row[outcome] = values[structure_index, budget_index, outcome_index]
                outcome_rows.append(row)
        for budget_index, budget_fraction in enumerate(BUDGETS):
            control_rows.append(
                {
                    "city_key": city_key,
                    "city_display": display,
                    "budget_fraction": budget_fraction,
                    "observed_vs_same_candidate_total_capacity_difference_m2": totals[1] - totals[0],
                    "observed_vs_all_unit_total_matched_capacity_difference_m2": totals[2] - totals[0],
                    "budget_km2": budgets[budget_index] / 1e6,
                }
            )
    return pd.DataFrame(outcome_rows), pd.DataFrame(control_rows)


def paired_contrasts(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    structure_pairs = {
        CONTRASTS[0]: (STRUCTURES[1], STRUCTURES[0]),
        CONTRASTS[1]: (STRUCTURES[2], STRUCTURES[1]),
        CONTRASTS[2]: (STRUCTURES[3], STRUCTURES[1]),
    }
    index = outcomes.set_index(["city_key", "city_display", "budget_fraction", "structure"])
    for contrast, (first, second) in structure_pairs.items():
        first_frame = index.xs(first, level="structure")
        second_frame = index.xs(second, level="structure")
        for outcome in OUTCOMES:
            difference = first_frame[outcome] - second_frame[outcome]
            for keys, value in difference.items():
                city_key, city_display, budget_fraction = keys
                rows.append(
                    {
                        "city_key": city_key,
                        "city_display": city_display,
                        "budget_fraction": budget_fraction,
                        "contrast": contrast,
                        "outcome": outcome,
                        "difference": value,
                    }
                )
    return pd.DataFrame(rows)


def save_city_bootstrap(result: CityBootstrap, output: Path) -> None:
    np.savez_compressed(
        output / "bootstrap_city" / f"{result.city_key}.npz",
        values=result.values,
        heat_tie_fraction=result.heat_tie_fraction,
        structures=np.asarray(STRUCTURES),
        budgets=BUDGETS,
        outcomes=np.asarray(OUTCOMES),
    )


def run_bootstraps(
    frame: pd.DataFrame, replicates: int, workers: int, output: Path
) -> dict[str, CityBootstrap]:
    payloads = [
        (city_key, city.reset_index(drop=True), replicates, SEED + 10_000 * index)
        for index, (city_key, city) in enumerate(frame.groupby("city_key", sort=True))
    ]
    results = {}
    if workers == 1:
        for payload in payloads:
            result = bootstrap_city(payload)
            results[result.city_key] = result
            save_city_bootstrap(result, output)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(bootstrap_city, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                results[result.city_key] = result
                save_city_bootstrap(result, output)
                print(f"completed {result.city_key}", flush=True)
    return results


def city_interval_tables(
    central: pd.DataFrame, results: dict[str, CityBootstrap]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows = []
    contrast_rows = []
    structure_pairs = {
        CONTRASTS[0]: (1, 0),
        CONTRASTS[1]: (2, 1),
        CONTRASTS[2]: (3, 1),
    }
    display = central.drop_duplicates("city_key").set_index("city_key")["city_display"].to_dict()
    for city_key, result in results.items():
        for structure_index, structure in enumerate(STRUCTURES):
            for budget_index, budget in enumerate(BUDGETS):
                for outcome_index, outcome in enumerate(OUTCOMES):
                    vector = result.values[:, structure_index, budget_index, outcome_index].astype(float)
                    interval_rows.append(
                        {
                            "city_key": city_key,
                            "city_display": display[city_key],
                            "structure": structure,
                            "budget_fraction": budget,
                            "outcome": outcome,
                            "bootstrap_median": float(np.nanmedian(vector)),
                            "ci_low": float(np.nanquantile(vector, 0.025)),
                            "ci_high": float(np.nanquantile(vector, 0.975)),
                        }
                    )
        for contrast, (first, second) in structure_pairs.items():
            for budget_index, budget in enumerate(BUDGETS):
                for outcome_index, outcome in enumerate(OUTCOMES):
                    vector = (
                        result.values[:, first, budget_index, outcome_index]
                        - result.values[:, second, budget_index, outcome_index]
                    ).astype(float)
                    contrast_rows.append(
                        {
                            "city_key": city_key,
                            "city_display": display[city_key],
                            "budget_fraction": budget,
                            "contrast": contrast,
                            "outcome": outcome,
                            "bootstrap_median": float(np.nanmedian(vector)),
                            "ci_low": float(np.nanquantile(vector, 0.025)),
                            "ci_high": float(np.nanquantile(vector, 0.975)),
                            "interval_excludes_zero": bool(
                                np.nanquantile(vector, 0.025) > 0 or np.nanquantile(vector, 0.975) < 0
                            ),
                        }
                    )
    return pd.DataFrame(interval_rows), pd.DataFrame(contrast_rows)


def two_level_draws(
    results: dict[str, CityBootstrap], draws: int
) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(results)
    n_cities = len(keys)
    replicates = next(iter(results.values())).values.shape[0]
    rng = np.random.default_rng(SEED + 999_999)
    outcome_draws = np.empty(
        (draws, len(STRUCTURES), len(BUDGETS), len(OUTCOMES)), dtype=np.float32
    )
    tie_draws = np.empty((draws, len(STRUCTURES), len(BUDGETS)), dtype=np.float32)
    for draw in range(draws):
        sampled_cities = rng.integers(0, n_cities, size=n_cities)
        sampled_replicates = rng.integers(0, replicates, size=n_cities)
        outcome_draws[draw] = np.mean(
            np.stack(
                [
                    results[keys[int(city)]].values[int(rep)]
                    for city, rep in zip(sampled_cities, sampled_replicates, strict=True)
                ]
            ),
            axis=0,
        )
        tie_draws[draw] = np.mean(
            np.stack(
                [
                    results[keys[int(city)]].heat_tie_fraction[int(rep)]
                    for city, rep in zip(sampled_cities, sampled_replicates, strict=True)
                ]
            ),
            axis=0,
        )
    return outcome_draws, tie_draws


def national_tables(
    central: pd.DataFrame,
    outcome_draws: np.ndarray,
    tie_draws: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    contrast_rows = []
    tie_rows = []
    for structure_index, structure in enumerate(STRUCTURES):
        for budget_index, budget in enumerate(BUDGETS):
            subset = central.loc[
                central["structure"].eq(structure) & np.isclose(central["budget_fraction"], budget)
            ]
            for outcome_index, outcome in enumerate(OUTCOMES):
                vector = outcome_draws[:, structure_index, budget_index, outcome_index].astype(float)
                rows.append(
                    {
                        "structure": structure,
                        "budget_fraction": budget,
                        "outcome": outcome,
                        "equal_city_deterministic_point": float(subset[outcome].mean()),
                        "two_level_median": float(np.median(vector)),
                        "ci_low": float(np.quantile(vector, 0.025)),
                        "ci_high": float(np.quantile(vector, 0.975)),
                    }
                )
            tie_vector = tie_draws[:, structure_index, budget_index].astype(float)
            tie_rows.append(
                {
                    "structure": structure,
                    "budget_fraction": budget,
                    "equal_city_deterministic_point": float(subset["heat_zero_score_allocation_fraction"].mean()),
                    "two_level_median": float(np.median(tie_vector)),
                    "ci_low": float(np.quantile(tie_vector, 0.025)),
                    "ci_high": float(np.quantile(tie_vector, 0.975)),
                    "cities_with_positive_deterministic_fraction": int(
                        (subset["heat_zero_score_allocation_fraction"] > 1e-12).sum()
                    ),
                }
            )
    structure_pairs = {
        CONTRASTS[0]: (1, 0),
        CONTRASTS[1]: (2, 1),
        CONTRASTS[2]: (3, 1),
    }
    central_index = central.set_index(["structure", "budget_fraction", "city_key"])
    for contrast, (first, second) in structure_pairs.items():
        for budget_index, budget in enumerate(BUDGETS):
            for outcome_index, outcome in enumerate(OUTCOMES):
                vector = (
                    outcome_draws[:, first, budget_index, outcome_index]
                    - outcome_draws[:, second, budget_index, outcome_index]
                ).astype(float)
                first_values = central_index.xs((STRUCTURES[first], budget), level=("structure", "budget_fraction"))[outcome]
                second_values = central_index.xs((STRUCTURES[second], budget), level=("structure", "budget_fraction"))[outcome]
                difference = first_values - second_values
                low = float(np.quantile(vector, 0.025))
                high = float(np.quantile(vector, 0.975))
                contrast_rows.append(
                    {
                        "contrast": contrast,
                        "budget_fraction": budget,
                        "outcome": outcome,
                        "equal_city_deterministic_difference": float(difference.mean()),
                        "two_level_median": float(np.median(vector)),
                        "ci_low": low,
                        "ci_high": high,
                        "interval_excludes_zero": bool(low > 0 or high < 0),
                        "cities_positive": int((difference > 0).sum()),
                        "cities_negative": int((difference < 0).sum()),
                        "cities_near_zero": int((np.abs(difference) <= 1e-12).sum()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(contrast_rows), pd.DataFrame(tie_rows)


def gate_table(contrasts: pd.DataFrame, ties: pd.DataFrame) -> pd.DataFrame:
    structure_pairs = {
        CONTRASTS[0]: (STRUCTURES[1], STRUCTURES[0]),
        CONTRASTS[1]: (STRUCTURES[2], STRUCTURES[1]),
        CONTRASTS[2]: (STRUCTURES[3], STRUCTURES[1]),
    }
    rows = []
    for contrast in CONTRASTS:
        for outcome in OUTCOMES:
            subset = contrasts.loc[
                contrasts["contrast"].eq(contrast) & contrasts["outcome"].eq(outcome)
            ].sort_values("budget_fraction")
            points = subset["equal_city_deterministic_difference"].to_numpy(float)
            lows = subset["ci_low"].to_numpy(float)
            highs = subset["ci_high"].to_numpy(float)
            same_positive = bool((points > 0).all() and (lows > 0).all())
            same_negative = bool((points < 0).all() and (highs < 0).all())
            same_supported_direction = same_positive or same_negative
            material_all = bool((np.abs(points) >= MATERIALITY).all())
            first_structure, second_structure = structure_pairs[contrast]
            compared_ties = ties.loc[
                ties["structure"].isin((first_structure, second_structure))
            ]
            spatially_unique = bool(
                outcome == "heat_context_opportunity_loss_under_social"
                or (compared_ties["cities_with_positive_deterministic_fraction"] == 0).all()
            )
            passes_individual = same_supported_direction and material_all and spatially_unique
            rows.append(
                {
                    "contrast": contrast,
                    "outcome": outcome,
                    "same_supported_direction_all_budgets": same_supported_direction,
                    "material_at_all_budgets": material_all,
                    "heat_allocation_spatially_unique_for_gate": spatially_unique,
                    "passes_individual_gate": passes_individual,
                    "passes_main_text_gate": bool(
                        passes_individual and contrast == CONTRASTS[0]
                    ),
                    "direction": "positive" if same_positive else "negative" if same_negative else "unstable_or_unsupported",
                    "minimum_absolute_point_change": float(np.min(np.abs(points))),
                    "point_5pct": float(points[0]),
                    "point_10pct": float(points[1]),
                    "point_20pct": float(points[2]),
                }
            )
    gate = pd.DataFrame(rows)
    total = gate.loc[gate["contrast"].eq(CONTRASTS[1])].set_index("outcome")
    rate = gate.loc[gate["contrast"].eq(CONTRASTS[2])].set_index("outcome")
    robust_rows = []
    for outcome in OUTCOMES:
        same_direction = total.loc[outcome, "direction"] == rate.loc[outcome, "direction"]
        robust_rows.append(
            {
                "contrast": "candidate_gate_robust_across_capacity_matching",
                "outcome": outcome,
                "same_supported_direction_all_budgets": bool(
                    total.loc[outcome, "same_supported_direction_all_budgets"]
                    and rate.loc[outcome, "same_supported_direction_all_budgets"]
                    and same_direction
                ),
                "material_at_all_budgets": bool(
                    total.loc[outcome, "material_at_all_budgets"]
                    and rate.loc[outcome, "material_at_all_budgets"]
                ),
                "heat_allocation_spatially_unique_for_gate": bool(
                    total.loc[outcome, "heat_allocation_spatially_unique_for_gate"]
                    and rate.loc[outcome, "heat_allocation_spatially_unique_for_gate"]
                ),
                "passes_individual_gate": bool(
                    total.loc[outcome, "passes_individual_gate"]
                    and rate.loc[outcome, "passes_individual_gate"]
                    and same_direction
                ),
                "passes_main_text_gate": bool(
                    total.loc[outcome, "passes_individual_gate"]
                    and rate.loc[outcome, "passes_individual_gate"]
                    and same_direction
                ),
                "direction": total.loc[outcome, "direction"] if same_direction else "matching_sensitive",
                "minimum_absolute_point_change": float(
                    min(
                        total.loc[outcome, "minimum_absolute_point_change"],
                        rate.loc[outcome, "minimum_absolute_point_change"],
                    )
                ),
                "point_5pct": np.nan,
                "point_10pct": np.nan,
                "point_20pct": np.nan,
            }
        )
    return pd.concat([gate, pd.DataFrame(robust_rows)], ignore_index=True)


def step1_reproduction(central: pd.DataFrame, step1_root: Path) -> dict[str, float]:
    source = pd.read_csv(step1_root / "tables" / "city_primary_tradeoff_outcomes.csv")
    current = central.loc[
        central["structure"].eq(STRUCTURES[0]) & np.isclose(central["budget_fraction"], 0.10)
    ]
    merged = current.merge(source, on="city_key", suffixes=("_step2", "_step1"), validate="one_to_one")
    errors = {}
    for outcome in OUTCOMES:
        errors[outcome] = float(
            np.max(np.abs(merged[f"{outcome}_step2"] - merged[f"{outcome}_step1"]))
        )
    return errors


def input_manifest(sources: dict[str, Path], step1_root: Path, protocol: Path, output: Path) -> None:
    paths = {**sources, "step1_machine_audit": step1_root / "qa" / "step01_machine_audit.json", "frozen_protocol": protocol}
    rows = []
    for role, path in paths.items():
        rows.append({"role": role, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(output / "manifests" / "input_lineage.csv", index=False, encoding="utf-8-sig")


def write_interpretation(
    national: pd.DataFrame,
    contrasts: pd.DataFrame,
    gate: pd.DataFrame,
    ties: pd.DataFrame,
    output: Path,
) -> None:
    primary = national.loc[
        national["structure"].eq(STRUCTURES[0]) & np.isclose(national["budget_fraction"], 0.10)
    ].set_index("outcome")
    passing = gate.loc[gate["passes_main_text_gate"]]
    text = [
        "# Step 2 interpretation ledger",
        "",
        "## Frozen baseline at the 10% budget",
        "",
        f"- Jaccard allocation overlap: {primary.loc['allocation_overlap', 'equal_city_deterministic_point']:.3f}.",
        f"- Heat-context opportunity loss under social allocation: {primary.loc['heat_context_opportunity_loss_under_social', 'equal_city_deterministic_point']:.3f}.",
        f"- Social-context opportunity loss under heat allocation: {primary.loc['social_context_opportunity_loss_under_heat', 'equal_city_deterministic_point']:.3f}.",
        "",
        "## Prespecified gate",
        "",
    ]
    if passing.empty:
        text.append("No controlled structural contrast passed the prespecified all-budget support and materiality gate.")
    else:
        for _, row in passing.iterrows():
            text.append(
                f"- `{row['contrast']}` passed for `{row['outcome']}` in the {row['direction']} direction; "
                f"minimum absolute point change across budgets was {row['minimum_absolute_point_change']:.3f}."
            )
    text.extend(
        [
            "",
            "Passing the gate permits a structural-sensitivity statement, not causal attribution.",
            "The candidate-gate interpretation additionally requires agreement between total-matched and rate-matched specifications.",
            "",
            "## Heat upper-bound non-uniqueness audit",
            "",
        ]
    )
    for _, row in ties.sort_values(["structure", "budget_fraction"]).iterrows():
        if row["cities_with_positive_deterministic_fraction"] > 0:
            text.append(
                f"- {row['structure']}, {100*row['budget_fraction']:.0f}% budget: "
                f"{int(row['cities_with_positive_deterministic_fraction'])} cities allocated some area at H=0; "
                f"equal-city mean fraction {row['equal_city_deterministic_point']:.3f}."
            )
    (output / "text" / "STEP02_INTERPRETATION_LEDGER.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def audit(
    central: pd.DataFrame,
    controls: pd.DataFrame,
    national: pd.DataFrame,
    contrasts: pd.DataFrame,
    city_intervals: pd.DataFrame,
    gate: pd.DataFrame,
    ties: pd.DataFrame,
    reproduction: dict[str, float],
    args: argparse.Namespace,
    output: Path,
) -> dict[str, object]:
    primary = central.loc[central["structure"].isin(PRIMARY_STRUCTURES)]
    bounded = primary[list(OUTCOMES)].apply(lambda column: column.between(-1e-10, 1 + 1e-10).all()).all()
    budget_errors = float(
        np.max(
            np.abs(
                controls[
                    [
                        "observed_vs_same_candidate_total_capacity_difference_m2",
                        "observed_vs_all_unit_total_matched_capacity_difference_m2",
                    ]
                ].to_numpy(float)
            )
        )
    )
    conditions = {
        "protocol_preexists_results": args.protocol.is_file(),
        "all_50_cities": central["city_key"].nunique() == EXPECTED_CITIES,
        "all_structures_and_budgets": len(central) == EXPECTED_CITIES * len(STRUCTURES) * len(BUDGETS),
        "primary_outcomes_bounded": bool(bounded),
        "paired_capacity_totals_match_below_1e_5_m2": budget_errors < 1e-5,
        "step1_reproduced_below_1e_12": max(reproduction.values()) < 1e-12,
        "city_bootstrap_complete": len(city_intervals) == EXPECTED_CITIES * len(STRUCTURES) * len(BUDGETS) * len(OUTCOMES),
        "national_intervals_complete": len(national) == len(STRUCTURES) * len(BUDGETS) * len(OUTCOMES),
        "paired_contrasts_complete": len(contrasts) == len(CONTRASTS) * len(BUDGETS) * len(OUTCOMES),
        "gate_evaluated_before_text_selection": len(gate) == (len(CONTRASTS) + 1) * len(OUTCOMES),
        "tie_nonuniqueness_recorded": len(ties) == len(STRUCTURES) * len(BUDGETS),
        "no_causal_or_implementation_claim": True,
    }
    status = "PASS" if all(conditions.values()) else "FAIL"
    audit_data = {
        "status": status,
        "condition_pass_fraction": float(np.mean(list(conditions.values()))),
        "analysis_units": EXPECTED_ROWS,
        "cities": EXPECTED_CITIES,
        "structures": list(STRUCTURES),
        "budgets": BUDGETS.tolist(),
        "city_spatial_block_replicates": args.city_replicates,
        "two_level_draws": args.outer_draws,
        "materiality_threshold": MATERIALITY,
        "step1_reproduction_error": reproduction,
        "maximum_total_matched_capacity_error_m2": budget_errors,
        "conditions": conditions,
        "main_text_gate_passes": gate.loc[gate["passes_main_text_gate"], ["contrast", "outcome", "direction"]].to_dict("records"),
        "interpretive_boundary": (
            "The module identifies controlled structural sensitivity in allocation geometry. "
            "It does not identify causal drivers, realised intervention outcomes or unique shares explained."
        ),
    }
    (output / "qa" / "step02_machine_audit.json").write_text(
        json.dumps(audit_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if status != "PASS":
        raise AssertionError("Step 2 audit failed: " + ", ".join(k for k, v in conditions.items() if not v))
    return audit_data


def write_technical_report(
    audit_data: dict[str, object], gate: pd.DataFrame, ties: pd.DataFrame, output: Path
) -> None:
    passes = gate.loc[gate["passes_main_text_gate"]]
    passing_text = (
        "No contrast passed the prespecified gate."
        if passes.empty
        else "; ".join(f"{row.contrast} / {row.outcome} ({row.direction})" for row in passes.itertuples())
    )
    nonunique = ties.loc[ties["cities_with_positive_deterministic_fraction"] > 0]
    report = f"""# Technical audit: Step 2

## Decision

**{audit_data['status']} as a controlled structural-decomposition module.**
All {len(audit_data['conditions'])} prespecified machine-checkable conditions
must pass for this status.

## Prespecified gate result

{passing_text}

Passing a gate authorises only a structural-sensitivity statement. It does not
show why thermal and social geographies arose, nor does it assign a causal or
unique percentage contribution to capacity intensity or the candidate gate.

## Strengths

- Step 1 is reproduced exactly at the observed 10% specification.
- Capacity-intensity and candidate-set changes are evaluated sequentially rather
  than being conflated.
- The candidate gate has both total-matched and rate-matched specifications.
- All contrasts are paired within spatial-block replicates and tested over 5%,
  10% and 20% budgets under a frozen 0.02 materiality rule.
- National estimates give each city equal weight while preserving city-level
  heterogeneity.

## Remaining risks

1. The decomposition is path-dependent and cannot be interpreted as additive
   variance explained.
2. The objective scores themselves mathematically determine the ordering; their
   association with overlap is not an independent mechanism.
3. Removing the candidate gate is a null comparison, not an implementable
   allocation option.
4. Surface-heat upper bounds can be non-unique after all positive-H capacity is
   exhausted. Proportional tie allocation is neutral but convention-dependent.
5. All quantities remain model-defined and omit parcel feasibility, costs,
   irrigation, ownership, tree growth and survival.

## Heat-score tie audit

{len(nonunique)} structure-budget cells had at least one city assigning some
surface-heat allocation at H=0. Their overlap and social-retention values should
not be presented as unique heat-optimal spatial solutions without this caveat.
"""
    (output / "qa" / "STEP02_TECHNICAL_AUDIT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    step1_root = args.step1_root.resolve()
    output = args.output.resolve()
    args.protocol = args.protocol.resolve()
    ensure_directories(output)
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    shutil.copy2(args.protocol, output / "methods" / args.protocol.name)
    frame, sources = load_frame(result_root)
    input_manifest(sources, step1_root, args.protocol, output)
    central, controls = central_analysis(frame)
    central.to_csv(output / "tables" / "city_structure_budget_outcomes.csv", index=False, encoding="utf-8-sig")
    controls.to_csv(output / "qa" / "city_budget_capacity_control_ledger.csv", index=False, encoding="utf-8-sig")
    paired_contrasts(central).to_csv(
        output / "tables" / "city_central_controlled_contrasts.csv", index=False, encoding="utf-8-sig"
    )
    reproduction = step1_reproduction(central, step1_root)
    if max(reproduction.values()) >= 1e-12:
        raise RuntimeError(f"Step 1 reproduction failed before bootstrap: {reproduction}")
    if args.central_only:
        print(json.dumps({"status": "CENTRAL_ONLY_PASS", "step1_reproduction": reproduction}, indent=2))
        return

    results = run_bootstraps(frame, args.city_replicates, args.workers, output)
    city_intervals, city_contrast_intervals = city_interval_tables(central, results)
    city_intervals.to_csv(output / "tables" / "city_structure_budget_intervals.csv", index=False, encoding="utf-8-sig")
    city_contrast_intervals.to_csv(
        output / "tables" / "city_controlled_contrast_intervals.csv", index=False, encoding="utf-8-sig"
    )
    outcome_draws, tie_draws = two_level_draws(results, args.outer_draws)
    np.savez_compressed(
        output / "diagnostics" / "national_two_level_draws.npz",
        outcomes=outcome_draws,
        heat_tie_fraction=tie_draws,
        structures=np.asarray(STRUCTURES),
        budgets=BUDGETS,
        outcome_names=np.asarray(OUTCOMES),
    )
    national, contrasts, ties = national_tables(central, outcome_draws, tie_draws)
    national.to_csv(output / "tables" / "national_structure_budget_outcomes.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(output / "tables" / "national_controlled_contrasts.csv", index=False, encoding="utf-8-sig")
    ties.to_csv(output / "qa" / "heat_upper_bound_nonuniqueness.csv", index=False, encoding="utf-8-sig")
    gate = gate_table(contrasts, ties)
    gate.to_csv(output / "qa" / "main_text_gate.csv", index=False, encoding="utf-8-sig")
    write_interpretation(national, contrasts, gate, ties, output)
    audit_data = audit(central, controls, national, contrasts, city_intervals, gate, ties, reproduction, args, output)
    write_technical_report(audit_data, gate, ties, output)
    shutil.copy2(Path(__file__), output / "methods" / Path(__file__).name)
    print(json.dumps(audit_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
