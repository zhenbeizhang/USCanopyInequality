from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_ROOT = Path(str(analysis_store_root().joinpath('Result')))
LEGACY_PATH = Path(__file__).with_name("run_equal_budget_allocation.py")
CURRENT_TABLE = (
    RESULT_ROOT
    / "Result5"
    / "versions"
    / "step2_multicriteria_pareto"
    / "tables"
    / "01_city_rule_outcomes.csv"
)
OUT = RESULT_ROOT / "Result5" / "versions" / "budget_sensitivity"

BUDGETS = np.asarray([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00], dtype=float)
SELECTED_METRICS = (
    "pareto_heat_efficiency",
    "pareto_social_efficiency",
    "pareto_population_efficiency",
    "heat_oracle_social_efficiency",
    "social_oracle_heat_efficiency",
    "heat_social_oracle_overlap",
)
BASE_SEED = 38_110_001


def load_legacy():
    name = "result5_budget_base"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy()


def prepare_context(arrays: tuple[np.ndarray, ...]) -> dict[str, np.ndarray | float]:
    population, land, canopy, log_density, temperature, v_indicators, _ = arrays
    gap, _ = LEGACY.recompute_gap(canopy, log_density)
    heat_reference = float(np.average(temperature, weights=population))
    heat = np.maximum(temperature - heat_reference, 0.0)
    social = LEGACY.recompute_social_condition(v_indicators)
    capacity = land * gap / 100.0
    return {
        "population": population,
        "land": land,
        "heat": heat,
        "social": social,
        "capacity": capacity,
        "marginal_heat": 100.0 * population * heat / land,
        "marginal_social": 100.0 * population * social / land,
        "marginal_population": 100.0 * population / land,
    }


def evaluate_selected(context: dict[str, np.ndarray | float], fraction: float) -> np.ndarray:
    population = np.asarray(context["population"], dtype=float)
    land = np.asarray(context["land"], dtype=float)
    heat = np.asarray(context["heat"], dtype=float)
    social = np.asarray(context["social"], dtype=float)
    capacity = np.asarray(context["capacity"], dtype=float)
    marginal_heat = np.asarray(context["marginal_heat"], dtype=float)
    marginal_social = np.asarray(context["marginal_social"], dtype=float)
    marginal_population = np.asarray(context["marginal_population"], dtype=float)

    budget = float(fraction * capacity.sum())
    if budget <= 0:
        raise RuntimeError("Non-positive budget")
    heat_allocation = LEGACY.hard_allocate(capacity, budget, marginal_heat)
    social_allocation = LEGACY.hard_allocate(capacity, budget, marginal_social)
    population_allocation = LEGACY.hard_allocate(capacity, budget, marginal_population)

    heat_oracle = LEGACY.allocation_quantities(
        population, land, heat, social, heat_allocation
    )[0]
    social_oracle = LEGACY.allocation_quantities(
        population, land, heat, social, social_allocation
    )[1]
    population_oracle = LEGACY.allocation_quantities(
        population, land, heat, social, population_allocation
    )[2]
    if min(heat_oracle, social_oracle, population_oracle) <= 0:
        raise RuntimeError("Non-positive oracle quantity")

    pareto_score = 0.5 * marginal_heat / heat_oracle + 0.5 * marginal_social / social_oracle
    pareto_allocation = LEGACY.hard_allocate(capacity, budget, pareto_score)
    pareto_quantities = LEGACY.allocation_quantities(
        population, land, heat, social, pareto_allocation
    )
    heat_cross = LEGACY.allocation_quantities(
        population, land, heat, social, heat_allocation
    )[1]
    social_cross = LEGACY.allocation_quantities(
        population, land, heat, social, social_allocation
    )[0]

    return np.asarray(
        [
            pareto_quantities[0] / heat_oracle,
            pareto_quantities[1] / social_oracle,
            pareto_quantities[2] / population_oracle,
            heat_cross / social_oracle,
            social_cross / heat_oracle,
            LEGACY.continuous_overlap(heat_allocation, social_allocation),
        ],
        dtype=float,
    )


def central_all_rules(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    original_budget = LEGACY.PRIMARY_BUDGET
    try:
        for city_key, city_frame in frame.groupby("city_key", sort=True):
            arrays = LEGACY.city_arrays(city_frame)
            display = str(city_frame["city_display"].iloc[0])
            for fraction in BUDGETS:
                LEGACY.PRIMARY_BUDGET = float(fraction)
                metrics, _, overlap, budget, allocated = LEGACY.evaluate_city_arrays(*arrays)
                for rule_index, rule in enumerate(LEGACY.ALL_RULES):
                    rows.append(
                        {
                            "city_key": city_key,
                            "city_display": display,
                            "budget_fraction": fraction,
                            "rule": rule,
                            "budget_m2": budget,
                            "allocated_m2": allocated[rule_index],
                            "heat_context_efficiency": metrics[rule_index, 0],
                            "social_condition_efficiency": metrics[rule_index, 1],
                            "population_efficiency": metrics[rule_index, 2],
                            "heat_social_oracle_overlap": overlap,
                        }
                    )
    finally:
        LEGACY.PRIMARY_BUDGET = original_budget
    return pd.DataFrame(rows)


def bootstrap_city(payload: tuple[str, pd.DataFrame, int, int]) -> tuple[str, np.ndarray]:
    city_key, city_frame, replicates, seed = payload
    rng = np.random.default_rng(seed)
    output = np.empty((replicates, len(BUDGETS), len(SELECTED_METRICS)), dtype=np.float32)
    for replicate in range(replicates):
        indices = LEGACY.block_resample_indices(city_frame, rng)
        context = prepare_context(LEGACY.city_arrays(city_frame, indices))
        for budget_index, fraction in enumerate(BUDGETS):
            output[replicate, budget_index] = evaluate_selected(context, float(fraction))
    return city_key, output


def run_bootstrap(frame: pd.DataFrame, replicates: int, workers: int) -> dict[str, np.ndarray]:
    directory = OUT / "bootstrap_city"
    directory.mkdir(parents=True, exist_ok=True)
    payloads = []
    for city_index, (city_key, city_frame) in enumerate(frame.groupby("city_key", sort=True)):
        payloads.append(
            (
                city_key,
                city_frame.reset_index(drop=True),
                replicates,
                BASE_SEED + city_index * 10_000,
            )
        )
    results: dict[str, np.ndarray] = {}
    if workers == 1:
        for payload in payloads:
            city_key, array = bootstrap_city(payload)
            np.savez_compressed(directory / f"{city_key}.npz", metrics=array)
            results[city_key] = array
            print(f"Completed {city_key}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(bootstrap_city, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                city_key, array = future.result()
                np.savez_compressed(directory / f"{city_key}.npz", metrics=array)
                results[city_key] = array
                print(f"Completed {city_key}", flush=True)
    return results


def two_level_intervals(
    city_results: dict[str, np.ndarray],
    central: pd.DataFrame,
    draws: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    city_keys = sorted(city_results)
    stack = np.stack([city_results[key] for key in city_keys], axis=0)
    n_cities, n_replicates = stack.shape[:2]
    rng = np.random.default_rng(BASE_SEED + 999_999)
    selected_cities = rng.integers(0, n_cities, size=(draws, n_cities))
    selected_replicates = rng.integers(0, n_replicates, size=(draws, n_cities))
    sampled = stack[selected_cities, selected_replicates]
    draw_values = np.nanmean(sampled, axis=1)

    selected_central = []
    for city_key, city_frame in central.groupby("city_key", sort=True):
        rows = []
        for fraction in BUDGETS:
            budget_rows = city_frame[np.isclose(city_frame["budget_fraction"], fraction)]
            pareto = budget_rows[budget_rows["rule"].eq("pareto_0.50")].iloc[0]
            heat_oracle = budget_rows[budget_rows["rule"].eq("heat_context_oracle")].iloc[0]
            social_oracle = budget_rows[budget_rows["rule"].eq("social_condition_oracle")].iloc[0]
            rows.append(
                [
                    pareto["heat_context_efficiency"],
                    pareto["social_condition_efficiency"],
                    pareto["population_efficiency"],
                    heat_oracle["social_condition_efficiency"],
                    social_oracle["heat_context_efficiency"],
                    heat_oracle["heat_social_oracle_overlap"],
                ]
            )
        selected_central.append(rows)
    central_equal_city = np.mean(np.asarray(selected_central, dtype=float), axis=0)

    rows = []
    for budget_index, fraction in enumerate(BUDGETS):
        for metric_index, metric in enumerate(SELECTED_METRICS):
            values = draw_values[:, budget_index, metric_index]
            rows.append(
                {
                    "budget_fraction": fraction,
                    "metric": metric,
                    "equal_city_central": central_equal_city[budget_index, metric_index],
                    "two_level_median": float(np.nanmedian(values)),
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows), draw_values


def audit(
    frame: pd.DataFrame,
    central: pd.DataFrame,
    intervals: pd.DataFrame,
    replicates: int,
    draws: int,
) -> dict:
    current = pd.read_csv(CURRENT_TABLE)
    new_10 = central[np.isclose(central["budget_fraction"], 0.10)].copy()
    joined = new_10.merge(
        current,
        on=["city_key", "city_display", "rule"],
        suffixes=("_new", ""),
        validate="one_to_one",
    )
    compare_columns = (
        "budget_m2",
        "allocated_m2",
        "heat_context_efficiency",
        "social_condition_efficiency",
        "population_efficiency",
    )
    max_differences = {
        column: float(np.max(np.abs(joined[f"{column}_new"] - joined[f"{column}"])))
        for column in compare_columns
    }
    budget_error = float(
        np.max(
            np.abs(central["allocated_m2"] - central["budget_m2"])
            / central["budget_m2"]
        )
    )
    at_full = central[np.isclose(central["budget_fraction"], 1.0)]
    full_efficiency_error = float(
        np.max(
            np.abs(
                at_full[
                    [
                        "heat_context_efficiency",
                        "social_condition_efficiency",
                        "population_efficiency",
                    ]
                ].to_numpy(float)
                - 1.0
            )
        )
    )
    qa = {
        "status": "PASS",
        "rows": int(len(frame)),
        "cities": int(frame["city_key"].nunique()),
        "budgets": BUDGETS.tolist(),
        "city_spatial_block_replicates": replicates,
        "two_level_draws": draws,
        "ten_percent_reproduction_max_absolute_difference": max_differences,
        "max_relative_budget_error": budget_error,
        "full_budget_efficiency_error": full_efficiency_error,
        "interval_rows": int(len(intervals)),
        "checks": {
            "fixed_frame_34566": len(frame) == 34566,
            "all_50_cities": frame["city_key"].nunique() == 50,
            "all_8_budgets": central["budget_fraction"].nunique() == 8,
            "ten_percent_exactly_reproduces": max(max_differences.values()) < 1e-8,
            "equal_area_control": budget_error < 1e-10,
            "all_rules_converge_at_full_budget": full_efficiency_error < 1e-8,
            "efficiencies_bounded": bool(
                central[
                    [
                        "heat_context_efficiency",
                        "social_condition_efficiency",
                        "population_efficiency",
                    ]
                ].to_numpy(float).min()
                >= -1e-9
                and central[
                    [
                        "heat_context_efficiency",
                        "social_condition_efficiency",
                        "population_efficiency",
                    ]
                ].to_numpy(float).max()
                <= 1.0 + 1e-8
            ),
        },
    }
    qa["status"] = "PASS" if all(bool(value) for value in qa["checks"].values()) else "FAIL"
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-replicates", type=int, default=500)
    parser.add_argument("--outer-draws", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    for directory in (OUT / "tables", OUT / "data", OUT / "qa", OUT / "bootstrap_city"):
        directory.mkdir(parents=True, exist_ok=True)

    frame = LEGACY.load_frame()
    central = central_all_rules(frame)
    central.to_csv(OUT / "tables" / "01_city_rule_outcomes_by_budget.csv", index=False)

    results = run_bootstrap(frame, args.city_replicates, args.workers)
    intervals, draw_values = two_level_intervals(results, central, args.outer_draws)
    intervals.to_csv(OUT / "tables" / "02_equal_city_budget_intervals.csv", index=False)
    np.savez_compressed(
        OUT / "data" / "two_level_budget_draws.npz",
        draws=draw_values,
        budgets=BUDGETS,
        metrics=np.asarray(SELECTED_METRICS),
    )

    qa = audit(frame, central, intervals, args.city_replicates, args.outer_draws)
    (OUT / "qa" / "budget_sensitivity_audit.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
