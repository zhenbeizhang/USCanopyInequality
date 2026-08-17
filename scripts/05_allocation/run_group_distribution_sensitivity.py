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


PROJECT = analysis_store_root()
RESULT = PROJECT / "Result"
LEGACY_PATH = Path(__file__).with_name("run_equal_budget_allocation.py")
CENTRAL = (
    RESULT
    / "Result5"
    / "versions"
    / "step2_multicriteria_pareto"
    / "tables"
    / "02_city_group_distribution_ratios.csv"
)
OUT = RESULT / "Result5" / "versions" / "group_budget_sensitivity"

BUDGETS = np.asarray([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00], dtype=float)
RULES = ("uniform", "heat_context_oracle", "social_condition_oracle", "pareto_0.50")
GROUPS = (
    "people_of_color_pct",
    "hispanic_or_latino_pct",
    "black_alone_pct",
    "asian_alone_pct",
    "white_alone_pct",
    "age_65_plus_pct",
)
BASE_SEED = 38_140_001


def load_legacy():
    name = "result5_group_budget_base"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy()


def allocation_context(arrays: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
    population, land, canopy, log_density, temperature, v_indicators, group_pct = arrays
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
        "group_pct": group_pct,
        "marginal_heat": 100.0 * population * heat / land,
        "marginal_social": 100.0 * population * social / land,
    }


def evaluate_groups(context: dict[str, np.ndarray], fraction: float) -> np.ndarray:
    population = context["population"]
    land = context["land"]
    heat = context["heat"]
    social = context["social"]
    capacity = context["capacity"]
    group_pct = context["group_pct"]
    marginal_heat = context["marginal_heat"]
    marginal_social = context["marginal_social"]
    budget = float(fraction * capacity.sum())

    uniform = capacity * fraction
    heat_allocation = LEGACY.hard_allocate(capacity, budget, marginal_heat)
    social_allocation = LEGACY.hard_allocate(capacity, budget, marginal_social)
    heat_oracle = LEGACY.allocation_quantities(population, land, heat, social, heat_allocation)[0]
    social_oracle = LEGACY.allocation_quantities(population, land, heat, social, social_allocation)[1]
    if min(heat_oracle, social_oracle) <= 0:
        raise RuntimeError("Non-positive heat or social oracle quantity")
    balanced_score = 0.5 * marginal_heat / heat_oracle + 0.5 * marginal_social / social_oracle
    balanced_allocation = LEGACY.hard_allocate(capacity, budget, balanced_score)
    allocations = (uniform, heat_allocation, social_allocation, balanced_allocation)
    return np.stack(
        [LEGACY.group_distribution_ratio(population, land, group_pct, allocation) for allocation in allocations],
        axis=0,
    )


def central_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city_key, city in frame.groupby("city_key", sort=True):
        context = allocation_context(LEGACY.city_arrays(city))
        city_display = str(city["city_display"].iloc[0])
        for fraction in BUDGETS:
            values = evaluate_groups(context, float(fraction))
            for rule_index, rule in enumerate(RULES):
                for group_index, group in enumerate(GROUPS):
                    rows.append(
                        {
                            "city_key": city_key,
                            "city_display": city_display,
                            "budget_fraction": fraction,
                            "budget_pct": 100.0 * fraction,
                            "rule": rule,
                            "group": group,
                            "allocation_share_to_population_share": values[rule_index, group_index],
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_city(payload: tuple[str, pd.DataFrame, int, int]) -> tuple[str, np.ndarray]:
    city_key, city, replicates, seed = payload
    rng = np.random.default_rng(seed)
    output = np.empty((replicates, len(BUDGETS), len(RULES), len(GROUPS)), dtype=np.float32)
    for replicate in range(replicates):
        indices = LEGACY.block_resample_indices(city, rng)
        context = allocation_context(LEGACY.city_arrays(city, indices))
        for budget_index, fraction in enumerate(BUDGETS):
            output[replicate, budget_index] = evaluate_groups(context, float(fraction))
    return city_key, output


def run_bootstrap(frame: pd.DataFrame, replicates: int, workers: int) -> dict[str, np.ndarray]:
    directory = OUT / "bootstrap_city"
    directory.mkdir(parents=True, exist_ok=True)
    payloads = [
        (city_key, city.reset_index(drop=True), replicates, BASE_SEED + 10_000 * city_index)
        for city_index, (city_key, city) in enumerate(frame.groupby("city_key", sort=True))
    ]
    results: dict[str, np.ndarray] = {}
    if workers == 1:
        for payload in payloads:
            city_key, values = bootstrap_city(payload)
            np.savez_compressed(directory / f"{city_key}.npz", group_ratios=values)
            results[city_key] = values
            print(f"Completed {city_key}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(bootstrap_city, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                city_key, values = future.result()
                np.savez_compressed(directory / f"{city_key}.npz", group_ratios=values)
                results[city_key] = values
                print(f"Completed {city_key}", flush=True)
    return results


def two_level_draws(
    city_results: dict[str, np.ndarray], draws: int, batch_size: int = 250
) -> np.ndarray:
    city_keys = sorted(city_results)
    stack = np.stack([city_results[key] for key in city_keys], axis=0)
    n_cities, n_replicates = stack.shape[:2]
    rng = np.random.default_rng(BASE_SEED + 999_999)
    output = np.empty((draws, len(BUDGETS), len(RULES), len(GROUPS)), dtype=np.float32)
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        size = stop - start
        city_indices = rng.integers(0, n_cities, size=(size, n_cities))
        replicate_indices = rng.integers(0, n_replicates, size=(size, n_cities))
        sampled = stack[city_indices, replicate_indices]
        output[start:stop] = np.nanmean(sampled, axis=1)
    return output


def interval_tables(central: pd.DataFrame, draws: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    central_mean = (
        central.groupby(["budget_fraction", "rule", "group"], sort=False)[
            "allocation_share_to_population_share"
        ]
        .mean()
        .reindex(
            pd.MultiIndex.from_product([BUDGETS, RULES, GROUPS], names=["budget_fraction", "rule", "group"])
        )
    )
    absolute_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    for budget_index, fraction in enumerate(BUDGETS):
        for rule_index, rule in enumerate(RULES):
            for group_index, group in enumerate(GROUPS):
                values = draws[:, budget_index, rule_index, group_index]
                central_value = float(central_mean.loc[(fraction, rule, group)])
                absolute_rows.append(
                    {
                        "budget_fraction": fraction,
                        "budget_pct": 100.0 * fraction,
                        "rule": rule,
                        "group": group,
                        "equal_city_central": central_value,
                        "two_level_median": float(np.nanmedian(values)),
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                    }
                )
                if rule != "uniform":
                    difference_values = values - draws[:, budget_index, 0, group_index]
                    central_difference = central_value - float(central_mean.loc[(fraction, "uniform", group)])
                    low = float(np.nanquantile(difference_values, 0.025))
                    high = float(np.nanquantile(difference_values, 0.975))
                    difference_rows.append(
                        {
                            "budget_fraction": fraction,
                            "budget_pct": 100.0 * fraction,
                            "rule": rule,
                            "group": group,
                            "equal_city_central_difference_from_uniform": central_difference,
                            "two_level_median_difference_from_uniform": float(np.nanmedian(difference_values)),
                            "ci_low": low,
                            "ci_high": high,
                            "paired_interval_excludes_zero": bool(low > 0 or high < 0),
                        }
                    )
    return pd.DataFrame(absolute_rows), pd.DataFrame(difference_rows)


def audit(
    frame: pd.DataFrame,
    central: pd.DataFrame,
    absolute: pd.DataFrame,
    differences: pd.DataFrame,
    replicates: int,
    draws: int,
) -> dict[str, object]:
    old = pd.read_csv(CENTRAL)
    ten = central[np.isclose(central["budget_fraction"], 0.10)].drop(columns=["budget_fraction", "budget_pct"])
    joined = ten.merge(
        old,
        on=["city_key", "city_display", "rule", "group"],
        suffixes=("_new", ""),
        validate="one_to_one",
    )
    reproduction_error = float(
        np.nanmax(
            np.abs(
                joined["allocation_share_to_population_share_new"]
                - joined["allocation_share_to_population_share"]
            )
        )
    )
    full = differences[np.isclose(differences["budget_fraction"], 1.0)]
    full_difference_error = float(np.nanmax(np.abs(full["equal_city_central_difference_from_uniform"])))
    uniform = absolute[absolute["rule"].eq("uniform")].pivot(
        index="group", columns="budget_fraction", values="equal_city_central"
    )
    uniform_budget_error = float(np.nanmax(np.abs(uniform.to_numpy(float) - uniform.iloc[:, [0]].to_numpy(float))))
    qa = {
        "status": "PASS",
        "rows": int(len(frame)),
        "cities": int(frame["city_key"].nunique()),
        "budgets": BUDGETS.tolist(),
        "rules": list(RULES),
        "groups": list(GROUPS),
        "city_spatial_block_replicates": replicates,
        "two_level_draws": draws,
        "central_rows": int(len(central)),
        "absolute_interval_rows": int(len(absolute)),
        "paired_difference_rows": int(len(differences)),
        "ten_percent_reproduction_max_absolute_difference": reproduction_error,
        "full_budget_difference_from_uniform_error": full_difference_error,
        "uniform_budget_invariance_error": uniform_budget_error,
        "checks": {
            "fixed_frame_34566": len(frame) == 34566,
            "all_50_cities": frame["city_key"].nunique() == 50,
            "complete_central_grid": len(central) == 50 * 8 * 4 * 6,
            "complete_absolute_interval_grid": len(absolute) == 8 * 4 * 6,
            "complete_paired_difference_grid": len(differences) == 8 * 3 * 6,
            "ten_percent_exactly_reproduces": reproduction_error < 1e-10,
            "uniform_is_budget_invariant": uniform_budget_error < 1e-10,
            "all_rules_equal_uniform_at_full_budget": full_difference_error < 1e-8,
            "all_intervals_finite_and_ordered": bool(
                absolute[["ci_low", "ci_high"]].notna().all().all()
                and (absolute["ci_low"] <= absolute["ci_high"]).all()
                and differences[["ci_low", "ci_high"]].notna().all().all()
                and (differences["ci_low"] <= differences["ci_high"]).all()
            ),
        },
    }
    qa["status"] = "PASS" if all(qa["checks"].values()) else "FAIL"
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Result 5 demographic allocation sensitivity across budgets")
    parser.add_argument("--city-replicates", type=int, default=500)
    parser.add_argument("--outer-draws", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    for directory in (OUT / "tables", OUT / "data", OUT / "qa", OUT / "bootstrap_city", OUT / "methods"):
        directory.mkdir(parents=True, exist_ok=True)
    frame = LEGACY.load_frame()
    central = central_analysis(frame)
    central.to_csv(OUT / "tables" / "01_city_group_ratios_by_budget.csv", index=False)
    city_results = run_bootstrap(frame, args.city_replicates, args.workers)
    draws = two_level_draws(city_results, args.outer_draws)
    absolute, differences = interval_tables(central, draws)
    absolute.to_csv(OUT / "tables" / "02_equal_city_group_ratio_intervals.csv", index=False)
    differences.to_csv(OUT / "tables" / "03_paired_differences_from_uniform.csv", index=False)
    np.savez_compressed(
        OUT / "data" / "two_level_group_budget_draws.npz",
        draws=draws,
        budgets=BUDGETS,
        rules=np.asarray(RULES),
        groups=np.asarray(GROUPS),
    )
    qa = audit(frame, central, absolute, differences, args.city_replicates, args.outer_draws)
    (OUT / "qa" / "group_budget_sensitivity_audit.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(OUT / "methods" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
