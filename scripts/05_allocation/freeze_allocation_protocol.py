from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(str(analysis_store_root().joinpath('Result')))
INPUT = ROOT / "Result4_5_neighbourhood_typology" / "data" / "block_group_GHV_primary_classification.csv.gz"
THERMAL_INPUT = ROOT / "Result3" / "Result3_support_aligned" / "data" / "support_aligned_model_frame_34566.csv"
OUT = ROOT / "Result5" / "versions" / "step1_frozen_multicriteria_design"

PRIMARY_SCOPE = "G_positive"
PRIMARY_BUDGET = 0.10
PARETO_LAMBDAS = np.linspace(0.0, 1.0, 21)
EXPECTED_ROWS = 34_566
EXPECTED_CITIES = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_geoid(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def hard_allocate(capacity: np.ndarray, budget: float, score: np.ndarray, geoid: np.ndarray) -> np.ndarray:
    allocation = np.zeros_like(capacity, dtype=float)
    valid_score = np.where(np.isfinite(score), score, -np.inf)
    order = np.lexsort((geoid, -valid_score))
    remaining = float(budget)
    for index in order:
        if remaining <= 1e-9:
            break
        take = min(float(capacity[index]), remaining)
        if take > 0:
            allocation[index] = take
            remaining -= take
    if remaining > max(1e-6, 1e-10 * max(budget, 1.0)):
        raise RuntimeError(f"Budget not exhausted: {remaining}")
    return allocation


def build_social_condition(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Build a transparent three-domain score without the ambiguous transit item.

    Poverty represents the economic domain; renter occupancy and no-vehicle
    households share the housing-and-mobility domain; age 65+ represents the
    age-sensitivity domain. The indicator columns are already oriented 0-1
    empirical-percentile scores. Rare missing values are replaced by the
    within-city indicator median. Income, transit and alternative imputation
    enter later sensitivity analyses rather than the frozen primary score.
    """

    columns = [
        "V_indicator_poverty_rate_pct",
        "V_indicator_no_vehicle_households_pct",
        "V_indicator_renter_occupied_pct",
        "V_indicator_age_65_plus_pct",
    ]
    values = frame[columns].copy()
    missing = {column: int(values[column].isna().sum()) for column in columns}
    for column in columns:
        medians = values.groupby(frame["city_key"])[column].transform("median")
        values[column] = values[column].fillna(medians).fillna(values[column].median())

    economic = values["V_indicator_poverty_rate_pct"]
    housing_mobility = values[
        ["V_indicator_no_vehicle_households_pct", "V_indicator_renter_occupied_pct"]
    ].mean(axis=1)
    age = values["V_indicator_age_65_plus_pct"]
    score = pd.concat([economic, housing_mobility, age], axis=1).mean(axis=1)
    return score, missing


def allocation_metrics(
    population: np.ndarray,
    land: np.ndarray,
    heat: np.ndarray,
    social: np.ndarray,
    allocation: np.ndarray,
) -> tuple[float, float, float]:
    delta_canopy_pp = 100.0 * allocation / land
    q_heat = float(np.sum(population * heat * delta_canopy_pp))
    q_social = float(np.sum(population * social * delta_canopy_pp))
    q_population = float(np.sum(population * delta_canopy_pp))
    return q_heat, q_social, q_population


def continuous_overlap(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.maximum(a, b).sum())
    return float(np.minimum(a, b).sum() / denominator) if denominator > 0 else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for subdir in ("tables", "qa", "methods"):
        (OUT / subdir).mkdir(exist_ok=True)

    frame = pd.read_csv(INPUT, dtype={"GEOID_str": str}, low_memory=False)
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    thermal = pd.read_csv(
        THERMAL_INPUT,
        dtype={"GEOID_str": str},
        usecols=["city_key", "GEOID_str", "mean_LST_C"],
        low_memory=False,
    )
    thermal["GEOID_str"] = normalize_geoid(thermal["GEOID_str"])
    frame = frame.merge(
        thermal,
        on=["city_key", "GEOID_str"],
        how="left",
        validate="one_to_one",
    )
    city_reference = frame.groupby("city_key", sort=False).apply(
        lambda d: np.average(d["mean_LST_C"], weights=d["population"]),
        include_groups=False,
    )
    frame["residential_reference_LST_C"] = frame["city_key"].map(city_reference)
    frame["H"] = np.maximum(
        frame["mean_LST_C"] - frame["residential_reference_LST_C"], 0.0
    )
    frame["V"], social_missing = build_social_condition(frame)

    required = ["population", "land_area_m2", "G", "H", "V"]
    if frame[required].isna().any().any():
        missing = frame[required].isna().sum().to_dict()
        raise RuntimeError(f"Missing primary inputs: {missing}")
    if len(frame) != EXPECTED_ROWS or frame["city_key"].nunique() != EXPECTED_CITIES:
        raise RuntimeError("Frozen development frame count mismatch")

    rows: list[dict] = []
    overlaps: list[dict] = []
    diagnostics: list[dict] = []
    max_budget_error = 0.0
    max_capacity_violation = 0.0
    max_oracle_error = 0.0
    endpoint_mismatch = 0
    monotonic_violations = 0

    for city, sub in frame.groupby("city_key", sort=True):
        sub = sub.sort_values("GEOID_str", kind="mergesort").copy()
        population = sub["population"].to_numpy(float)
        land = sub["land_area_m2"].to_numpy(float)
        gap = sub["G"].to_numpy(float)
        heat = sub["H"].to_numpy(float)
        social = sub["V"].to_numpy(float)
        geoid = sub["GEOID_str"].to_numpy(str)

        capacity = np.where(gap > 0, land * gap / 100.0, 0.0)
        total_capacity = float(capacity.sum())
        budget = PRIMARY_BUDGET * total_capacity
        if total_capacity <= 0 or budget <= 0:
            raise RuntimeError(f"No positive capacity in {city}")

        density = population / land
        marginal_heat = 100.0 * population * heat / land
        marginal_social = 100.0 * population * social / land
        marginal_population = 100.0 * population / land

        allocations: dict[str, np.ndarray] = {
            "uniform": capacity * PRIMARY_BUDGET,
            "canopy_deficit_first": hard_allocate(capacity, budget, gap, geoid),
            "surface_heat_only": hard_allocate(capacity, budget, heat, geoid),
            "social_condition_only": hard_allocate(capacity, budget, social, geoid),
            "population_oracle": hard_allocate(capacity, budget, marginal_population, geoid),
            "heat_context_oracle": hard_allocate(capacity, budget, marginal_heat, geoid),
            "social_condition_oracle": hard_allocate(capacity, budget, marginal_social, geoid),
        }

        oracle_heat = allocation_metrics(
            population, land, heat, social, allocations["heat_context_oracle"]
        )[0]
        oracle_social = allocation_metrics(
            population, land, heat, social, allocations["social_condition_oracle"]
        )[1]
        oracle_population = allocation_metrics(
            population, land, heat, social, allocations["population_oracle"]
        )[2]
        if min(oracle_heat, oracle_social, oracle_population) <= 0:
            raise RuntimeError(f"Non-positive oracle objective in {city}")

        for lam in PARETO_LAMBDAS:
            combined = lam * marginal_heat / oracle_heat + (1.0 - lam) * marginal_social / oracle_social
            allocations[f"pareto_{lam:.2f}"] = hard_allocate(capacity, budget, combined, geoid)

        pareto_trace: list[tuple[float, float, float]] = []
        for rule, allocation in allocations.items():
            allocated = float(allocation.sum())
            budget_error = abs(allocated - budget) / budget
            capacity_violation = float(np.maximum(allocation - capacity, 0.0).max(initial=0.0))
            max_budget_error = max(max_budget_error, budget_error)
            max_capacity_violation = max(max_capacity_violation, capacity_violation)

            q_heat, q_social, q_population = allocation_metrics(
                population, land, heat, social, allocation
            )
            heat_efficiency = q_heat / oracle_heat
            social_efficiency = q_social / oracle_social
            population_efficiency = q_population / oracle_population
            rows.append(
                {
                    "city_key": city,
                    "city_display": sub["city_display"].iloc[0],
                    "scope": PRIMARY_SCOPE,
                    "budget_fraction": PRIMARY_BUDGET,
                    "rule": rule,
                    "eligible_units": int(np.sum(capacity > 0)),
                    "eligible_capacity_m2": total_capacity,
                    "allocated_area_m2": allocated,
                    "heat_context_quantity": q_heat,
                    "social_condition_quantity": q_social,
                    "population_quantity": q_population,
                    "heat_context_efficiency": heat_efficiency,
                    "social_condition_efficiency": social_efficiency,
                    "population_efficiency": population_efficiency,
                    "heat_regret": 1.0 - heat_efficiency,
                    "social_regret": 1.0 - social_efficiency,
                }
            )
            if rule in {
                "heat_context_oracle",
                "social_condition_oracle",
                "population_oracle",
            }:
                expected = {
                    "heat_context_oracle": heat_efficiency,
                    "social_condition_oracle": social_efficiency,
                    "population_oracle": population_efficiency,
                }[rule]
                max_oracle_error = max(max_oracle_error, abs(expected - 1.0))
            if rule.startswith("pareto_"):
                pareto_trace.append((float(rule.split("_")[1]), heat_efficiency, social_efficiency))

        if not np.allclose(allocations["pareto_0.00"], allocations["social_condition_oracle"]):
            endpoint_mismatch += 1
        if not np.allclose(allocations["pareto_1.00"], allocations["heat_context_oracle"]):
            endpoint_mismatch += 1

        pareto_trace.sort()
        heat_values = np.asarray([item[1] for item in pareto_trace])
        social_values = np.asarray([item[2] for item in pareto_trace])
        monotonic_violations += int(np.sum(np.diff(heat_values) < -1e-10))
        monotonic_violations += int(np.sum(np.diff(social_values) > 1e-10))

        heat_oracle = allocations["heat_context_oracle"]
        social_oracle = allocations["social_condition_oracle"]
        simple_heat = allocations["surface_heat_only"]
        simple_social = allocations["social_condition_only"]
        overlaps.append(
            {
                "city_key": city,
                "heat_social_oracle_overlap": continuous_overlap(heat_oracle, social_oracle),
                "surface_heat_only_vs_heat_oracle_overlap": continuous_overlap(simple_heat, heat_oracle),
                "social_condition_only_vs_social_oracle_overlap": continuous_overlap(simple_social, social_oracle),
            }
        )

        city_rows = {row["rule"]: row for row in rows if row["city_key"] == city}
        diagnostics.append(
            {
                "city_key": city,
                "city_display": sub["city_display"].iloc[0],
                "heat_oracle_social_efficiency": city_rows["heat_context_oracle"]["social_condition_efficiency"],
                "social_oracle_heat_efficiency": city_rows["social_condition_oracle"]["heat_context_efficiency"],
                "surface_heat_only_heat_efficiency": city_rows["surface_heat_only"]["heat_context_efficiency"],
                "social_condition_only_social_efficiency": city_rows["social_condition_only"]["social_condition_efficiency"],
                "population_oracle_heat_efficiency": city_rows["population_oracle"]["heat_context_efficiency"],
                "population_oracle_social_efficiency": city_rows["population_oracle"]["social_condition_efficiency"],
            }
        )

    outcomes = pd.DataFrame(rows)
    overlap_frame = pd.DataFrame(overlaps)
    diagnostic_frame = pd.DataFrame(diagnostics)

    outcomes.to_csv(OUT / "tables" / "01_city_rule_and_pareto_outcomes.csv", index=False)
    overlap_frame.to_csv(OUT / "tables" / "02_city_allocation_overlap.csv", index=False)
    diagnostic_frame.to_csv(OUT / "tables" / "03_city_tradeoff_diagnostics.csv", index=False)

    named_rules = outcomes.loc[~outcomes["rule"].str.startswith("pareto_")].copy()
    national = (
        named_rules.groupby("rule", as_index=False)
        .agg(
            cities=("city_key", "nunique"),
            mean_heat_efficiency=("heat_context_efficiency", "mean"),
            median_heat_efficiency=("heat_context_efficiency", "median"),
            mean_social_efficiency=("social_condition_efficiency", "mean"),
            median_social_efficiency=("social_condition_efficiency", "median"),
            mean_population_efficiency=("population_efficiency", "mean"),
            median_population_efficiency=("population_efficiency", "median"),
        )
        .sort_values("rule")
    )
    national.to_csv(OUT / "tables" / "04_equal_city_rule_summary.csv", index=False)

    pareto = outcomes.loc[outcomes["rule"].str.startswith("pareto_")].copy()
    pareto["lambda_heat"] = pareto["rule"].str.split("_").str[1].astype(float)
    pareto_summary = (
        pareto.groupby("lambda_heat", as_index=False)
        .agg(
            mean_heat_efficiency=("heat_context_efficiency", "mean"),
            q25_heat_efficiency=("heat_context_efficiency", lambda x: np.quantile(x, 0.25)),
            q75_heat_efficiency=("heat_context_efficiency", lambda x: np.quantile(x, 0.75)),
            mean_social_efficiency=("social_condition_efficiency", "mean"),
            q25_social_efficiency=("social_condition_efficiency", lambda x: np.quantile(x, 0.25)),
            q75_social_efficiency=("social_condition_efficiency", lambda x: np.quantile(x, 0.75)),
        )
    )
    pareto_summary.to_csv(OUT / "tables" / "05_equal_city_pareto_frontier.csv", index=False)

    material_tradeoff = (
        (diagnostic_frame["heat_oracle_social_efficiency"] < 0.90)
        | (diagnostic_frame["social_oracle_heat_efficiency"] < 0.90)
    )
    both_material = (
        (diagnostic_frame["heat_oracle_social_efficiency"] < 0.90)
        & (diagnostic_frame["social_oracle_heat_efficiency"] < 0.90)
    )

    qa = {
        "status": "PASS",
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "thermal_input": str(THERMAL_INPUT),
        "thermal_input_sha256": sha256(THERMAL_INPUT),
        "development_frame_rows": int(len(frame)),
        "cities": int(frame["city_key"].nunique()),
        "primary_scope": PRIMARY_SCOPE,
        "primary_budget_fraction": PRIMARY_BUDGET,
        "primary_heat_context": "positive deviation from the population-weighted mean LST among analysed residential units within each city",
        "pareto_lambda_count": int(len(PARETO_LAMBDAS)),
        "primary_social_indicators": [
            "poverty",
            "no_vehicle",
            "renter_occupancy",
            "age_65_plus",
        ],
        "excluded_from_primary_social_score": [
            "median_household_income_due_to_10.8_percent_missingness",
            "public_transit_due_to_context-dependent_direction",
        ],
        "primary_social_indicator_missing_before_city_median_imputation": social_missing,
        "max_relative_budget_error": max_budget_error,
        "max_capacity_violation_m2": max_capacity_violation,
        "max_oracle_efficiency_error": max_oracle_error,
        "pareto_endpoint_mismatches": endpoint_mismatch,
        "pareto_monotonicity_violations": monotonic_violations,
        "cities_with_material_one_axis_tradeoff": int(material_tradeoff.sum()),
        "cities_with_material_two_axis_tradeoff": int(both_material.sum()),
        "median_heat_social_oracle_overlap": float(overlap_frame["heat_social_oracle_overlap"].median()),
        "median_surface_heat_only_heat_efficiency": float(
            diagnostic_frame["surface_heat_only_heat_efficiency"].median()
        ),
        "median_social_condition_only_social_efficiency": float(
            diagnostic_frame["social_condition_only_social_efficiency"].median()
        ),
        "quality_gate": {
            "budgets_exact": max_budget_error < 1e-10,
            "capacity_respected": max_capacity_violation < 1e-6,
            "oracles_verified": max_oracle_error < 1e-10,
            "pareto_endpoints_verified": endpoint_mismatch == 0,
            "pareto_monotone": monotonic_violations == 0,
            "nondegenerate_tradeoff": int(material_tradeoff.sum()) >= 10,
        },
    }
    qa["quality_gate_passed"] = all(qa["quality_gate"].values())
    if not qa["quality_gate_passed"]:
        qa["status"] = "FAIL"
    write_json(OUT / "qa" / "step1_design_audit.json", qa)

    if not qa["quality_gate_passed"]:
        raise RuntimeError(f"Quality gate failed: {qa['quality_gate']}")

    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
