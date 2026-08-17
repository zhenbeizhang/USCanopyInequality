from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = analysis_store_root()
ROOT = (
    PROJECT
    / "Result"
    / "Result4"
    / "versions"
    / "joint_context_boundary_spatial_audit"
)
PRIMARY = ROOT / "data" / "fully_contained_primary_GHV27.csv.gz"
RANGES = (
    PROJECT
    / "Result"
    / "Result3"
    / "Result3_support_aligned"
    / "spatial_block_bootstrap"
    / "tables"
    / "city_residual_range_and_block_sizes.csv"
)
POINT = ROOT / "tables" / "03_cooccurrence_benchmarks.csv"
DRAWS = ROOT / "diagnostics" / "cooccurrence_two_level_block_bootstrap_draws.csv.gz"
BOUNDARY = ROOT / "tables" / "07_boundary_support_assignment_agreement.csv"
TAIL = ROOT / "tables" / "04_tail_threshold_surface.csv"
CONSTRUCT = ROOT / "tables" / "05_construct_threshold_sensitivity.csv"
R5_CITY = ROOT / "tables" / "08_result5_city_boundary_sensitivity.csv.gz"
R5_NATIONAL = ROOT / "tables" / "09_result5_national_boundary_sensitivity.csv"
R5_EXISTING = (
    PROJECT
    / "Result"
    / "Result5"
    / "versions"
    / "step2_multicriteria_pareto"
    / "tables"
    / "01_city_rule_outcomes.csv"
)
SUPPORTS = (
    "fully_contained_primary",
    "intersection_ge_0p95_block_allocated",
    "intersection_ge_0p80_block_allocated",
    "any_intersection_block_allocated",
)
OFFSETS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def context_from_levels(frame: pd.DataFrame) -> np.ndarray:
    g = frame["G_level"].ge(2)
    h = frame["H_level"].eq(3)
    v = frame["V_level"].eq(3)
    return np.select(
        [g & h & v, g & h & ~v, g & ~h & v, g & ~h & ~v, ~g & (h | v)],
        ["A", "B", "C", "D", "E"],
        default="F",
    )


def assign(coords: np.ndarray, anchor: np.ndarray, size: float, offset: tuple[float, float]) -> np.ndarray:
    grid = np.floor((coords - anchor + np.asarray(offset) * size) / size).astype(int)
    return np.asarray([f"{x}:{y}" for x, y in grid], dtype=object)


def spatial_ratio(frame: pd.DataFrame, scale: str) -> tuple[float, list[float]]:
    ranges = pd.read_csv(RANGES).set_index("city_key")
    observed_city: list[float] = []
    expected_by_origin: list[list[float]] = [[] for _ in OFFSETS]
    for city, local in frame.groupby("city_key", sort=True):
        local = local.sort_values("GEOID_str")
        event = local["context_id"].eq("A").to_numpy(int)
        g = local["G_level"].ge(2).to_numpy(int)
        h = local["H_level"].eq(3).to_numpy(int)
        v = local["V_level"].eq(3).to_numpy(int)
        observed_city.append(float(event.mean()))
        coords = local[["x_5070_m", "y_5070_m"]].to_numpy(float)
        anchor = coords.min(axis=0)
        size = float(ranges.loc[city, f"{scale}_block_size_km"]) * 1000
        for origin, offset in enumerate(OFFSETS):
            table = pd.DataFrame(
                {
                    "block": assign(coords, anchor, size, offset),
                    "g": g,
                    "h": h,
                    "v": v,
                }
            ).groupby("block").agg(n=("g", "size"), g=("g", "sum"), h=("h", "sum"), v=("v", "sum"))
            expected = float(
                (table["g"] * table["h"] * table["v"] / table["n"].pow(2)).sum()
                / table["n"].sum()
            )
            expected_by_origin[origin].append(expected)
    observed = float(np.mean(observed_city))
    ratios = [observed / float(np.mean(values)) for values in expected_by_origin]
    ratio_mean_expectation = observed / float(
        np.mean([np.mean(values) for values in expected_by_origin])
    )
    return ratio_mean_expectation, ratios


def main() -> int:
    frame = pd.read_csv(PRIMARY, dtype={"GEOID_str": str}, low_memory=False)
    reconstructed_type = (
        "G"
        + frame["G_level"].astype(int).astype(str)
        + "H"
        + frame["H_level"].astype(int).astype(str)
        + "V"
        + frame["V_level"].astype(int).astype(str)
    )
    reconstructed_context = context_from_levels(frame)
    event = frame["context_id"].eq("A")
    city_observed = frame.assign(event=event).groupby("city_key")["event"].mean()
    city_expected = frame.groupby("city_key", sort=True).apply(
        lambda sub: sub["G_level"].ge(2).mean()
        * sub["H_level"].eq(3).mean()
        * sub["V_level"].eq(3).mean(),
        include_groups=False,
    )
    independent_city_ratio = float(city_observed.mean() / city_expected.mean())

    point = pd.read_csv(POINT)
    point_rows = point.loc[
        point["event"].eq("joint_context")
        & point["grid_origin"].isin(["not_applicable", "origin_mean"])
    ].set_index("benchmark")
    independent_spatial: dict[str, float] = {}
    origin_ranges: dict[str, list[float]] = {}
    for scale in ("small", "main", "large"):
        value, origins = spatial_ratio(frame, scale)
        independent_spatial[scale] = value
        origin_ranges[scale] = [min(origins), max(origins)]

    draws = pd.read_csv(DRAWS)
    quantiles = (
        draws.groupby(["event", "benchmark"])["ratio"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
    )
    interval_diff = 0.0
    name_map = {
        "city_marginal_independence": "city_marginal",
        "small_block_conditioned": "small",
        "main_block_conditioned": "main",
        "large_block_conditioned": "large",
    }
    for benchmark, draw_name in name_map.items():
        row = point_rows.loc[benchmark]
        values = quantiles.loc[("joint_context", draw_name)]
        interval_diff = max(
            interval_diff,
            abs(float(row["bootstrap_p025"]) - float(values.loc[0.025])),
            abs(float(row["bootstrap_p975"]) - float(values.loc[0.975])),
        )

    boundary = pd.read_csv(BOUNDARY)
    tail = pd.read_csv(TAIL)
    construct = pd.read_csv(CONSTRUCT)
    r5_city = pd.read_csv(R5_CITY)
    r5_national = pd.read_csv(R5_NATIONAL)
    existing = pd.read_csv(R5_EXISTING)
    primary_r5 = r5_city.loc[r5_city["support"].eq("fully_contained_primary")]
    reproduced = primary_r5.merge(
        existing,
        on=["city_key", "city_display", "rule"],
        suffixes=("_audit", "_existing"),
        validate="one_to_one",
    )
    check_columns = (
        "budget_m2",
        "allocated_m2",
        "heat_context_efficiency",
        "social_condition_efficiency",
        "population_efficiency",
    )
    reproduction_difference = max(
        float(
            np.abs(
                reproduced[f"{column}_audit"].to_numpy(float)
                - reproduced[f"{column}_existing"].to_numpy(float)
            ).max()
        )
        for column in check_columns
    )
    budget_error = float(
        np.abs(r5_city["allocated_m2"] - r5_city["budget_m2"]).max()
    )
    pareto = r5_national.loc[r5_national["rule"].eq("pareto_0.50")].set_index("support")
    primary_pareto = pareto.loc["fully_contained_primary"]
    max_pareto_support_change = float(
        np.max(
            np.abs(
                pareto[["heat_context_efficiency", "social_condition_efficiency", "population_efficiency"]].to_numpy(float)
                - primary_pareto[["heat_context_efficiency", "social_condition_efficiency", "population_efficiency"]].to_numpy(float)
            )
        )
    )
    audit = {
        "status": "PASS",
        "rows": len(frame),
        "cities": frame["city_key"].nunique(),
        "unique_city_geoid": not frame.duplicated(["city_key", "GEOID_str"]).any(),
        "type_code_exact_reconstruction": bool(reconstructed_type.eq(frame["type_code"]).all()),
        "context_exact_reconstruction": bool((reconstructed_context == frame["context_id"]).all()),
        "types_nonempty": int(frame["type_code"].nunique()),
        "contexts_nonempty": int(frame["context_id"].nunique()),
        "joint_context_count": int(event.sum()),
        "joint_context_population": float(frame.loc[event, "population"].sum()),
        "independent_city_marginal_ratio": independent_city_ratio,
        "stored_city_marginal_ratio": float(point_rows.loc["city_marginal_independence", "observed_expected_ratio"]),
        "independent_block_conditioned_ratios": independent_spatial,
        "stored_block_conditioned_ratios": {
            scale: float(point_rows.loc[f"{scale}_block_conditioned", "observed_expected_ratio"])
            for scale in ("small", "main", "large")
        },
        "grid_origin_ratio_ranges": origin_ranges,
        "bootstrap_rows": len(draws),
        "bootstrap_interval_max_abs_recalculation_difference": interval_diff,
        "tail_surface_minimum_city_marginal_ratio": float(tail["city_marginal_ratio"].min()),
        "tail_surface_minimum_main_block_ratio": float(tail["main_block_conditioned_ratio"].min()),
        "construct_minimum_context_agreement": float(construct["context_agreement"].min()),
        "construct_minimum_joint_context_jaccard": float(construct["joint_context_jaccard"].min()),
        "boundary_minimum_context_agreement": float(boundary["context_agreement"].min()),
        "boundary_minimum_joint_context_jaccard": float(boundary["joint_context_jaccard"].min()),
        "r5_rows": len(r5_city),
        "r5_supports": int(r5_city["support"].nunique()),
        "r5_cities_per_support": r5_city.groupby("support")["city_key"].nunique().to_dict(),
        "r5_rules_per_support": r5_city.groupby("support")["rule"].nunique().to_dict(),
        "r5_budget_max_abs_error_m2": budget_error,
        "r5_primary_reproduction_max_abs_difference": reproduction_difference,
        "r5_balanced_pareto_max_abs_support_change": max_pareto_support_change,
        "interpretation": {
            "city_marginal_enrichment_is_descriptive": True,
            "spatial_conditioning_materially_attenuates_ratio": True,
            "local_labels_are_not_scale_invariant": True,
            "primary_claim_must_use_modest_spatially_conditioned_colocation": True,
            "universal_priority_or_causal_interaction_claim_allowed": False,
        },
    }
    numeric_match = max(
        abs(
            audit["independent_city_marginal_ratio"]
            - audit["stored_city_marginal_ratio"]
        ),
        max(
            abs(
                audit["independent_block_conditioned_ratios"][scale]
                - audit["stored_block_conditioned_ratios"][scale]
            )
            for scale in ("small", "main", "large")
        ),
    )
    audit["key_numeric_max_abs_independent_difference"] = numeric_match
    required = (
        audit["rows"] == 34_566
        and audit["cities"] == 50
        and audit["unique_city_geoid"]
        and audit["type_code_exact_reconstruction"]
        and audit["context_exact_reconstruction"]
        and audit["types_nonempty"] == 27
        and audit["contexts_nonempty"] == 6
        and numeric_match < 1e-12
        and interval_diff < 1e-12
        and budget_error < 1e-4
        and reproduction_difference < 1e-8
    )
    if not required:
        audit["status"] = "FAIL"
    write_json(ROOT / "qa" / "step5_independent_technical_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
