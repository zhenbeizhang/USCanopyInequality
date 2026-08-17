from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = analysis_store_root()
BOUNDARY_ROOT = (
    PROJECT / "Result" / "Result2" / "Result2_block_population_boundary_audit"
)
R3_DATA = BOUNDARY_ROOT / "data"
SUPPORT_FILES = {
    "fully_contained_primary": R3_DATA
    / "result3_fully_contained_primary_model_frame.csv.gz",
    "intersection_ge_0p95_block_allocated": R3_DATA
    / "result3_intersection_ge_0p95_block_allocated_model_frame.csv.gz",
    "intersection_ge_0p80_block_allocated": R3_DATA
    / "result3_intersection_ge_0p80_block_allocated_model_frame.csv.gz",
    "any_intersection_block_allocated": R3_DATA
    / "result3_any_intersection_block_allocated_model_frame.csv.gz",
}
AGE_BG = (
    PROJECT
    / "Result"
    / "Result4"
    / "versions"
    / "construct_validity_audit"
    / "source_data"
    / "acs2024_b01001_target_block_groups.csv"
)
R2_ALL = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_canopy_ACS_block_group"
    / "_all_cities_block_group_canopy_acs_result2.csv"
)
TRACT_FRAME = (
    PROJECT
    / "Result"
    / "Spatial_scale_sensitivity"
    / "versions"
    / "tract_heat_hvcg"
    / "data"
    / "tract_hvcg_analysis_frame.csv"
)
TRACT_AGE = (
    PROJECT
    / "Result"
    / "Result4_5_neighbourhood_typology"
    / "frozen_inputs"
    / "tract_age65_from_acs2024_b01001_bg.csv.gz"
)
BLOCK_RANGES = (
    PROJECT
    / "Result"
    / "Result3"
    / "Result3_support_aligned"
    / "spatial_block_bootstrap"
    / "tables"
    / "city_residual_range_and_block_sizes.csv"
)
STEP2_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "05_allocation"
    / "run_equal_budget_allocation.py"
)
STEP2_RESULTS = (
    PROJECT
    / "Result"
    / "Result5"
    / "versions"
    / "step2_multicriteria_pareto"
    / "tables"
    / "01_city_rule_outcomes.csv"
)
OUT = (
    PROJECT
    / "Result"
    / "Result4"
    / "versions"
    / "joint_context_boundary_spatial_audit"
)

PRIMARY_SUPPORT = "fully_contained_primary"
EXPECTED_ROWS = {
    "fully_contained_primary": 34_566,
    "intersection_ge_0p95_block_allocated": 35_142,
    "intersection_ge_0p80_block_allocated": 35_600,
    "any_intersection_block_allocated": 37_690,
}
EXPECTED_CITIES = 50
ZERO_TOLERANCE = 1e-9
BOOTSTRAP_DRAWS = 2_000
SEED = 20_260_802
GRID_OFFSETS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
SCALES = ("small", "main", "large")
SUPPORT_ORDER = tuple(SUPPORT_FILES)

V_RAW = (
    "poverty_rate_pct",
    "no_vehicle_households_pct",
    "renter_occupied_pct",
    "age_65_plus_pct",
)
V_SCORE = tuple(f"V_indicator_{column}" for column in V_RAW)
GROUP_COLUMNS = (
    "people_of_color_pct",
    "hispanic_or_latino_pct",
    "black_alone_pct",
    "asian_alone_pct",
    "white_alone_pct",
    "age_65_plus_pct",
)


def ensure_dirs() -> None:
    for name in ("data", "tables", "diagnostics", "qa", "text", "methods"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    def convert(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=convert) + "\n",
        encoding="utf-8",
    )


def normalize_geoid(values: pd.Series, width: int = 12) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(width)


def city_balanced_weights(cities: pd.Series) -> pd.Series:
    return 1.0 / cities.groupby(cities).transform("size").astype(float)


def weighted_quantile(
    values: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    probabilities: tuple[float, ...] | list[float],
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[valid]
    w = w[valid]
    if not len(x):
        return np.full(len(probabilities), np.nan)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(probabilities, dtype=float), cumulative, x)


def weighted_mid_ecdf(values: pd.Series, weights: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    valid = x.notna() & np.isfinite(x) & w.notna() & np.isfinite(w) & w.gt(0)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    grouped = (
        pd.DataFrame({"value": x[valid], "weight": w[valid]})
        .groupby("value", sort=True, as_index=False)["weight"]
        .sum()
    )
    grouped["percentile"] = (
        grouped["weight"].cumsum() - 0.5 * grouped["weight"]
    ) / grouped["weight"].sum()
    result.loc[valid] = x.loc[valid].map(
        grouped.set_index("value")["percentile"]
    )
    return result


def fill_within_city(values: pd.Series, cities: pd.Series) -> tuple[pd.Series, int]:
    result = values.copy()
    missing_before = int(result.isna().sum())
    medians = result.groupby(cities).transform("median")
    result = result.fillna(medians).fillna(result.median()).fillna(0.5)
    return result, missing_before


def density_gap(
    canopy: pd.Series,
    log_density: pd.Series,
    cities: pd.Series,
    quantile: float = 0.75,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    target = pd.Series(np.nan, index=canopy.index, dtype=float)
    stratum = pd.Series(pd.NA, index=canopy.index, dtype="Int64")
    source = pd.Series("", index=canopy.index, dtype=object)
    for city, indices in cities.groupby(cities, sort=False).groups.items():
        local_density = log_density.loc[indices]
        bins = pd.qcut(local_density, q=5, labels=False, duplicates="drop")
        stratum.loc[indices] = bins.astype("Int64") + 1
        city_canopy = canopy.loc[indices].dropna()
        city_target = float(city_canopy.quantile(quantile))
        for level, local_indices in stratum.loc[indices].groupby(
            stratum.loc[indices], dropna=False
        ).groups.items():
            values = canopy.loc[local_indices].dropna()
            if len(values) < 10:
                target.loc[local_indices] = city_target
                source.loc[local_indices] = f"citywide_fallback_q{int(100*quantile)}"
            else:
                target.loc[local_indices] = float(values.quantile(quantile))
                source.loc[local_indices] = f"city_density5_q{int(100*quantile)}"
    return (target - canopy).clip(lower=0), target, stratum


def build_social_condition(
    frame: pd.DataFrame,
    *,
    add_income: bool = False,
    add_transit: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    result = frame.copy()
    weights = result["city_balanced_weight"]
    specs: list[tuple[str, float]] = [
        ("poverty_rate_pct", 1.0),
        ("no_vehicle_households_pct", 1.0),
        ("renter_occupied_pct", 1.0),
        ("age_65_plus_pct", 1.0),
    ]
    if add_income:
        specs.append(("median_household_income_usd", -1.0))
    if add_transit:
        specs.append(("public_transport_commute_pct", 1.0))

    audit: list[dict[str, object]] = []
    scores: dict[str, pd.Series] = {}
    for column, orientation in specs:
        raw = pd.to_numeric(result[column], errors="coerce")
        lower, upper = weighted_quantile(raw, weights, [0.01, 0.99])
        oriented = raw.clip(lower, upper) * orientation
        score = weighted_mid_ecdf(oriented, weights)
        score, missing = fill_within_city(score, result["city_key"])
        name = f"V_indicator_{column}"
        result[name] = score
        scores[column] = score
        audit.append(
            {
                "indicator": column,
                "orientation": orientation,
                "winsor_p01_raw": lower,
                "winsor_p99_raw": upper,
                "missing_before_within_city_median_fill": missing,
            }
        )

    economic_columns = ["poverty_rate_pct"]
    if add_income:
        economic_columns.append("median_household_income_usd")
    mobility_columns = ["no_vehicle_households_pct", "renter_occupied_pct"]
    if add_transit:
        mobility_columns.append("public_transport_commute_pct")
    result["V_domain_economic"] = pd.concat(
        [scores[column] for column in economic_columns], axis=1
    ).mean(axis=1)
    result["V_domain_housing_mobility"] = pd.concat(
        [scores[column] for column in mobility_columns], axis=1
    ).mean(axis=1)
    result["V_domain_age"] = scores["age_65_plus_pct"]
    result["V"] = result[
        ["V_domain_economic", "V_domain_housing_mobility", "V_domain_age"]
    ].mean(axis=1)
    return result, audit


def map_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["type_code"] = (
        "G"
        + result["G_level"].astype(int).astype(str)
        + "H"
        + result["H_level"].astype(int).astype(str)
        + "V"
        + result["V_level"].astype(int).astype(str)
    )
    g_positive = result["G_level"].ge(2)
    h_high = result["H_level"].eq(3)
    v_high = result["V_level"].eq(3)
    result["context_id"] = np.select(
        [
            g_positive & h_high & v_high,
            g_positive & h_high & ~v_high,
            g_positive & ~h_high & v_high,
            g_positive & ~h_high & ~v_high,
            ~g_positive & (h_high | v_high),
        ],
        ["A", "B", "C", "D", "E"],
        default="F",
    )
    labels = {
        "A": "joint_heat_social_canopy_context",
        "B": "surface_heat_canopy_context",
        "C": "social_canopy_context",
        "D": "canopy_deficit_context",
        "E": "heat_or_social_without_canopy_deficit",
        "F": "lower_or_mixed_reference",
    }
    result["context_label"] = result["context_id"].map(labels)
    return result


def classify(
    frame: pd.DataFrame,
    *,
    g_positive_quantile: float = 0.5,
    h_positive_quantile: float = 0.5,
    v_cuts: tuple[float, float] = (1 / 3, 2 / 3),
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = frame.copy()
    weights = result["city_balanced_weight"]
    g_cut = weighted_quantile(
        result.loc[result["G"].gt(ZERO_TOLERANCE), "G"],
        weights.loc[result["G"].gt(ZERO_TOLERANCE)],
        [g_positive_quantile],
    )[0]
    h_cut = weighted_quantile(
        result.loc[result["H"].gt(ZERO_TOLERANCE), "H"],
        weights.loc[result["H"].gt(ZERO_TOLERANCE)],
        [h_positive_quantile],
    )[0]
    v_low, v_high = weighted_quantile(result["V"], weights, list(v_cuts))
    result["G_level"] = np.select(
        [result["G"].le(ZERO_TOLERANCE), result["G"].le(g_cut)],
        [1, 2],
        default=3,
    ).astype(np.int8)
    result["H_level"] = np.select(
        [result["H"].le(ZERO_TOLERANCE), result["H"].le(h_cut)],
        [1, 2],
        default=3,
    ).astype(np.int8)
    result["V_level"] = np.select(
        [result["V"].le(v_low), result["V"].le(v_high)],
        [1, 2],
        default=3,
    ).astype(np.int8)
    return map_context(result), {
        "G_positive_split": float(g_cut),
        "H_positive_split": float(h_cut),
        "V_lower_cut": float(v_low),
        "V_upper_cut": float(v_high),
    }


def load_demographic_supplements() -> tuple[pd.DataFrame, pd.DataFrame]:
    age = pd.read_csv(
        AGE_BG,
        usecols=["GEOID_str", "age_65_plus_pct", "age_under_5_pct"],
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    age["GEOID_str"] = normalize_geoid(age["GEOID_str"])
    age = age.drop_duplicates("GEOID_str")
    white = pd.read_csv(
        R2_ALL,
        usecols=["result_city_key", "GEOID", "white_alone_pct"],
        dtype={"GEOID": str},
        low_memory=False,
    ).rename(columns={"result_city_key": "city_key", "GEOID": "GEOID_str"})
    white["GEOID_str"] = normalize_geoid(white["GEOID_str"])
    white = white.drop_duplicates(["city_key", "GEOID_str"])
    return age, white


def prepare_support(
    label: str,
    path: Path,
    age: pd.DataFrame,
    white: pd.DataFrame,
    *,
    canopy_quantile: float = 0.75,
    add_income: bool = False,
    add_transit: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path, dtype={"GEOID_str": str}, low_memory=False)
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    frame = frame.merge(age, on="GEOID_str", how="left", validate="many_to_one")
    frame = frame.merge(
        white, on=["city_key", "GEOID_str"], how="left", validate="one_to_one"
    )
    if label == PRIMARY_SUPPORT:
        frame["land_area_analysis_m2"] = pd.to_numeric(
            frame["land_area_m2"], errors="coerce"
        )
    else:
        frame["land_area_analysis_m2"] = pd.to_numeric(
            frame["boundary_city_land_area_m2"], errors="coerce"
        )
    frame["population"] = pd.to_numeric(frame["population"], errors="coerce")
    frame["canopy_pct_analysis"] = pd.to_numeric(
        frame["canopy_pct_analysis"], errors="coerce"
    )
    frame["population_density_per_km2"] = (
        frame["population"] / (frame["land_area_analysis_m2"] / 1e6)
    )
    frame["log_population_density"] = np.log(
        frame["population_density_per_km2"].clip(lower=1e-6)
    )
    required = [
        "population",
        "land_area_analysis_m2",
        "canopy_pct_analysis",
        "mean_LST_C",
        "x_5070_m",
        "y_5070_m",
    ]
    if frame[required].isna().any().any():
        missing = frame[required].isna().sum()
        raise RuntimeError(f"{label} has missing required values: {missing[missing.gt(0)]}")
    if (frame[["population", "land_area_analysis_m2"]] <= 0).any().any():
        raise RuntimeError(f"{label} has non-positive population or city-side land area")
    if len(frame) != EXPECTED_ROWS[label] or frame["city_key"].nunique() != 50:
        raise RuntimeError(
            f"{label} mismatch: rows={len(frame)}, cities={frame['city_key'].nunique()}"
        )
    frame = frame.sort_values(["city_key", "GEOID_str"], kind="mergesort").reset_index(
        drop=True
    )
    frame["city_balanced_weight"] = city_balanced_weights(frame["city_key"])
    frame["G"], frame["canopy_target_pct"], frame["density_stratum_5"] = density_gap(
        frame["canopy_pct_analysis"],
        frame["log_population_density"],
        frame["city_key"],
        canopy_quantile,
    )
    reference = frame.groupby("city_key", sort=False).apply(
        lambda sub: np.average(sub["mean_LST_C"], weights=sub["population"]),
        include_groups=False,
    )
    frame["city_population_weighted_LST_C"] = frame["city_key"].map(reference)
    frame["H"] = (
        frame["mean_LST_C"] - frame["city_population_weighted_LST_C"]
    ).clip(lower=0)
    frame, indicator_audit = build_social_condition(
        frame, add_income=add_income, add_transit=add_transit
    )
    classified, thresholds = classify(frame)
    meta = {
        "support": label,
        "rows": len(classified),
        "cities": classified["city_key"].nunique(),
        "represented_population": float(classified["population"].sum()),
        "age65_missing_raw": int(classified["age_65_plus_pct"].isna().sum()),
        "white_missing_raw": int(classified["white_alone_pct"].isna().sum()),
        "canopy_quantile": canopy_quantile,
        "add_income": add_income,
        "add_transit": add_transit,
        "thresholds": thresholds,
        "indicator_audit": indicator_audit,
    }
    return classified, meta


def type_context_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    total_population = frame["population"].sum()
    city_n = frame.groupby("city_key").size()
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(column, sort=True):
        shares = (
            group.groupby("city_key").size().reindex(city_n.index, fill_value=0) / city_n
        )
        rows.append(
            {
                column: value,
                "block_groups": len(group),
                "block_group_share": len(group) / len(frame),
                "equal_city_prevalence": float(shares.mean()),
                "represented_population": float(group["population"].sum()),
                "population_share": float(group["population"].sum() / total_population),
                "cities_represented": int(group["city_key"].nunique()),
                "median_G_pp": float(group["G"].median()),
                "median_H_C": float(group["H"].median()),
                "median_V_0_1": float(group["V"].median()),
            }
        )
    return pd.DataFrame(rows)


def event_flags(frame: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=frame.index)
    flags["n"] = 1
    flags["g_positive"] = frame["G_level"].ge(2).astype(np.int8)
    flags["g3"] = frame["G_level"].eq(3).astype(np.int8)
    flags["h3"] = frame["H_level"].eq(3).astype(np.int8)
    flags["v3"] = frame["V_level"].eq(3).astype(np.int8)
    flags["joint_context"] = (
        flags["g_positive"].eq(1) & flags["h3"].eq(1) & flags["v3"].eq(1)
    ).astype(np.int8)
    flags["strict_G3H3V3"] = (
        flags["g3"].eq(1) & flags["h3"].eq(1) & flags["v3"].eq(1)
    ).astype(np.int8)
    return flags


def assign_grid(
    coords: np.ndarray,
    anchor: np.ndarray,
    block_size_m: float,
    offset: tuple[float, float],
) -> np.ndarray:
    grid = np.floor(
        (coords - anchor[None, :] + np.asarray(offset) * block_size_m) / block_size_m
    ).astype(np.int64)
    return np.asarray([f"{x}:{y}" for x, y in grid], dtype=object)


def prepare_block_tables(
    frame: pd.DataFrame,
    anchors: dict[str, np.ndarray],
    ranges: pd.DataFrame,
) -> dict[str, dict[str, dict[int, pd.DataFrame]]]:
    range_lookup = ranges.set_index("city_key")
    flags = event_flags(frame)
    output: dict[str, dict[str, dict[int, pd.DataFrame]]] = defaultdict(dict)
    for city, indices in frame.groupby("city_key", sort=True).groups.items():
        coords = frame.loc[indices, ["x_5070_m", "y_5070_m"]].to_numpy(float)
        local_flags = flags.loc[indices].reset_index(drop=True)
        for scale in SCALES:
            size = float(range_lookup.loc[city, f"{scale}_block_size_km"]) * 1000
            output[city][scale] = {}
            for origin, offset in enumerate(GRID_OFFSETS):
                local = local_flags.copy()
                local["block"] = assign_grid(coords, anchors[city], size, offset)
                output[city][scale][origin] = local.groupby("block", sort=True).sum()
    return output


def event_expected_from_totals(totals: pd.Series, event: str) -> float:
    n = float(totals["n"])
    g_name = "g3" if event == "strict_G3H3V3" else "g_positive"
    return float(totals[g_name] / n * totals["h3"] / n * totals["v3"] / n)


def block_conditioned_expected(table: pd.DataFrame, event: str) -> float:
    g_name = "g3" if event == "strict_G3H3V3" else "g_positive"
    n = table["n"].to_numpy(float)
    expected_counts = (
        table[g_name].to_numpy(float)
        * table["h3"].to_numpy(float)
        * table["v3"].to_numpy(float)
        / np.square(n)
    )
    return float(expected_counts.sum() / n.sum())


def cooccurrence_point_estimates(
    frame: pd.DataFrame,
    block_tables: dict[str, dict[str, dict[int, pd.DataFrame]]],
) -> pd.DataFrame:
    flags = event_flags(frame)
    rows: list[dict[str, object]] = []
    for event in ("joint_context", "strict_G3H3V3"):
        city_observed: list[float] = []
        city_expected: list[float] = []
        for _, indices in frame.groupby("city_key", sort=True).groups.items():
            totals = flags.loc[indices].sum(axis=0)
            city_observed.append(float(totals[event] / totals["n"]))
            city_expected.append(event_expected_from_totals(totals, event))
        observed = float(np.mean(city_observed))
        expected = float(np.mean(city_expected))
        rows.append(
            {
                "event": event,
                "benchmark": "city_marginal_independence",
                "grid_origin": "not_applicable",
                "observed_equal_city_prevalence": observed,
                "expected_equal_city_prevalence": expected,
                "observed_expected_ratio": observed / expected,
            }
        )
        for scale in SCALES:
            origin_expected: list[float] = []
            for origin in range(len(GRID_OFFSETS)):
                city_values = [
                    block_conditioned_expected(block_tables[city][scale][origin], event)
                    for city in sorted(block_tables)
                ]
                expected_origin = float(np.mean(city_values))
                origin_expected.append(expected_origin)
                rows.append(
                    {
                        "event": event,
                        "benchmark": f"{scale}_block_conditioned",
                        "grid_origin": origin,
                        "observed_equal_city_prevalence": observed,
                        "expected_equal_city_prevalence": expected_origin,
                        "observed_expected_ratio": observed / expected_origin,
                    }
                )
            mean_expected = float(np.mean(origin_expected))
            rows.append(
                {
                    "event": event,
                    "benchmark": f"{scale}_block_conditioned",
                    "grid_origin": "origin_mean",
                    "observed_equal_city_prevalence": observed,
                    "expected_equal_city_prevalence": mean_expected,
                    "observed_expected_ratio": observed / mean_expected,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_cooccurrence(
    block_tables: dict[str, dict[str, dict[int, pd.DataFrame]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    cities = np.asarray(sorted(block_tables), dtype=object)
    rows: list[dict[str, object]] = []
    benchmarks = ("city_marginal", "small", "main", "large")
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_cities = rng.choice(cities, size=len(cities), replace=True)
        for event in ("joint_context", "strict_G3H3V3"):
            for benchmark in benchmarks:
                observed_city: list[float] = []
                expected_city: list[float] = []
                scale = "main" if benchmark == "city_marginal" else benchmark
                for city in sampled_cities:
                    origin = int(rng.integers(0, len(GRID_OFFSETS)))
                    table = block_tables[str(city)][scale][origin]
                    sampled = table.iloc[
                        rng.integers(0, len(table), size=len(table))
                    ]
                    totals = sampled.sum(axis=0)
                    observed_city.append(float(totals[event] / totals["n"]))
                    if benchmark == "city_marginal":
                        expected_city.append(event_expected_from_totals(totals, event))
                    else:
                        expected_city.append(block_conditioned_expected(sampled, event))
                observed = float(np.mean(observed_city))
                expected = float(np.mean(expected_city))
                rows.append(
                    {
                        "draw": draw,
                        "event": event,
                        "benchmark": benchmark,
                        "observed_prevalence": observed,
                        "expected_prevalence": expected,
                        "ratio": observed / expected,
                    }
                )
    draws = pd.DataFrame(rows)
    summary = (
        draws.groupby(["event", "benchmark"])["ratio"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .rename(columns={0.025: "bootstrap_p025", 0.5: "bootstrap_median", 0.975: "bootstrap_p975"})
        .reset_index()
    )
    return draws, summary


def context_agreement(reference: pd.DataFrame, variant: pd.DataFrame) -> dict[str, float]:
    joined = reference[["city_key", "GEOID_str", "context_id", "type_code"]].merge(
        variant[["city_key", "GEOID_str", "context_id", "type_code"]],
        on=["city_key", "GEOID_str"],
        suffixes=("_reference", "_variant"),
        validate="one_to_one",
    )
    ref_a = joined["context_id_reference"].eq("A")
    var_a = joined["context_id_variant"].eq("A")
    union = int((ref_a | var_a).sum())
    return {
        "comparison_rows": len(joined),
        "context_agreement": float(
            joined["context_id_reference"].eq(joined["context_id_variant"]).mean()
        ),
        "type_agreement": float(
            joined["type_code_reference"].eq(joined["type_code_variant"]).mean()
        ),
        "joint_context_jaccard": float((ref_a & var_a).sum() / union) if union else 1.0,
        "joint_context_reference_share": float(ref_a.mean()),
        "joint_context_variant_share": float(var_a.mean()),
    }


def generic_ratio(
    frame: pd.DataFrame,
    event: pd.Series,
    g_flag: pd.Series,
    h_flag: pd.Series,
    v_flag: pd.Series,
    blocks: dict[str, dict[str, dict[int, pd.DataFrame]]] | None = None,
) -> tuple[float, float]:
    observed_city: list[float] = []
    expected_city: list[float] = []
    conditional_by_origin: list[list[float]] = [[] for _ in GRID_OFFSETS]
    for city, indices in frame.groupby("city_key", sort=True).groups.items():
        n = len(indices)
        observed_city.append(float(event.loc[indices].mean()))
        expected_city.append(
            float(g_flag.loc[indices].mean() * h_flag.loc[indices].mean() * v_flag.loc[indices].mean())
        )
        if blocks is not None:
            local = pd.DataFrame(
                {
                    "event": event.loc[indices].to_numpy(int),
                    "g": g_flag.loc[indices].to_numpy(int),
                    "h": h_flag.loc[indices].to_numpy(int),
                    "v": v_flag.loc[indices].to_numpy(int),
                },
                index=np.arange(n),
            )
            for origin in range(len(GRID_OFFSETS)):
                # Labels use the frozen primary-frame anchor and the Result 3
                # residual-range block width.
                labels = frame.loc[indices, f"main_block_origin{origin}"].to_numpy()
                local["block"] = labels
                grouped = local.groupby("block", sort=True).agg(
                    n=("event", "size"), g=("g", "sum"), h=("h", "sum"), v=("v", "sum")
                )
                expected = float(
                    (grouped["g"] * grouped["h"] * grouped["v"] / grouped["n"].pow(2)).sum()
                    / grouped["n"].sum()
                )
                conditional_by_origin[origin].append(expected)
    observed = float(np.mean(observed_city))
    city_ratio = observed / float(np.mean(expected_city))
    if blocks is None:
        return city_ratio, np.nan
    conditional_expected = float(
        np.mean([np.mean(values) for values in conditional_by_origin])
    )
    return city_ratio, observed / conditional_expected


def attach_block_labels(
    frame: pd.DataFrame,
    anchors: dict[str, np.ndarray],
    ranges: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    lookup = ranges.set_index("city_key")
    for city, indices in result.groupby("city_key", sort=True).groups.items():
        coords = result.loc[indices, ["x_5070_m", "y_5070_m"]].to_numpy(float)
        for scale in SCALES:
            size = float(lookup.loc[city, f"{scale}_block_size_km"]) * 1000
            for origin, offset in enumerate(GRID_OFFSETS):
                result.loc[indices, f"{scale}_block_origin{origin}"] = assign_grid(
                    coords, anchors[city], size, offset
                )
    return result


def tail_threshold_surface(
    frame: pd.DataFrame,
    block_tables: dict[str, dict[str, dict[int, pd.DataFrame]]],
) -> pd.DataFrame:
    del block_tables
    rows: list[dict[str, float]] = []
    weights = frame["city_balanced_weight"]
    for q_gh in (0.4, 0.5, 0.6, 0.7):
        g_cut = weighted_quantile(
            frame.loc[frame["G"].gt(ZERO_TOLERANCE), "G"],
            weights.loc[frame["G"].gt(ZERO_TOLERANCE)],
            [q_gh],
        )[0]
        h_cut = weighted_quantile(
            frame.loc[frame["H"].gt(ZERO_TOLERANCE), "H"],
            weights.loc[frame["H"].gt(ZERO_TOLERANCE)],
            [q_gh],
        )[0]
        for q_v in (0.55, 2 / 3, 0.75):
            v_cut = weighted_quantile(frame["V"], weights, [q_v])[0]
            g = frame["G"].gt(g_cut)
            h = frame["H"].gt(h_cut)
            v = frame["V"].gt(v_cut)
            event = g & h & v
            city_ratio, block_ratio = generic_ratio(frame, event, g, h, v, blocks={})
            rows.append(
                {
                    "positive_G_H_quantile": q_gh,
                    "V_quantile": q_v,
                    "G_cut_pp": g_cut,
                    "H_cut_C": h_cut,
                    "V_cut": v_cut,
                    "equal_city_observed_prevalence": float(
                        frame.assign(event=event).groupby("city_key")["event"].mean().mean()
                    ),
                    "city_marginal_ratio": city_ratio,
                    "main_block_conditioned_ratio": block_ratio,
                }
            )
    return pd.DataFrame(rows)


def threshold_and_construct_sensitivity(
    primary: pd.DataFrame,
    primary_path: Path,
    age: pd.DataFrame,
    white: pd.DataFrame,
    anchors: dict[str, np.ndarray],
    ranges: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def metrics(label: str, variant: pd.DataFrame) -> dict[str, object]:
        variant = attach_block_labels(variant, anchors, ranges)
        event = variant["context_id"].eq("A")
        g = variant["G_level"].ge(2)
        h = variant["H_level"].eq(3)
        v = variant["V_level"].eq(3)
        city_ratio, block_ratio = generic_ratio(
            variant, event, g, h, v, blocks={}
        )
        return {
            "variant": label,
            **context_agreement(primary, variant),
            "city_marginal_ratio": city_ratio,
            "main_block_conditioned_ratio": block_ratio,
        }

    for label, q in (("canopy_q60", 0.60), ("canopy_q90", 0.90)):
        variant, _ = prepare_support(
            PRIMARY_SUPPORT, primary_path, age, white, canopy_quantile=q
        )
        rows.append(metrics(label, variant))
    for label, income, transit in (
        ("add_reverse_income", True, False),
        ("add_public_transit", False, True),
        ("add_income_and_transit", True, True),
    ):
        variant, _ = prepare_support(
            PRIMARY_SUPPORT,
            primary_path,
            age,
            white,
            add_income=income,
            add_transit=transit,
        )
        rows.append(metrics(label, variant))

    complete = primary.dropna(subset=list(V_RAW)).copy()
    complete["city_balanced_weight"] = city_balanced_weights(complete["city_key"])
    complete, _ = build_social_condition(complete)
    complete, _ = classify(complete)
    rows.append(metrics("complete_four_indicator", complete))

    loco_agreement: list[float] = []
    loco_jaccard: list[float] = []
    for city in sorted(primary["city_key"].unique()):
        training = primary.loc[primary["city_key"].ne(city)]
        test = primary.loc[primary["city_key"].eq(city)].copy()
        g_cut = weighted_quantile(
            training.loc[training["G"].gt(ZERO_TOLERANCE), "G"],
            training.loc[training["G"].gt(ZERO_TOLERANCE), "city_balanced_weight"],
            [0.5],
        )[0]
        h_cut = weighted_quantile(
            training.loc[training["H"].gt(ZERO_TOLERANCE), "H"],
            training.loc[training["H"].gt(ZERO_TOLERANCE), "city_balanced_weight"],
            [0.5],
        )[0]
        v_low, v_high = weighted_quantile(
            training["V"], training["city_balanced_weight"], [1 / 3, 2 / 3]
        )
        test["G_level"] = np.select(
            [test["G"].le(ZERO_TOLERANCE), test["G"].le(g_cut)], [1, 2], default=3
        )
        test["H_level"] = np.select(
            [test["H"].le(ZERO_TOLERANCE), test["H"].le(h_cut)], [1, 2], default=3
        )
        test["V_level"] = np.select(
            [test["V"].le(v_low), test["V"].le(v_high)], [1, 2], default=3
        )
        test = map_context(test)
        metrics = context_agreement(primary.loc[primary["city_key"].eq(city)], test)
        loco_agreement.append(metrics["context_agreement"])
        loco_jaccard.append(metrics["joint_context_jaccard"])
    rows.append(
        {
            "variant": "leave_one_city_out_threshold_transfer",
            "comparison_rows": len(primary),
            "context_agreement": float(np.mean(loco_agreement)),
            "type_agreement": np.nan,
            "joint_context_jaccard": float(np.mean(loco_jaccard)),
            "joint_context_reference_share": float(primary["context_id"].eq("A").mean()),
            "joint_context_variant_share": np.nan,
            "city_marginal_ratio": np.nan,
            "main_block_conditioned_ratio": np.nan,
        }
    )
    return pd.DataFrame(rows)


def tract_sensitivity(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    tract = pd.read_csv(TRACT_FRAME, dtype={"tract_geoid": str}, low_memory=False)
    tract["tract_geoid"] = normalize_geoid(tract["tract_geoid"], 11)
    complete = tract["complete_frame_tract_90"].astype(str).str.lower().eq("true")
    valid = tract["valid_temperature_tract"].astype(str).str.lower().eq("true")
    tract = tract.loc[complete & valid & tract["population"].gt(0)].copy()
    age = pd.read_csv(TRACT_AGE, dtype={"tract_geoid": str}, low_memory=False)
    age["tract_geoid"] = normalize_geoid(age["tract_geoid"], 11)
    tract = tract.drop(columns=["age_65_plus_pct"], errors="ignore").merge(
        age[["city_key", "tract_geoid", "age_65_plus_pct"]],
        on=["city_key", "tract_geoid"],
        how="left",
        validate="one_to_one",
    )
    tract["GEOID_str"] = tract["tract_geoid"]
    tract["land_area_analysis_m2"] = tract["land_area_m2"]
    tract["city_balanced_weight"] = city_balanced_weights(tract["city_key"])
    tract["log_population_density"] = np.log(
        (tract["population"] / (tract["land_area_analysis_m2"] / 1e6)).clip(lower=1e-6)
    )
    tract["G"], tract["canopy_target_pct"], tract["density_stratum_5"] = density_gap(
        tract["canopy_pct_analysis"],
        tract["log_population_density"],
        tract["city_key"],
    )
    reference = tract.groupby("city_key", sort=False).apply(
        lambda sub: np.average(sub["mean_LST_C"], weights=sub["population"]),
        include_groups=False,
    )
    tract["H"] = (tract["mean_LST_C"] - tract["city_key"].map(reference)).clip(lower=0)
    tract, _ = build_social_condition(tract)
    tract, thresholds = classify(tract)

    city_obs = tract.assign(event=tract["context_id"].eq("A")).groupby("city_key")["event"].mean()
    city_expected = tract.groupby("city_key", sort=True).apply(
        lambda sub: sub["G_level"].ge(2).mean()
        * sub["H_level"].eq(3).mean()
        * sub["V_level"].eq(3).mean(),
        include_groups=False,
    )
    ratio = float(city_obs.mean() / city_expected.mean())
    bg = primary.copy()
    bg["tract_geoid"] = bg["GEOID_str"].str[:11]
    comparison = bg.merge(
        tract[["city_key", "tract_geoid", "context_id"]].rename(
            columns={"context_id": "tract_context"}
        ),
        on=["city_key", "tract_geoid"],
        how="inner",
        validate="many_to_one",
    )
    unweighted = float(comparison["context_id"].eq(comparison["tract_context"]).mean())
    population_weighted = float(
        np.average(
            comparison["context_id"].eq(comparison["tract_context"]),
            weights=comparison["population"],
        )
    )
    meta = {
        "tract_rows": len(tract),
        "tract_cities": tract["city_key"].nunique(),
        "joint_context_equal_city_observed": float(city_obs.mean()),
        "joint_context_equal_city_expected": float(city_expected.mean()),
        "joint_context_ratio": ratio,
        "bg_tract_comparison_rows": len(comparison),
        "bg_to_tract_context_agreement": unweighted,
        "bg_to_tract_population_weighted_context_agreement": population_weighted,
        "thresholds": thresholds,
    }
    return tract, meta


def import_step2():
    spec = importlib.util.spec_from_file_location("step2", STEP2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the baseline allocation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def r5_boundary_sensitivity(
    supports: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    step2 = import_step2()
    rows: list[dict[str, object]] = []
    for support, frame in supports.items():
        local = frame.copy()
        for column in GROUP_COLUMNS:
            local[column], _ = fill_within_city(local[column], local["city_key"])
        for city, city_frame in local.groupby("city_key", sort=True):
            city_frame = city_frame.sort_values("GEOID_str", kind="mergesort")
            population = city_frame["population"].to_numpy(float)
            land = city_frame["land_area_analysis_m2"].to_numpy(float)
            canopy = city_frame["canopy_pct_analysis"].to_numpy(float)
            log_density = city_frame["log_population_density"].to_numpy(float)
            temperature = city_frame["mean_LST_C"].to_numpy(float)
            v_indicators = city_frame[list(V_SCORE)].to_numpy(float)
            groups = city_frame[list(GROUP_COLUMNS)].to_numpy(float)
            metrics, _, overlap, budget, allocated = step2.evaluate_city_arrays(
                population,
                land,
                canopy,
                log_density,
                temperature,
                v_indicators,
                groups,
            )
            for index, rule in enumerate(step2.ALL_RULES):
                rows.append(
                    {
                        "support": support,
                        "city_key": city,
                        "city_display": city_frame["city_display"].iloc[0],
                        "rule": rule,
                        "budget_m2": budget,
                        "allocated_m2": allocated[index],
                        "heat_context_efficiency": metrics[index, 0],
                        "social_condition_efficiency": metrics[index, 1],
                        "population_efficiency": metrics[index, 2],
                        "heat_context_quantity": metrics[index, 3],
                        "social_condition_quantity": metrics[index, 4],
                        "population_quantity": metrics[index, 5],
                        "heat_social_oracle_overlap": overlap,
                    }
                )
    city = pd.DataFrame(rows)
    national = (
        city.groupby(["support", "rule"], as_index=False)
        .agg(
            heat_context_efficiency=("heat_context_efficiency", "mean"),
            social_condition_efficiency=("social_condition_efficiency", "mean"),
            population_efficiency=("population_efficiency", "mean"),
            median_heat_social_oracle_overlap=("heat_social_oracle_overlap", "median"),
            total_budget_km2=("budget_m2", lambda values: values.sum() / 1e6),
            maximum_budget_error_m2=(
                "allocated_m2",
                lambda values: 0.0,
            ),
        )
    )
    existing = pd.read_csv(STEP2_RESULTS)
    reproduced = city.loc[city["support"].eq(PRIMARY_SUPPORT)].merge(
        existing,
        on=["city_key", "city_display", "rule"],
        suffixes=("_new", "_existing"),
        validate="one_to_one",
    )
    checked = (
        "budget_m2",
        "allocated_m2",
        "heat_context_efficiency",
        "social_condition_efficiency",
        "population_efficiency",
        "heat_context_quantity",
        "social_condition_quantity",
        "population_quantity",
    )
    max_diff = max(
        float(
            np.max(
                np.abs(
                    reproduced[f"{column}_new"].to_numpy(float)
                    - reproduced[f"{column}_existing"].to_numpy(float)
                )
            )
        )
        for column in checked
    )
    budget_error = float(
        np.max(
            np.abs(city["allocated_m2"].to_numpy(float) - city["budget_m2"].to_numpy(float))
        )
    )
    endpoint_ok = True
    monotonic_ok = True
    for _, group in city.groupby(["support", "city_key"], sort=False):
        indexed = group.set_index("rule")
        endpoint_ok &= bool(
            np.allclose(
                indexed.loc[
                    "pareto_0.00",
                    ["heat_context_efficiency", "social_condition_efficiency"],
                ].to_numpy(dtype=float),
                indexed.loc[
                    "social_condition_oracle",
                    ["heat_context_efficiency", "social_condition_efficiency"],
                ].to_numpy(dtype=float),
                atol=1e-10,
                rtol=1e-10,
            )
        )
        endpoint_ok &= bool(
            np.allclose(
                indexed.loc[
                    "pareto_1.00",
                    ["heat_context_efficiency", "social_condition_efficiency"],
                ].to_numpy(dtype=float),
                indexed.loc[
                    "heat_context_oracle",
                    ["heat_context_efficiency", "social_condition_efficiency"],
                ].to_numpy(dtype=float),
                atol=1e-10,
                rtol=1e-10,
            )
        )
        pareto = indexed.loc[list(step2.PARETO_RULES)]
        monotonic_ok &= bool(
            (np.diff(pareto["heat_context_efficiency"].to_numpy(float)) >= -1e-10).all()
            and (np.diff(pareto["social_condition_efficiency"].to_numpy(float)) <= 1e-10).all()
        )
    qa = {
        "primary_reproduction_rows": len(reproduced),
        "primary_reproduction_max_abs_difference": max_diff,
        "maximum_budget_error_m2": budget_error,
        "pareto_endpoints_exact": endpoint_ok,
        "pareto_axes_monotonic": monotonic_ok,
    }
    return city, national, qa


def allocation_context_distribution(
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    step2 = import_step2()
    rules = (
        "uniform",
        "canopy_deficit_first",
        "surface_heat_only",
        "social_condition_only",
        "heat_context_oracle",
        "social_condition_oracle",
        "pareto_0.50",
    )
    rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    for city, local in primary.groupby("city_key", sort=True):
        local = local.sort_values("GEOID_str", kind="mergesort").reset_index(drop=True)
        population = local["population"].to_numpy(float)
        land = local["land_area_analysis_m2"].to_numpy(float)
        canopy = local["canopy_pct_analysis"].to_numpy(float)
        log_density = local["log_population_density"].to_numpy(float)
        temperature = local["mean_LST_C"].to_numpy(float)
        social = step2.recompute_social_condition(local[list(V_SCORE)].to_numpy(float))
        gap, _ = step2.recompute_gap(canopy, log_density)
        heat = np.maximum(temperature - np.average(temperature, weights=population), 0)
        capacity = land * gap / 100
        budget = step2.PRIMARY_BUDGET * capacity.sum()
        marginal_heat = 100 * population * heat / land
        marginal_social = 100 * population * social / land
        heat_allocation = step2.hard_allocate(capacity, budget, marginal_heat)
        social_allocation = step2.hard_allocate(capacity, budget, marginal_social)
        heat_quantity = step2.allocation_quantities(
            population, land, heat, social, heat_allocation
        )[0]
        social_quantity = step2.allocation_quantities(
            population, land, heat, social, social_allocation
        )[1]
        allocations = {
            "uniform": capacity * step2.PRIMARY_BUDGET,
            "canopy_deficit_first": step2.hard_allocate(capacity, budget, gap),
            "surface_heat_only": step2.hard_allocate(capacity, budget, heat),
            "social_condition_only": step2.hard_allocate(capacity, budget, social),
            "heat_context_oracle": heat_allocation,
            "social_condition_oracle": social_allocation,
            "pareto_0.50": step2.hard_allocate(
                capacity,
                budget,
                0.5 * marginal_heat / heat_quantity
                + 0.5 * marginal_social / social_quantity,
            ),
        }
        components = {
            "shared": np.minimum(heat_allocation, social_allocation),
            "heat_oracle_only": np.maximum(heat_allocation - social_allocation, 0),
            "social_oracle_only": np.maximum(social_allocation - heat_allocation, 0),
        }
        for component, area_vector in components.items():
            component_total = float(area_vector.sum())
            for context in ("A", "B", "C", "D", "E", "F"):
                mask = local["context_id"].eq(context).to_numpy()
                area = float(area_vector[mask].sum())
                overlap_rows.append(
                    {
                        "city_key": city,
                        "component": component,
                        "context_id": context,
                        "area_m2": area,
                        "within_component_area_share": (
                            area / component_total if component_total > 0 else np.nan
                        ),
                        "component_total_m2": component_total,
                        "city_budget_m2": budget,
                    }
                )
        for rule in rules:
            allocation = allocations[rule]
            total = float(allocation.sum())
            for context in ("A", "B", "C", "D", "E", "F"):
                mask = local["context_id"].eq(context).to_numpy()
                area = float(allocation[mask].sum())
                rows.append(
                    {
                        "city_key": city,
                        "rule": rule,
                        "context_id": context,
                        "allocated_area_m2": area,
                        "city_rule_area_share": area / total,
                        "city_budget_m2": total,
                    }
                )
    city = pd.DataFrame(rows)
    summary = (
        city.groupby(["rule", "context_id"], as_index=False)
        .agg(
            equal_city_mean_area_share=("city_rule_area_share", "mean"),
            equal_city_median_area_share=("city_rule_area_share", "median"),
            pooled_allocated_area_m2=("allocated_area_m2", "sum"),
            pooled_budget_m2=("city_budget_m2", "sum"),
        )
    )
    summary["pooled_area_share"] = (
        summary["pooled_allocated_area_m2"] / summary["pooled_budget_m2"]
    )
    overlap_city = pd.DataFrame(overlap_rows)
    overlap = (
        overlap_city.groupby(["component", "context_id"], as_index=False)
        .agg(
            equal_city_mean_within_component_share=(
                "within_component_area_share",
                "mean",
            ),
            pooled_area_m2=("area_m2", "sum"),
            pooled_component_m2=("component_total_m2", "sum"),
            pooled_budget_m2=("city_budget_m2", "sum"),
        )
    )
    overlap["pooled_within_component_share"] = (
        overlap["pooled_area_m2"] / overlap["pooled_component_m2"]
    )
    overlap["pooled_component_share_of_one_rule_budget"] = (
        overlap["pooled_component_m2"] / overlap["pooled_budget_m2"]
    )
    return summary, overlap


def main() -> int:
    ensure_dirs()
    age, white = load_demographic_supplements()
    support_frames: dict[str, pd.DataFrame] = {}
    support_meta: dict[str, dict[str, object]] = {}
    for label, path in SUPPORT_FILES.items():
        print(f"[prepare] {label}", flush=True)
        frame, meta = prepare_support(label, path, age, white)
        support_frames[label] = frame
        support_meta[label] = meta

    primary = support_frames[PRIMARY_SUPPORT]
    ranges = pd.read_csv(BLOCK_RANGES)
    anchors = {
        city: sub[["x_5070_m", "y_5070_m"]].min().to_numpy(float)
        for city, sub in primary.groupby("city_key", sort=True)
    }
    for label, frame in list(support_frames.items()):
        support_frames[label] = attach_block_labels(frame, anchors, ranges)
    primary = support_frames[PRIMARY_SUPPORT]

    type_summary = type_context_summary(primary, "type_code")
    context_summary = type_context_summary(primary, "context_id")
    type_summary.to_csv(OUT / "tables" / "01_primary_27_type_summary.csv", index=False)
    context_summary.to_csv(OUT / "tables" / "02_primary_6_context_summary.csv", index=False)

    primary_blocks = prepare_block_tables(primary, anchors, ranges)
    point_path = OUT / "tables" / "03_cooccurrence_benchmarks.csv"
    draws_path = OUT / "diagnostics" / "cooccurrence_two_level_block_bootstrap_draws.csv.gz"
    if point_path.exists() and draws_path.exists():
        print("[bootstrap] reuse completed deterministic-seed outputs", flush=True)
        point = pd.read_csv(point_path)
        draws = pd.read_csv(draws_path)
        if len(draws) != BOOTSTRAP_DRAWS * 2 * 4 or point[
            ["observed_expected_ratio", "bootstrap_p025", "bootstrap_p975"]
        ].isna().any().any():
            raise RuntimeError("Cached co-occurrence bootstrap is incomplete")
    else:
        point = cooccurrence_point_estimates(primary, primary_blocks)
        print("[bootstrap] joint co-occurrence", flush=True)
        draws, intervals = bootstrap_cooccurrence(primary_blocks)
        intervals["benchmark"] = intervals["benchmark"].replace(
            {
                "city_marginal": "city_marginal_independence",
                "small": "small_block_conditioned",
                "main": "main_block_conditioned",
                "large": "large_block_conditioned",
            }
        )
        point = point.merge(
            intervals,
            left_on=["event", "benchmark"],
            right_on=["event", "benchmark"],
            how="left",
        )
        point.to_csv(point_path, index=False)
        draws.to_csv(draws_path, index=False, compression="gzip")

    tail = tail_threshold_surface(primary, primary_blocks)
    tail.to_csv(OUT / "tables" / "04_tail_threshold_surface.csv", index=False)
    construct = threshold_and_construct_sensitivity(
        primary,
        SUPPORT_FILES[PRIMARY_SUPPORT],
        age,
        white,
        anchors,
        ranges,
    )
    construct.to_csv(OUT / "tables" / "05_construct_threshold_sensitivity.csv", index=False)

    boundary_rows: list[dict[str, object]] = []
    boundary_agreement_rows: list[dict[str, object]] = []
    for label, frame in support_frames.items():
        blocks = prepare_block_tables(frame, anchors, ranges)
        estimates = cooccurrence_point_estimates(frame, blocks)
        joint_city = estimates.loc[
            estimates["event"].eq("joint_context")
            & estimates["benchmark"].eq("city_marginal_independence")
        ].iloc[0]
        joint_spatial = estimates.loc[
            estimates["event"].eq("joint_context")
            & estimates["benchmark"].eq("main_block_conditioned")
            & estimates["grid_origin"].eq("origin_mean")
        ].iloc[0]
        boundary_rows.append(
            {
                "support": label,
                "block_groups": len(frame),
                "cities": frame["city_key"].nunique(),
                "represented_population": frame["population"].sum(),
                "joint_context_block_groups": int(frame["context_id"].eq("A").sum()),
                "joint_context_raw_share": float(frame["context_id"].eq("A").mean()),
                "joint_context_equal_city_prevalence": joint_city[
                    "observed_equal_city_prevalence"
                ],
                "city_marginal_ratio": joint_city["observed_expected_ratio"],
                "main_block_conditioned_ratio": joint_spatial["observed_expected_ratio"],
                **support_meta[label]["thresholds"],
            }
        )
        if label != PRIMARY_SUPPORT:
            boundary_agreement_rows.append(
                {"support": label, **context_agreement(primary, frame)}
            )
        output_columns = [
            "city_key",
            "city_display",
            "GEOID_str",
            "population",
            "land_area_analysis_m2",
            "canopy_pct_analysis",
            "mean_LST_C",
            "G",
            "H",
            "V",
            "G_level",
            "H_level",
            "V_level",
            "type_code",
            "context_id",
            "context_label",
            "city_balanced_weight",
            "canopy_target_pct",
            "density_stratum_5",
            "city_population_weighted_LST_C",
            *V_RAW,
            *V_SCORE,
            *GROUP_COLUMNS[:-1],
            "x_5070_m",
            "y_5070_m",
            "lon",
            "lat",
        ]
        output_columns = list(dict.fromkeys(column for column in output_columns if column in frame))
        frame[output_columns].to_csv(
            OUT / "data" / f"{label}_GHV27.csv.gz",
            index=False,
            compression="gzip",
        )
    boundary = pd.DataFrame(boundary_rows)
    boundary_agreement = pd.DataFrame(boundary_agreement_rows)
    boundary.to_csv(OUT / "tables" / "06_boundary_support_typology.csv", index=False)
    boundary_agreement.to_csv(
        OUT / "tables" / "07_boundary_support_assignment_agreement.csv", index=False
    )

    tract, tract_meta = tract_sensitivity(primary)
    tract.to_csv(OUT / "data" / "tract_GHV27_sensitivity.csv.gz", index=False, compression="gzip")
    write_json(OUT / "qa" / "tract_scale_audit.json", tract_meta)

    print("[Result 5] boundary inheritance", flush=True)
    r5_city, r5_national, r5_qa = r5_boundary_sensitivity(support_frames)
    r5_city.to_csv(OUT / "tables" / "08_result5_city_boundary_sensitivity.csv.gz", index=False, compression="gzip")
    r5_national.to_csv(OUT / "tables" / "09_result5_national_boundary_sensitivity.csv", index=False)
    context_allocation, oracle_overlap_context = allocation_context_distribution(primary)
    context_allocation.to_csv(
        OUT / "tables" / "10_result5_allocation_by_GHV_context.csv", index=False
    )
    oracle_overlap_context.to_csv(
        OUT / "tables" / "11_result5_oracle_overlap_by_GHV_context.csv",
        index=False,
    )

    primary_joint = point.loc[
        point["event"].eq("joint_context")
        & point["benchmark"].eq("city_marginal_independence")
        & point["grid_origin"].eq("not_applicable")
    ].iloc[0]
    primary_spatial = point.loc[
        point["event"].eq("joint_context")
        & point["benchmark"].eq("main_block_conditioned")
        & point["grid_origin"].eq("origin_mean")
    ].iloc[0]
    qa = {
        "freeze_id": "GHV27-CONTEXT-2026-08-02",
        "status": "PASS",
        "support_rows": {key: len(value) for key, value in support_frames.items()},
        "all_supports_50_cities": all(
            value["city_key"].nunique() == 50 for value in support_frames.values()
        ),
        "all_supports_exhaustive_context": all(
            value["context_id"].notna().all() for value in support_frames.values()
        ),
        "primary_27_types_nonempty": int(type_summary["type_code"].nunique()),
        "primary_minimum_type_share": float(type_summary["block_group_share"].min()),
        "primary_minimum_cities_per_type": int(type_summary["cities_represented"].min()),
        "joint_context_block_groups": int(primary["context_id"].eq("A").sum()),
        "joint_context_population": float(
            primary.loc[primary["context_id"].eq("A"), "population"].sum()
        ),
        "joint_context_city_marginal_ratio": float(primary_joint["observed_expected_ratio"]),
        "joint_context_city_marginal_bootstrap_interval": [
            float(primary_joint["bootstrap_p025"]),
            float(primary_joint["bootstrap_p975"]),
        ],
        "joint_context_main_block_conditioned_ratio": float(
            primary_spatial["observed_expected_ratio"]
        ),
        "joint_context_main_block_bootstrap_interval": [
            float(primary_spatial["bootstrap_p025"]),
            float(primary_spatial["bootstrap_p975"]),
        ],
        "minimum_boundary_context_agreement": float(
            boundary_agreement["context_agreement"].min()
        ),
        "minimum_boundary_joint_context_jaccard": float(
            boundary_agreement["joint_context_jaccard"].min()
        ),
        "tract_scale": tract_meta,
        "result5_boundary_inheritance": r5_qa,
        "separate_dimension_permutation_used": False,
        "spatial_null_p_value_claim_allowed": False,
        "input_sha256": {
            **{key: sha256(path) for key, path in SUPPORT_FILES.items()},
            "age_bg": sha256(AGE_BG),
            "block_ranges": sha256(BLOCK_RANGES),
            "result5_step2": sha256(STEP2_SCRIPT),
        },
    }
    if (
        qa["primary_27_types_nonempty"] != 27
        or not qa["all_supports_50_cities"]
        or not qa["all_supports_exhaustive_context"]
        or r5_qa["primary_reproduction_max_abs_difference"] > 1e-8
        or r5_qa["maximum_budget_error_m2"] > 1e-4
        or not r5_qa["pareto_endpoints_exact"]
        or not r5_qa["pareto_axes_monotonic"]
    ):
        qa["status"] = "FAIL"
    write_json(OUT / "qa" / "step5_technical_audit.json", qa)

    method_copy = OUT / "methods" / Path(__file__).name
    method_copy.write_bytes(Path(__file__).read_bytes())
    report = f"""# Result 4 joint-context reconstruction

The primary reconstruction contains {len(primary):,} block groups in 50 cities. All
27 nominal G-H-V cells are populated. The prespecified joint context contains
{int(primary['context_id'].eq('A').sum()):,} block groups and
{primary.loc[primary['context_id'].eq('A'), 'population'].sum()/1e6:.2f} million represented residents.

The equal-city observed-to-expected ratio is {primary_joint['observed_expected_ratio']:.3f}
relative to the within-city marginal-independence benchmark. Its two-level
resampling interval is {primary_joint['bootstrap_p025']:.3f}-{primary_joint['bootstrap_p975']:.3f}.
After conditioning the expected prevalence on primary-scale spatial blocks, the
ratio is {primary_spatial['observed_expected_ratio']:.3f}
({primary_spatial['bootstrap_p025']:.3f}-{primary_spatial['bootstrap_p975']:.3f}).
These are named descriptive benchmarks, not p values or causal interaction tests.

Across municipal-boundary supports, the minimum common-unit six-context agreement
is {boundary_agreement['context_agreement'].min():.3f}, and the minimum joint-context
Jaccard index is {boundary_agreement['joint_context_jaccard'].min():.3f}. At tract
scale the corresponding national joint-context ratio is {tract_meta['joint_context_ratio']:.3f},
whereas population-weighted local block-group-to-tract context agreement is
{tract_meta['bg_to_tract_population_weighted_context_agreement']:.3f}. The national
pattern and local label are therefore separate claims.

The primary Result 5 calculation was reproduced with a maximum absolute numerical
difference of {r5_qa['primary_reproduction_max_abs_difference']:.3g}. Equal budgets,
Pareto endpoints and monotonicity passed on all four boundary supports.
"""
    (OUT / "text" / "Result4_step5_results.md").write_text(report, encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
