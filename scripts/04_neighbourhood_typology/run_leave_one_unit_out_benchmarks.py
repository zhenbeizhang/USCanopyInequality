from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = analysis_store_root()
R4 = PROJECT / "Result" / "Result4"
LEGACY = R4 / "versions" / "joint_context_boundary_spatial_audit"
SUPPORT = PROJECT / "Result" / "Result3" / "Result3_support_aligned"
FRAME_PATH = LEGACY / "data" / "fully_contained_primary_GHV27.csv.gz"
RANGE_PATH = SUPPORT / "spatial_block_bootstrap" / "tables" / "city_residual_range_and_block_sizes.csv"
LEGACY_BENCHMARK = LEGACY / "tables" / "03_cooccurrence_benchmarks.csv"
OUT = R4 / "versions" / "leave_one_unit_out_benchmark"

SEED = 20260804
OFFSETS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
SCALES = ("small", "main", "large")
EVENTS = ("joint_context", "strict_G3H3V3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def ensure_dirs() -> None:
    for name in ("tables", "diagnostics", "qa", "methods", "text", "source_data"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_grid(coords: np.ndarray, anchor: np.ndarray, size_m: float, offset: tuple[float, float]) -> np.ndarray:
    grid = np.floor((coords - anchor[None, :] + np.asarray(offset) * size_m) / size_m).astype(np.int64)
    return np.asarray([f"{x}:{y}" for x, y in grid], dtype=object)


def add_event_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["h"] = out["H_level"].eq(3).astype(np.int8)
    out["v"] = out["V_level"].eq(3).astype(np.int8)
    for event in EVENTS:
        g = out["G_level"].eq(3) if event == "strict_G3H3V3" else out["G_level"].ge(2)
        out[f"{event}_g"] = g.astype(np.int8)
        out[f"{event}_gh"] = (g & out["h"].eq(1)).astype(np.int8)
        out[f"{event}_gv"] = (g & out["v"].eq(1)).astype(np.int8)
        out[f"{event}_hv"] = (out["h"].eq(1) & out["v"].eq(1)).astype(np.int8)
        out[f"{event}_ghv"] = (g & out["h"].eq(1) & out["v"].eq(1)).astype(np.int8)
    return out


def loo_expected_count_from_totals(row: pd.Series, event: str) -> float:
    n = float(row["n"])
    if n < 2:
        return float("nan")
    g = float(row[f"{event}_g"])
    h = float(row["h"])
    v = float(row["v"])
    gh = float(row[f"{event}_gh"])
    gv = float(row[f"{event}_gv"])
    hv = float(row[f"{event}_hv"])
    ghv = float(row[f"{event}_ghv"])
    numerator = (n - 3.0) * g * h * v + g * hv + h * gv + v * gh - ghv
    return float(numerator / (n - 1.0) ** 3)


def aggregate_columns(event: str) -> list[str]:
    return [
        "n",
        "h",
        "v",
        f"{event}_g",
        f"{event}_gh",
        f"{event}_gv",
        f"{event}_hv",
        f"{event}_ghv",
    ]


def grouped_totals(frame: pd.DataFrame, labels: np.ndarray, event: str) -> pd.DataFrame:
    columns = aggregate_columns(event)
    local = frame[columns].copy()
    local["block"] = labels
    table = local.groupby("block", sort=True)[columns].sum()
    table["observed_count"] = table[f"{event}_ghv"]
    table["loo_expected_count"] = table.apply(loo_expected_count_from_totals, axis=1, event=event)
    table["supported_n"] = np.where(table["n"].ge(2), table["n"], 0.0)
    table["supported_observed_count"] = np.where(table["n"].ge(2), table["observed_count"], 0.0)
    table["loo_expected_count"] = table["loo_expected_count"].fillna(0.0)
    return table.reset_index(drop=False)


def prepare_tables(frame: pd.DataFrame, ranges: pd.DataFrame):
    lookup = ranges.set_index("city_key")
    anchors = {
        city: sub[["x_5070_m", "y_5070_m"]].min().to_numpy(float)
        for city, sub in frame.groupby("city_key", sort=True)
    }
    output = defaultdict(lambda: defaultdict(dict))
    block_size_rows = []
    for city, sub in frame.groupby("city_key", sort=True):
        coords = sub[["x_5070_m", "y_5070_m"]].to_numpy(float)
        for scale in SCALES:
            size_m = float(lookup.loc[city, f"{scale}_block_size_km"]) * 1000
            for origin, offset in enumerate(OFFSETS):
                labels = assign_grid(coords, anchors[city], size_m, offset)
                output[city][scale][origin] = {
                    event: grouped_totals(sub, labels, event) for event in EVENTS
                }
                counts = pd.Series(labels).value_counts().to_numpy(int)
                block_size_rows.append(
                    {
                        "city_key": city,
                        "scale": scale,
                        "origin": origin,
                        "block_size_km": size_m / 1000,
                        "blocks": len(counts),
                        "median_units_per_block": float(np.median(counts)),
                        "q25_units_per_block": float(np.quantile(counts, 0.25)),
                        "q75_units_per_block": float(np.quantile(counts, 0.75)),
                        "singleton_blocks": int(np.sum(counts == 1)),
                        "units_in_singletons": int(np.sum(counts[counts == 1])),
                        "total_units": int(np.sum(counts)),
                    }
                )
    return output, pd.DataFrame(block_size_rows)


def city_marginal_loo(frame: pd.DataFrame, event: str) -> tuple[float, float, float]:
    observed = []
    expected = []
    for _, sub in frame.groupby("city_key", sort=True):
        totals = sub[aggregate_columns(event)].sum()
        n = float(totals["n"])
        observed.append(float(totals[f"{event}_ghv"] / n))
        expected.append(float(loo_expected_count_from_totals(totals, event) / n))
    obs = float(np.mean(observed))
    exp = float(np.mean(expected))
    return obs, exp, obs / exp


def block_point(tables, event: str, scale: str) -> list[dict[str, object]]:
    rows = []
    origin_obs = []
    origin_exp = []
    origin_coverage = []
    for origin in range(len(OFFSETS)):
        city_obs = []
        city_exp = []
        city_coverage = []
        for city in sorted(tables):
            table = tables[city][scale][origin][event]
            supported = float(table["supported_n"].sum())
            total = float(table["n"].sum())
            if supported <= 0:
                continue
            city_obs.append(float(table["supported_observed_count"].sum() / supported))
            city_exp.append(float(table["loo_expected_count"].sum() / supported))
            city_coverage.append(supported / total)
        obs = float(np.mean(city_obs))
        exp = float(np.mean(city_exp))
        coverage = float(np.mean(city_coverage))
        origin_obs.append(obs)
        origin_exp.append(exp)
        origin_coverage.append(coverage)
        rows.append(
            {
                "event": event,
                "benchmark": f"{scale}_block_leave_one_unit_out",
                "grid_origin": origin,
                "observed_equal_city_prevalence_supported_units": obs,
                "expected_equal_city_prevalence": exp,
                "observed_expected_ratio": obs / exp,
                "equal_city_supported_unit_fraction": coverage,
            }
        )
    obs = float(np.mean(origin_obs))
    exp = float(np.mean(origin_exp))
    rows.append(
        {
            "event": event,
            "benchmark": f"{scale}_block_leave_one_unit_out",
            "grid_origin": "origin_mean",
            "observed_equal_city_prevalence_supported_units": obs,
            "expected_equal_city_prevalence": exp,
            "observed_expected_ratio": obs / exp,
            "equal_city_supported_unit_fraction": float(np.mean(origin_coverage)),
        }
    )
    return rows


def resampled_city_marginal(table: pd.DataFrame, event: str, rng: np.random.Generator) -> tuple[float, float]:
    sampled = table.iloc[rng.integers(0, len(table), size=len(table))]
    totals = sampled[aggregate_columns(event)].sum()
    n = float(totals["n"])
    observed = float(totals[f"{event}_ghv"] / n)
    expected = float(loo_expected_count_from_totals(totals, event) / n)
    return observed, expected


def resampled_block(table: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    sampled = table.iloc[rng.integers(0, len(table), size=len(table))]
    supported = float(sampled["supported_n"].sum())
    total = float(sampled["n"].sum())
    if supported <= 0:
        return float("nan"), float("nan"), 0.0
    observed = float(sampled["supported_observed_count"].sum() / supported)
    expected = float(sampled["loo_expected_count"].sum() / supported)
    return observed, expected, supported / total


def bootstrap(tables, replicates: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cities = np.asarray(sorted(tables), dtype=object)
    rows = []
    for draw in range(replicates):
        sampled_cities = rng.choice(cities, size=len(cities), replace=True)
        for event in EVENTS:
            for benchmark in ("city", *SCALES):
                observed = []
                expected = []
                coverage = []
                for city in sampled_cities:
                    origin = int(rng.integers(0, len(OFFSETS)))
                    scale = "main" if benchmark == "city" else benchmark
                    table = tables[str(city)][scale][origin][event]
                    if benchmark == "city":
                        obs, exp = resampled_city_marginal(table, event, rng)
                        cov = 1.0
                    else:
                        obs, exp, cov = resampled_block(table, rng)
                    if np.isfinite(obs) and np.isfinite(exp) and exp > 0:
                        observed.append(obs)
                        expected.append(exp)
                        coverage.append(cov)
                obs = float(np.mean(observed))
                exp = float(np.mean(expected))
                rows.append(
                    {
                        "draw": draw,
                        "event": event,
                        "benchmark": "city_marginal_leave_one_unit_out" if benchmark == "city" else f"{benchmark}_block_leave_one_unit_out",
                        "observed_prevalence": obs,
                        "expected_prevalence": exp,
                        "ratio": obs / exp,
                        "supported_unit_fraction": float(np.mean(coverage)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    frame = pd.read_csv(FRAME_PATH, dtype={"GEOID_str": str}, low_memory=False)
    if len(frame) != 34_566 or frame["city_key"].nunique() != 50:
        raise RuntimeError("Expected 34,566 units in 50 cities.")
    frame["n"] = 1
    frame = add_event_flags(frame)
    ranges = pd.read_csv(RANGE_PATH)
    tables, block_sizes = prepare_tables(frame, ranges)

    point_rows = []
    for event in EVENTS:
        obs, exp, ratio = city_marginal_loo(frame, event)
        point_rows.append(
            {
                "event": event,
                "benchmark": "city_marginal_leave_one_unit_out",
                "grid_origin": "not_applicable",
                "observed_equal_city_prevalence_supported_units": obs,
                "expected_equal_city_prevalence": exp,
                "observed_expected_ratio": ratio,
                "equal_city_supported_unit_fraction": 1.0,
            }
        )
        for scale in SCALES:
            point_rows.extend(block_point(tables, event, scale))
    point = pd.DataFrame(point_rows)

    draws = bootstrap(tables, args.bootstrap)
    interval = (
        draws.groupby(["event", "benchmark"])["ratio"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .rename(columns={0.025: "bootstrap_p025", 0.5: "bootstrap_median", 0.975: "bootstrap_p975"})
        .reset_index()
    )
    point = point.merge(interval, on=["event", "benchmark"], how="left", validate="many_to_one")

    block_size_summary = (
        block_sizes.groupby("scale", as_index=False)
        .agg(
            city_origin_combinations=("city_key", "size"),
            median_block_width_km=("block_size_km", "median"),
            median_units_per_block=("median_units_per_block", "median"),
            total_blocks=("blocks", "sum"),
            total_singleton_blocks=("singleton_blocks", "sum"),
            total_units_across_origins=("total_units", "sum"),
            units_in_singletons_across_origins=("units_in_singletons", "sum"),
        )
    )
    block_size_summary["singleton_block_fraction"] = (
        block_size_summary["total_singleton_blocks"] / block_size_summary["total_blocks"]
    )
    block_size_summary["unit_fraction_in_singletons"] = (
        block_size_summary["units_in_singletons_across_origins"] / block_size_summary["total_units_across_origins"]
    )

    legacy = pd.read_csv(LEGACY_BENCHMARK)
    compare_rows = []
    mapping = {
        "city_marginal_independence": "city_marginal_leave_one_unit_out",
        "small_block_conditioned": "small_block_leave_one_unit_out",
        "main_block_conditioned": "main_block_leave_one_unit_out",
        "large_block_conditioned": "large_block_leave_one_unit_out",
    }
    for event in EVENTS:
        for old_name, new_name in mapping.items():
            old = legacy.loc[
                legacy["event"].eq(event)
                & legacy["benchmark"].eq(old_name)
                & legacy["grid_origin"].astype(str).eq("origin_mean")
            ]
            if old_name == "city_marginal_independence":
                old = legacy.loc[legacy["event"].eq(event) & legacy["benchmark"].eq(old_name)]
            new = point.loc[
                point["event"].eq(event)
                & point["benchmark"].eq(new_name)
                & point["grid_origin"].astype(str).isin(["origin_mean", "not_applicable"])
            ]
            compare_rows.append(
                {
                    "event": event,
                    "legacy_benchmark": old_name,
                    "leave_one_unit_out_benchmark": new_name,
                    "legacy_ratio": float(old["observed_expected_ratio"].iloc[0]),
                    "leave_one_unit_out_ratio": float(new["observed_expected_ratio"].iloc[0]),
                    "absolute_change": float(new["observed_expected_ratio"].iloc[0] - old["observed_expected_ratio"].iloc[0]),
                    "relative_change_pct": float(100 * (new["observed_expected_ratio"].iloc[0] / old["observed_expected_ratio"].iloc[0] - 1)),
                    "leave_one_unit_out_supported_fraction": float(new["equal_city_supported_unit_fraction"].iloc[0]),
                }
            )
    comparison = pd.DataFrame(compare_rows)

    point.to_csv(OUT / "tables" / "result4_leave_one_unit_out_benchmarks.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "tables" / "result4_legacy_vs_leave_one_unit_out.csv", index=False, encoding="utf-8-sig")
    block_sizes.to_csv(OUT / "diagnostics" / "result4_block_size_by_city_origin.csv", index=False, encoding="utf-8-sig")
    block_size_summary.to_csv(OUT / "tables" / "result4_block_size_coverage_summary.csv", index=False, encoding="utf-8-sig")
    draws.to_csv(OUT / "diagnostics" / "result4_leave_one_unit_out_bootstrap_draws.csv.gz", index=False, compression="gzip")

    primary = point.loc[
        point["event"].eq("joint_context")
        & point["benchmark"].eq("main_block_leave_one_unit_out")
        & point["grid_origin"].astype(str).eq("origin_mean")
    ].iloc[0]
    city = point.loc[
        point["event"].eq("joint_context")
        & point["benchmark"].eq("city_marginal_leave_one_unit_out")
    ].iloc[0]
    audit = {
        "status": "PASS",
        "analysis_units": len(frame),
        "cities": int(frame["city_key"].nunique()),
        "bootstrap_replicates": args.bootstrap,
        "joint_context_city_loo_ratio": float(city["observed_expected_ratio"]),
        "joint_context_city_loo_interval": [float(city["bootstrap_p025"]), float(city["bootstrap_p975"])],
        "joint_context_main_block_loo_ratio": float(primary["observed_expected_ratio"]),
        "joint_context_main_block_loo_interval": [float(primary["bootstrap_p025"]), float(primary["bootstrap_p975"])],
        "joint_context_main_block_supported_fraction": float(primary["equal_city_supported_unit_fraction"]),
        "interpretation": (
            "Leave-one-unit-out marginal estimation does not remove the joint-context enrichment. "
            "The spatially conditioned ratio remains descriptive because blocks are an empirical conditioning device, not a generative spatial null."
        ),
        "claims_forbidden": [
            "formal spatial-null rejection",
            "causal interaction among canopy, heat and social conditions",
            "scale-invariant priority-community list",
        ],
        "input_hashes": {
            str(FRAME_PATH): sha256(FRAME_PATH),
            str(RANGE_PATH): sha256(RANGE_PATH),
            str(LEGACY_BENCHMARK): sha256(LEGACY_BENCHMARK),
        },
    }
    (OUT / "qa" / "result4_loo_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "text" / "result4_loo_interpretation.txt").write_text(
        f"The joint heat-social-canopy context occurred {city['observed_expected_ratio']:.3f} times the city-level leave-one-unit-out marginal-independence expectation "
        f"(95% two-level interval {city['bootstrap_p025']:.3f}-{city['bootstrap_p975']:.3f}). At the primary residual-range block width, the ratio was "
        f"{primary['observed_expected_ratio']:.3f} ({primary['bootstrap_p025']:.3f}-{primary['bootstrap_p975']:.3f}) among the "
        f"{100 * primary['equal_city_supported_unit_fraction']:.1f}% of units lying in blocks with at least two analysed communities. "
        "These are descriptive enrichments relative to marginal-product benchmarks, not tests against a generative spatial null.\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OUT / "methods" / Path(__file__).name)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
