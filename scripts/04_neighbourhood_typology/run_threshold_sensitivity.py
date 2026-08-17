from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(str(analysis_store_root()))
R4 = PROJECT / "Result" / "Result4"
LEGACY = R4 / "versions" / "joint_context_boundary_spatial_audit"
FRAME_PATH = LEGACY / "data" / "fully_contained_primary_GHV27.csv.gz"
THRESHOLD_PATH = LEGACY / "tables" / "04_tail_threshold_surface.csv"
BLOCK_PATH = (
    PROJECT
    / "Result"
    / "Result3"
    / "Result3_support_aligned"
    / "spatial_block_bootstrap"
    / "data"
    / "spatial_block_assignments.csv"
)
OUT = R4 / "versions" / "threshold_leave_one_unit_out"

SEED = 20260804
STAT_NAMES = ("n", "g", "h", "v", "gh", "gv", "hv", "ghv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def expected_count(stats: np.ndarray) -> np.ndarray:
    n, g, h, v, gh, gv, hv, ghv = np.moveaxis(stats, -1, 0)
    numerator = (n - 3.0) * g * h * v + g * hv + h * gv + v * gh - ghv
    denominator = (n - 1.0) ** 3
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=n >= 2)


def totals_metrics(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    totals = stats.sum(axis=0)
    n = totals[:, 0]
    observed = totals[:, 7] / n
    expected = expected_count(totals) / n
    return observed, expected


def block_metrics(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supported = stats[:, :, 0] >= 2
    supported_n = np.sum(np.where(supported, stats[:, :, 0], 0.0), axis=0)
    total_n = np.sum(stats[:, :, 0], axis=0)
    observed_count = np.sum(np.where(supported, stats[:, :, 7], 0.0), axis=0)
    expected = expected_count(stats)
    expected_count_sum = np.nansum(np.where(supported, expected, np.nan), axis=0)
    observed_rate = observed_count / supported_n
    expected_rate = expected_count_sum / supported_n
    coverage = supported_n / total_n
    return observed_rate, expected_rate, coverage


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[np.ndarray]]]:
    frame = pd.read_csv(FRAME_PATH, dtype={"GEOID_str": str}, low_memory=False)
    blocks = pd.read_csv(BLOCK_PATH, dtype={"GEOID_str": str}, low_memory=False)
    frame["GEOID_str"] = normalize(frame["GEOID_str"])
    blocks["GEOID_str"] = normalize(blocks["GEOID_str"])
    block_columns = ["city_key", "GEOID_str", *[f"main_block_origin{i}" for i in range(4)]]
    frame = frame.merge(blocks[block_columns], on=["city_key", "GEOID_str"], validate="one_to_one")
    thresholds = pd.read_csv(THRESHOLD_PATH).sort_values(["positive_G_H_quantile", "V_quantile"]).reset_index(drop=True)
    if len(frame) != 34566 or frame["city_key"].nunique() != 50 or len(thresholds) != 12:
        raise RuntimeError("Frozen Result 4 support mismatch")

    g = frame["G"].to_numpy(float)[:, None] > thresholds["G_cut_pp"].to_numpy(float)[None, :]
    h = frame["H"].to_numpy(float)[:, None] > thresholds["H_cut_C"].to_numpy(float)[None, :]
    v = frame["V"].to_numpy(float)[:, None] > thresholds["V_cut"].to_numpy(float)[None, :]
    flags = np.stack([g, h, v, g & h, g & v, h & v, g & h & v], axis=2).astype(float)

    prepared: dict[str, list[np.ndarray]] = {}
    for city, index in frame.groupby("city_key", sort=True).indices.items():
        index = np.asarray(index, dtype=int)
        city_flags = flags[index]
        origins = []
        for origin in range(4):
            labels = frame.iloc[index][f"main_block_origin{origin}"].to_numpy(np.int64)
            _, inverse = np.unique(labels, return_inverse=True)
            n_blocks = int(inverse.max()) + 1
            stats = np.zeros((n_blocks, len(thresholds), len(STAT_NAMES)), dtype=float)
            stats[:, :, 0] = np.bincount(inverse, minlength=n_blocks)[:, None]
            for stat_index in range(7):
                np.add.at(stats[:, :, stat_index + 1], inverse, city_flags[:, :, stat_index])
            origins.append(stats)
        prepared[str(city)] = origins
    return frame, thresholds, prepared


def point_estimates(thresholds: pd.DataFrame, prepared: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    city_observed, city_expected = [], []
    origin_rows = []
    for city in sorted(prepared):
        observed, expected = totals_metrics(prepared[city][0])
        city_observed.append(observed)
        city_expected.append(expected)
    city_observed = np.asarray(city_observed)
    city_expected = np.asarray(city_expected)
    city_obs = city_observed.mean(axis=0)
    city_exp = city_expected.mean(axis=0)

    for origin in range(4):
        observed, expected, coverage = [], [], []
        for city in sorted(prepared):
            obs, exp, cov = block_metrics(prepared[city][origin])
            observed.append(obs)
            expected.append(exp)
            coverage.append(cov)
        observed = np.asarray(observed).mean(axis=0)
        expected = np.asarray(expected).mean(axis=0)
        coverage = np.asarray(coverage).mean(axis=0)
        origin_rows.append((observed, expected, coverage))
    block_obs = np.asarray([row[0] for row in origin_rows]).mean(axis=0)
    block_exp = np.asarray([row[1] for row in origin_rows]).mean(axis=0)
    block_cov = np.asarray([row[2] for row in origin_rows]).mean(axis=0)

    out = thresholds.copy()
    out["city_loo_observed_prevalence"] = city_obs
    out["city_loo_expected_prevalence"] = city_exp
    out["city_loo_ratio"] = city_obs / city_exp
    out["main_block_loo_observed_prevalence"] = block_obs
    out["main_block_loo_expected_prevalence"] = block_exp
    out["main_block_loo_ratio"] = block_obs / block_exp
    out["main_block_supported_fraction"] = block_cov
    return out


def bootstrap(thresholds: pd.DataFrame, prepared: dict[str, list[np.ndarray]], draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cities = np.asarray(sorted(prepared), dtype=object)
    values = np.empty((draws, len(thresholds), 2), dtype=np.float32)
    for draw in range(draws):
        sampled_cities = rng.choice(cities, size=len(cities), replace=True)
        city_obs, city_exp, block_obs, block_exp = [], [], [], []
        for city in sampled_cities:
            stats = prepared[str(city)][int(rng.integers(0, 4))]
            sampled = stats[rng.integers(0, len(stats), size=len(stats))]
            obs, exp = totals_metrics(sampled)
            b_obs, b_exp, _ = block_metrics(sampled)
            city_obs.append(obs)
            city_exp.append(exp)
            block_obs.append(b_obs)
            block_exp.append(b_exp)
        values[draw, :, 0] = np.mean(city_obs, axis=0) / np.mean(city_exp, axis=0)
        values[draw, :, 1] = np.mean(block_obs, axis=0) / np.mean(block_exp, axis=0)

    rows = []
    for threshold_index, threshold in thresholds.iterrows():
        for benchmark_index, benchmark in enumerate(("city_loo_ratio", "main_block_loo_ratio")):
            distribution = values[:, threshold_index, benchmark_index]
            rows.append(
                {
                    "positive_G_H_quantile": threshold["positive_G_H_quantile"],
                    "V_quantile": threshold["V_quantile"],
                    "benchmark": benchmark,
                    "bootstrap_median": float(np.quantile(distribution, 0.5)),
                    "bootstrap_ci_low": float(np.quantile(distribution, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(distribution, 0.975)),
                    "probability_ratio_gt_1": float(np.mean(distribution > 1)),
                }
            )
    return pd.DataFrame(rows)


def brute_force_audit(prepared: dict[str, list[np.ndarray]]) -> float:
    errors = []
    for city in sorted(prepared)[:20]:
        stats = prepared[city][0]
        for block in stats[: min(10, len(stats))]:
            n = int(block[0, 0])
            if n < 2:
                continue
            # The totals identity is checked against explicit leave-one-unit-out
            # marginal products reconstructed from all 2^3 contingency cells.
            for threshold_index in (0, 4, 11):
                row = block[threshold_index]
                n, g, h, v, gh, gv, hv, ghv = row
                cells = np.zeros((2, 2, 2), dtype=float)
                cells[1, 1, 1] = ghv
                cells[1, 1, 0] = gh - ghv
                cells[1, 0, 1] = gv - ghv
                cells[0, 1, 1] = hv - ghv
                cells[1, 0, 0] = g - gh - gv + ghv
                cells[0, 1, 0] = h - gh - hv + ghv
                cells[0, 0, 1] = v - gv - hv + ghv
                cells[0, 0, 0] = n - cells.sum()
                explicit = 0.0
                for gi in (0, 1):
                    for hi in (0, 1):
                        for vi in (0, 1):
                            count = int(round(cells[gi, hi, vi]))
                            if count:
                                explicit += count * ((g - gi) / (n - 1)) * ((h - hi) / (n - 1)) * ((v - vi) / (n - 1))
                errors.append(abs(explicit - float(expected_count(row))))
    return max(errors) if errors else 0.0


def main() -> None:
    args = parse_args()
    for name in ("tables", "diagnostics", "qa", "methods", "text"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    frame, thresholds, prepared = prepare()
    point = point_estimates(thresholds, prepared)
    intervals = bootstrap(thresholds, prepared, args.bootstrap)
    for benchmark in ("city_loo_ratio", "main_block_loo_ratio"):
        subset = intervals.loc[intervals["benchmark"].eq(benchmark)].drop(columns="benchmark")
        rename = {column: f"{benchmark}_{column}" for column in ("bootstrap_median", "bootstrap_ci_low", "bootstrap_ci_high", "probability_ratio_gt_1")}
        point = point.merge(subset.rename(columns=rename), on=["positive_G_H_quantile", "V_quantile"], validate="one_to_one")
    formula_error = brute_force_audit(prepared)
    if formula_error > 1e-10:
        raise RuntimeError(f"LOO identity error: {formula_error}")

    point.to_csv(OUT / "tables" / "result4_threshold_leave_one_unit_out.csv", index=False)
    intervals.to_csv(OUT / "diagnostics" / "result4_threshold_loo_bootstrap_intervals.csv", index=False)
    report = {
        "status": "PASS",
        "analysis_units": len(frame),
        "cities": int(frame["city_key"].nunique()),
        "threshold_combinations": len(thresholds),
        "bootstrap_draws": args.bootstrap,
        "brute_force_max_abs_error": formula_error,
        "minimum_city_loo_ratio": float(point["city_loo_ratio"].min()),
        "minimum_main_block_loo_ratio": float(point["main_block_loo_ratio"].min()),
        "all_city_loo_intervals_above_one": bool((point["city_loo_ratio_bootstrap_ci_low"] > 1).all()),
        "all_main_block_loo_intervals_above_one": bool((point["main_block_loo_ratio_bootstrap_ci_low"] > 1).all()),
        "main_block_supported_fraction_range": [float(point["main_block_supported_fraction"].min()), float(point["main_block_supported_fraction"].max())],
        "input_hashes": {str(path): sha256(path) for path in (FRAME_PATH, THRESHOLD_PATH, BLOCK_PATH)},
        "interpretation": "Leave-one-unit-out ratios are descriptive enrichment relative to a marginal-product benchmark, not a spatial generative null or causal interaction test.",
    }
    (OUT / "qa" / "result4_threshold_loo_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "methods" / Path(__file__).name)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
