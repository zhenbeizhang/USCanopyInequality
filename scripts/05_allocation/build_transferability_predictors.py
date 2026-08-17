from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from uscanopy.paths import analysis_store_root


MONTH_DAYS = {"JUN": 30, "JUL": 31, "AUG": 31}
REGION_BY_STATE = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    # Midwest
    "IN": "Midwest", "IL": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    # South
    "DE": "South", "DC": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "WV": "South", "AL": "South",
    "KY": "South", "MS": "South", "TN": "South", "AR": "South", "LA": "South",
    "OK": "South", "TX": "South",
    # West
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "UT": "West", "WY": "West", "AK": "West", "CA": "West",
    "HI": "West", "OR": "West", "WA": "West",
}

ANALYSIS_ROOT = analysis_store_root()
RESULT_ROOT = ANALYSIS_ROOT / "Result"
DEFAULT_ACS_CITY_LIST = (
    ANALYSIS_ROOT
    / "Top50城市_ACS2024_5yr"
    / "_summary"
    / "top50_city_list_matched.csv"
)
DEFAULT_COORDINATES = (
    RESULT_ROOT
    / "Result4_5_neighbourhood_typology"
    / "Fig4_candidate_NatureCities_3_20260729"
    / "source_data"
    / "Fig4d_city_boundary_compactness_audit.csv"
)
DEFAULT_OUTPUT = (
    RESULT_ROOT
    / "Result5"
    / "versions"
    / "step03_cross_city_transferability"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--acs-city-list", type=Path, default=DEFAULT_ACS_CITY_LIST)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    x = values[valid]
    w = weights[valid]
    mean = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mean) ** 2, weights=w)))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    x = values[valid]
    w = weights[valid]
    order = np.argsort(x, kind="stable")
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= w.sum()
    return float(np.interp(probability, cumulative, x))


def power_climatology(longitude: float, latitude: float) -> tuple[dict, str]:
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "AG",
        "longitude": f"{longitude:.6f}",
        "latitude": f"{latitude:.6f}",
        "start": "1991",
        "end": "2020",
        "format": "JSON",
    }
    url = "https://power.larc.nasa.gov/api/temporal/climatology/point?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "USCanopyInequality/Step3"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
            return payload, url
        except Exception as error:  # pragma: no cover - network retry
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"NASA POWER request failed: {last_error}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    for directory in ("data", "raw_climate", "qa", "manifests", "methods"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    frame_path = (
        args.result_root / "Result3" / "Result3_support_aligned" /
        "data" / "support_aligned_model_frame_34566.csv"
    )
    frame = pd.read_csv(frame_path, low_memory=False)
    if len(frame) != 34_566 or frame["city_key"].nunique() != 50:
        raise RuntimeError("Step 3 fixed frame mismatch")
    frame["land_area_m2"] = pd.to_numeric(frame["land_area_m2"], errors="coerce")
    required = [
        "worldcover_built_pct", "distance_to_worldcover_water_m",
        "terrain_relief_1km_m", "land_area_m2",
    ]
    if frame[required].isna().any().any() or (frame["land_area_m2"] <= 0).any():
        raise RuntimeError("Missing or invalid urban-environment predictors")

    rows = []
    for city_key, city in frame.groupby("city_key", sort=True):
        land = city["land_area_m2"].to_numpy(float)
        built = city["worldcover_built_pct"].to_numpy(float)
        log_water = np.log1p(city["distance_to_worldcover_water_m"].to_numpy(float))
        log_relief = np.log1p(city["terrain_relief_1km_m"].to_numpy(float))
        rows.append(
            {
                "city_key": city_key,
                "city_display": city["city_display"].iloc[0],
                "n_block_groups": len(city),
                "analysis_land_km2": float(land.sum() / 1e6),
                "built_fraction_area_weighted_mean_pct": weighted_mean(built, land),
                "built_fraction_area_weighted_sd_pct": weighted_sd(built, land),
                "log_distance_to_water_weighted_median": weighted_quantile(log_water, land, 0.5),
                "log_terrain_relief_weighted_median": weighted_quantile(log_relief, land, 0.5),
            }
        )
    predictors = pd.DataFrame(rows)

    coordinates = pd.read_csv(args.coordinates)[
        ["city_key", "city_display", "longitude", "latitude"]
    ].drop_duplicates("city_key")
    predictors = predictors.merge(
        coordinates, on=["city_key", "city_display"], validate="one_to_one"
    )

    city_list = pd.read_csv(args.acs_city_list)
    city_list["city_display"] = (
        city_list["city_name"].str.replace(" city", "", regex=False)
        .str.replace("Louisville/Jefferson County metro government (balance)", "Louisville Jefferson County", regex=False)
        .str.replace("Nashville-Davidson metropolitan government (balance)", "Nashville-Davidson", regex=False)
        .str.replace("Indianapolis (balance)", "Indianapolis", regex=False)
    )
    predictors = predictors.merge(
        city_list[["city_display", "state_abbr"]], on="city_display", validate="one_to_one"
    )
    predictors["census_region"] = predictors["state_abbr"].map(REGION_BY_STATE)
    if predictors["census_region"].isna().any():
        raise RuntimeError("Unmapped Census region")

    climate_rows = []
    for row in predictors.sort_values("city_key").itertuples(index=False):
        payload, url = power_climatology(row.longitude, row.latitude)
        raw_path = output / "raw_climate" / f"{row.city_key}.json"
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        parameter = payload["properties"]["parameter"]
        t2m = parameter["T2M"]
        precipitation = parameter["PRECTOTCORR"]
        days = sum(MONTH_DAYS.values())
        jja_temperature = sum(t2m[month] * MONTH_DAYS[month] for month in MONTH_DAYS) / days
        jja_precipitation = sum(precipitation[month] * MONTH_DAYS[month] for month in MONTH_DAYS)
        climate_rows.append(
            {
                "city_key": row.city_key,
                "jja_air_temperature_C": float(jja_temperature),
                "jja_precipitation_mm": float(jja_precipitation),
                "nasa_power_request_url": url,
                "nasa_power_raw_file": raw_path.name,
                "nasa_power_raw_sha256": sha256(raw_path),
            }
        )
        print(f"downloaded {row.city_display}", flush=True)
    predictors = predictors.merge(pd.DataFrame(climate_rows), on="city_key", validate="one_to_one")

    frozen = [
        "jja_air_temperature_C", "jja_precipitation_mm",
        "built_fraction_area_weighted_mean_pct", "built_fraction_area_weighted_sd_pct",
        "log_distance_to_water_weighted_median", "log_terrain_relief_weighted_median",
    ]
    if len(predictors) != 50 or predictors["city_key"].nunique() != 50:
        raise RuntimeError("Predictor ledger does not cover 50 cities")
    if predictors[frozen].isna().any().any() or not np.isfinite(predictors[frozen].to_numpy(float)).all():
        raise RuntimeError("Non-finite frozen predictor")
    if (predictors[frozen].nunique() < 3).any():
        raise RuntimeError("Insufficient predictor variation")

    predictors.to_csv(output / "data" / "city_external_predictors.csv", index=False, encoding="utf-8-sig")
    summary = predictors[frozen].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    summary.to_csv(output / "qa" / "predictor_distribution_audit.csv", encoding="utf-8-sig")
    region_counts = predictors.groupby("census_region", as_index=False).agg(cities=("city_key", "size"))
    region_counts.to_csv(output / "qa" / "census_region_counts.csv", index=False, encoding="utf-8-sig")

    sources = [frame_path, args.coordinates, args.acs_city_list]
    pd.DataFrame(
        [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in sources]
    ).to_csv(output / "manifests" / "predictor_input_lineage.csv", index=False, encoding="utf-8-sig")
    audit = {
        "status": "PASS",
        "cities": 50,
        "analysis_units": 34_566,
        "predictors": frozen,
        "predictors_complete": True,
        "census_region_counts": dict(zip(region_counts["census_region"], region_counts["cities"], strict=True)),
        "outcome_loaded_or_inspected": False,
        "nasa_power_period": "1991-2020 climatology",
        "interpretive_boundary": "External city conditions only; no G/H/V, allocation or outcome-derived predictor.",
    }
    (output / "qa" / "predictor_ledger_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
