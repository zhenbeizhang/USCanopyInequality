from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import hashlib
import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests


PROJECT = analysis_store_root()
ACS_ROOT = PROJECT / "Top50\u57ce\u5e02_ACS2024_5yr"
SUPPORT_ROOT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_municipal_support_audit"
)
SUPPORT_FILE = SUPPORT_ROOT / "data" / "block_group_support_classification.csv.gz"
OUT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_block_population_boundary_audit"
)
RAW = OUT / "raw_census_pl2020"
DATA = OUT / "data"
TABLES = OUT / "tables"
QA = OUT / "qa"

PL_BASE = (
    "https://www2.census.gov/programs-surveys/decennial/2020/data/"
    "01-Redistricting_File--PL_94-171"
)


@dataclass(frozen=True)
class StateSpec:
    fips: str
    abbreviation: str
    folder: str


STATE_SPECS = {
    spec.fips: spec
    for spec in (
        StateSpec("04", "az", "Arizona"),
        StateSpec("06", "ca", "California"),
        StateSpec("08", "co", "Colorado"),
        StateSpec("11", "dc", "District_of_Columbia"),
        StateSpec("12", "fl", "Florida"),
        StateSpec("13", "ga", "Georgia"),
        StateSpec("17", "il", "Illinois"),
        StateSpec("18", "in", "Indiana"),
        StateSpec("21", "ky", "Kentucky"),
        StateSpec("24", "md", "Maryland"),
        StateSpec("25", "ma", "Massachusetts"),
        StateSpec("26", "mi", "Michigan"),
        StateSpec("27", "mn", "Minnesota"),
        StateSpec("29", "mo", "Missouri"),
        StateSpec("31", "ne", "Nebraska"),
        StateSpec("32", "nv", "Nevada"),
        StateSpec("35", "nm", "New_Mexico"),
        StateSpec("36", "ny", "New_York"),
        StateSpec("37", "nc", "North_Carolina"),
        StateSpec("39", "oh", "Ohio"),
        StateSpec("40", "ok", "Oklahoma"),
        StateSpec("41", "or", "Oregon"),
        StateSpec("42", "pa", "Pennsylvania"),
        StateSpec("47", "tn", "Tennessee"),
        StateSpec("48", "tx", "Texas"),
        StateSpec("51", "va", "Virginia"),
        StateSpec("53", "wa", "Washington"),
        StateSpec("55", "wi", "Wisconsin"),
    )
}

ADDITIONAL_PLACE_CODES: dict[tuple[str, str], tuple[str, ...]] = {}


def ensure_dirs() -> None:
    for path in (OUT, RAW, DATA, TABLES, QA):
        path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_geoid(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(12)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def city_key_from_directory(path: Path) -> str:
    stem = re.sub(r"^\d+_", "", path.name)
    stem = re.sub(r"_[A-Z]{2}$", "", stem)
    return normalize_text(stem)


def read_city_inventory(support: pd.DataFrame) -> pd.DataFrame:
    support_keys = {
        normalize_text(key): key
        for key in support["result_city_key"].drop_duplicates().astype(str)
    }
    rows: list[dict[str, object]] = []
    for directory in sorted(path for path in ACS_ROOT.iterdir() if path.is_dir()):
        city_file = directory / "gis" / "city_acs2024_5yr.gpkg"
        if not city_file.exists():
            continue
        city = gpd.read_file(city_file)
        if len(city) != 1:
            raise RuntimeError(f"Expected one city feature in {city_file}; found {len(city)}")
        record = city.iloc[0]
        normalized = city_key_from_directory(directory)
        if normalized not in support_keys:
            candidates = [
                key for norm, key in support_keys.items()
                if normalize_text(str(record["NAME"])) in norm
                and norm.endswith("city")
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Could not match {directory.name} to Result 2 city key; candidates={candidates}"
                )
            result_city_key = candidates[0]
        else:
            result_city_key = support_keys[normalized]
        rows.append(
            {
                "directory_rank": int(directory.name[:2]),
                "result_city_key": result_city_key,
                "city_display": support.loc[
                    support["result_city_key"].eq(result_city_key), "city_display"
                ].iloc[0],
                "state_fips": str(record["STATEFP"]).zfill(2),
                "place_fips": str(record["PLACEFP"]).zfill(5),
                "allocation_place_fips": ";".join(
                    (
                        str(record["PLACEFP"]).zfill(5),
                        *ADDITIONAL_PLACE_CODES.get(
                            (
                                str(record["STATEFP"]).zfill(2),
                                str(record["PLACEFP"]).zfill(5),
                            ),
                            (),
                        ),
                    )
                ),
                "city_geoid": str(record["GEOID"]),
                "city_name_acs": str(record["NAME"]),
                "city_namelsad_acs": str(record["NAMELSAD"]),
                "acs2024_city_population": float(record["total_population"]),
                "city_file": str(city_file),
            }
        )
    inventory = pd.DataFrame(rows).sort_values("directory_rank").reset_index(drop=True)
    if len(inventory) != 50 or inventory["result_city_key"].nunique() != 50:
        raise RuntimeError(
            f"Expected 50 unique city records; found {len(inventory)} rows and "
            f"{inventory['result_city_key'].nunique()} city keys"
        )
    missing = sorted(set(inventory["state_fips"]) - set(STATE_SPECS))
    if missing:
        raise RuntimeError(f"Missing PL download specifications for states {missing}")
    return inventory


def download_file(url: str, path: Path) -> dict[str, object]:
    if path.exists() and path.stat().st_size > 0:
        return {
            "url": url,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "downloaded": False,
        }
    temporary = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 9):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 60),
            ) as response:
                response.raise_for_status()
                resumable = offset > 0 and response.status_code == 206
                if offset > 0 and not resumable:
                    temporary.unlink()
                    offset = 0
                mode = "ab" if resumable else "wb"
                with temporary.open(mode) as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            last_error = None
            break
        except (requests.RequestException, OSError) as error:
            last_error = error
            print(
                f"  retry {attempt}/8 after {type(error).__name__}; "
                f"preserved {temporary.stat().st_size if temporary.exists() else 0:,} bytes",
                flush=True,
            )
            time.sleep(min(2**attempt, 30))
    if last_error is not None:
        raise RuntimeError(f"Could not download {url} after 8 attempts") from last_error
    temporary.replace(path)
    return {
        "url": url,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "downloaded": True,
    }


def download_states(inventory: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for number, state_fips in enumerate(sorted(inventory["state_fips"].unique()), start=1):
        spec = STATE_SPECS[state_fips]
        filename = f"{spec.abbreviation}2020.pl.zip"
        url = f"{PL_BASE}/{spec.folder}/{filename}"
        print(f"[download {number:02d}/{inventory['state_fips'].nunique():02d}] {state_fips} {filename}", flush=True)
        record = download_file(url, RAW / filename)
        record.update(asdict(spec))
        records.append(record)
    manifest = pd.DataFrame(records).sort_values("fips")
    manifest.to_csv(RAW / "download_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def member_name(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one member ending {suffix}; found {matches}")
    return matches[0]


def parse_state(
    state_fips: str,
    inventory: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = STATE_SPECS[state_fips]
    path = RAW / f"{spec.abbreviation}2020.pl.zip"
    state_cities = inventory.loc[inventory["state_fips"].eq(state_fips)].copy()
    target_keys = set(state_cities["result_city_key"])
    target_bgs = set(
        support.loc[support["result_city_key"].isin(target_keys), "GEOID_str"]
    )
    target_places = {
        code
        for value in state_cities["allocation_place_fips"]
        for code in str(value).split(";")
    }
    log_to_geo: dict[str, tuple[str, str]] = {}
    city_log: dict[str, str] = {}

    with zipfile.ZipFile(path) as archive:
        geo_name = member_name(archive, f"{spec.abbreviation}geo2020.pl")
        with archive.open(geo_name) as handle:
            for raw_line in handle:
                row = raw_line.decode("utf-8", errors="replace").rstrip("\r\n").split("|")
                if len(row) < 32:
                    continue
                summary_level = row[2]
                if summary_level == "750":
                    block_geoid = row[9]
                    if len(block_geoid) >= 12 and block_geoid[:12] in target_bgs:
                        # PLACE is field 30 (zero-based index 29) in the
                        # 2020 PL state geographic header. Index 17 is COUSUB.
                        log_to_geo[row[7]] = (block_geoid[:12], row[29])
                elif summary_level == "160" and row[29] in target_places:
                    city_log[row[29]] = row[7]

        segment_name = member_name(archive, f"{spec.abbreviation}000012020.pl")
        block_rows: list[tuple[str, str, int]] = []
        city_population: dict[str, int] = {}
        with archive.open(segment_name) as handle:
            for raw_line in handle:
                row = raw_line.decode("utf-8", errors="replace").rstrip("\r\n").split("|")
                if len(row) < 6:
                    continue
                logrecno = row[4]
                if logrecno in log_to_geo:
                    bg_geoid, place_fips = log_to_geo[logrecno]
                    block_rows.append((bg_geoid, place_fips, int(row[5] or 0)))
                for place_fips, target_log in city_log.items():
                    if logrecno == target_log:
                        city_population[place_fips] = int(row[5] or 0)

    blocks = pd.DataFrame(
        block_rows,
        columns=["GEOID_str", "block_place_fips", "pl2020_population"],
    )
    if blocks.empty:
        raise RuntimeError(f"No relevant 2020 Census blocks were parsed for state {state_fips}")
    grouped = (
        blocks.groupby(["GEOID_str", "block_place_fips"], dropna=False, as_index=False)
        ["pl2020_population"]
        .sum()
    )
    bg_total = (
        grouped.groupby("GEOID_str", as_index=False)["pl2020_population"]
        .sum()
        .rename(columns={"pl2020_population": "pl2020_bg_population"})
    )
    allocations: list[pd.DataFrame] = []
    check_rows: list[dict[str, object]] = []
    for _, city in state_cities.iterrows():
        allocation_places = str(city["allocation_place_fips"]).split(";")
        inside = (
            grouped.loc[grouped["block_place_fips"].isin(allocation_places)]
            [["GEOID_str", "pl2020_population"]]
            .groupby("GEOID_str", as_index=False)["pl2020_population"]
            .sum()
            .rename(columns={"pl2020_population": "pl2020_city_block_population"})
        )
        city_alloc = bg_total.merge(inside, on="GEOID_str", how="left")
        city_alloc["pl2020_city_block_population"] = city_alloc[
            "pl2020_city_block_population"
        ].fillna(0).astype(int)
        city_alloc["result_city_key"] = city["result_city_key"]
        city_alloc["state_fips"] = state_fips
        city_alloc["place_fips"] = city["place_fips"]
        allocations.append(city_alloc)
        sum_inside = int(city_alloc["pl2020_city_block_population"].sum())
        published_values = [city_population.get(code) for code in allocation_places]
        published = (
            sum(int(value) for value in published_values if value is not None)
            if all(value is not None for value in published_values)
            else None
        )
        check_rows.append(
            {
                "result_city_key": city["result_city_key"],
                "city_display": city["city_display"],
                "state_fips": state_fips,
                "place_fips": city["place_fips"],
                "allocation_place_fips": city["allocation_place_fips"],
                "pl2020_population_from_relevant_blocks": sum_inside,
                "pl2020_published_place_population": published,
                "recovery_ratio": (
                    sum_inside / published
                    if published is not None and published > 0
                    else np.nan
                ),
                "n_relevant_block_groups": int(len(city_alloc)),
            }
        )
    return pd.concat(allocations, ignore_index=True), pd.DataFrame(check_rows)


def build_allocations(inventory: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    allocation_parts: list[pd.DataFrame] = []
    check_parts: list[pd.DataFrame] = []
    states = sorted(inventory["state_fips"].unique())
    for number, state_fips in enumerate(states, start=1):
        print(f"[parse {number:02d}/{len(states):02d}] state {state_fips}", flush=True)
        allocation, checks = parse_state(state_fips, inventory, support)
        allocation_parts.append(allocation)
        check_parts.append(checks)
    allocations = pd.concat(allocation_parts, ignore_index=True)
    checks = pd.concat(check_parts, ignore_index=True).sort_values("city_display")
    checks.to_csv(TABLES / "01_pl2020_place_population_recovery.csv", index=False, encoding="utf-8-sig")

    merged = support.merge(
        allocations,
        on=["result_city_key", "GEOID_str"],
        how="left",
        validate="one_to_one",
    )
    if merged["pl2020_bg_population"].isna().any():
        missing = merged.loc[merged["pl2020_bg_population"].isna(), ["result_city_key", "GEOID_str"]]
        missing.to_csv(QA / "missing_pl2020_block_groups.csv", index=False, encoding="utf-8-sig")
    merged["pl2020_bg_population"] = merged["pl2020_bg_population"].fillna(0)
    merged["pl2020_city_block_population"] = merged[
        "pl2020_city_block_population"
    ].fillna(0)
    block_fraction = np.divide(
        merged["pl2020_city_block_population"].to_numpy(float),
        merged["pl2020_bg_population"].to_numpy(float),
        out=np.full(len(merged), np.nan),
        where=merged["pl2020_bg_population"].to_numpy(float) > 0,
    )
    block_fraction = np.clip(block_fraction, 0.0, 1.0)
    merged["block_population_fraction_raw"] = block_fraction
    merged["allocation_method"] = "2020_block_population"
    zero_population = ~np.isfinite(block_fraction)
    merged.loc[zero_population, "allocation_method"] = "area_fraction_zero_2020_population"
    merged.loc[zero_population, "block_population_fraction_raw"] = merged.loc[
        zero_population, "city_intersection_fraction"
    ].clip(0, 1)
    complete = merged["fully_contained"].astype(bool)
    merged.loc[complete, "block_population_fraction"] = 1.0
    merged.loc[~complete, "block_population_fraction"] = merged.loc[
        ~complete, "block_population_fraction_raw"
    ]
    merged.loc[complete, "allocation_method"] = "fully_contained_exact"
    merged["acs2024_population_allocated_uncalibrated"] = (
        merged["population"] * merged["block_population_fraction"]
    )

    # Do not rake block-group allocations to the separate ACS place estimate.
    # For Louisville in particular, the ACS place aggregate and the analysed
    # balance polygon have different legal-geography semantics. A city-wide
    # multiplier would hide that mismatch while leaving within-city slopes
    # unchanged. The uncalibrated block allocation is therefore the estimand.
    merged["population_raking_factor"] = 1.0
    merged["population_block_allocated"] = merged[
        "acs2024_population_allocated_uncalibrated"
    ]
    intersection = merged["city_intersection_fraction"].clip(0, 1)
    merged["estimated_city_land_area_m2"] = merged["land_area_m2"] * intersection
    estimated_canopy = np.divide(
        100.0 * merged["canopy_area_m2"].to_numpy(float),
        merged["estimated_city_land_area_m2"].to_numpy(float),
        out=np.full(len(merged), np.nan),
        where=merged["estimated_city_land_area_m2"].to_numpy(float) > 0,
    )
    merged["canopy_pct_block_allocated_frame"] = estimated_canopy
    merged["canopy_denominator_method"] = "canopy_area_over_ALAND_times_intersection"
    merged.loc[complete, "canopy_pct_block_allocated_frame"] = merged.loc[
        complete, "canopy_pct_analysis"
    ]
    merged.loc[complete, "estimated_city_land_area_m2"] = merged.loc[
        complete, "land_area_m2"
    ]
    merged.loc[complete, "canopy_denominator_method"] = "fully_contained_primary_land_denominator"
    invalid_partial_canopy = (
        ~complete
        & ~merged["canopy_pct_block_allocated_frame"].between(0, 100)
    )
    merged.loc[invalid_partial_canopy, "canopy_pct_block_allocated_frame"] = merged.loc[
        invalid_partial_canopy, "mapped_portion_canopy_pct"
    ]
    merged.loc[invalid_partial_canopy, "estimated_city_land_area_m2"] = merged.loc[
        invalid_partial_canopy, "raster_valid_area_m2"
    ]
    merged.loc[invalid_partial_canopy, "canopy_denominator_method"] = (
        "mapped_valid_area_fallback_for_nonphysical_land_estimate"
    )
    merged["block_allocated_analysis_unit"] = (
        merged["population_block_allocated"].gt(0)
        & merged["canopy_pct_block_allocated_frame"].notna()
    )

    calibration = (
        merged.groupby(["result_city_key", "city_display"], as_index=False)
        .agg(
            n_intersecting_block_groups=("GEOID_str", "size"),
            n_positive_allocated_block_groups=("block_allocated_analysis_unit", "sum"),
            original_full_bg_population_sum=("population", "sum"),
            allocated_population_before_calibration=("acs2024_population_allocated_uncalibrated", "sum"),
            block_allocated_population=("population_block_allocated", "sum"),
            median_area_fraction=("city_intersection_fraction", "median"),
            median_block_population_fraction=("block_population_fraction", "median"),
        )
        .merge(
            inventory[["result_city_key", "acs2024_city_population"]],
            on="result_city_key",
            how="left",
            validate="one_to_one",
        )
        .sort_values("city_display")
    )
    calibration["acs_place_to_block_allocated_ratio"] = (
        calibration["acs2024_city_population"]
        / calibration["block_allocated_population"].replace(0, np.nan)
    )
    support_comparison_path = TABLES / "02_city_population_support_comparison.csv"
    calibration.to_csv(support_comparison_path, index=False, encoding="utf-8-sig")
    legacy_path = TABLES / "02_city_population_calibration.csv"
    if legacy_path.exists():
        legacy_path.unlink()

    partial = merged.loc[~complete].copy()
    partial_summary = (
        partial.groupby(["result_city_key", "city_display"], as_index=False)
        .agg(
            n_partial=("GEOID_str", "size"),
            n_zero_2020_population=("allocation_method", lambda x: int((x == "area_fraction_zero_2020_population").sum())),
            mean_area_fraction=("city_intersection_fraction", "mean"),
            mean_block_population_fraction=("block_population_fraction", "mean"),
            allocated_population=("population_block_allocated", "sum"),
        )
        .sort_values("city_display")
    )
    partial_summary.to_csv(TABLES / "03_partial_block_group_allocation_summary.csv", index=False, encoding="utf-8-sig")

    merged.to_csv(
        DATA / "block_population_boundary_frame.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    inventory.to_csv(DATA / "city_inventory.csv", index=False, encoding="utf-8-sig")
    return merged


def audit_frame(frame: pd.DataFrame, inventory: pd.DataFrame) -> dict[str, object]:
    checks = pd.read_csv(TABLES / "01_pl2020_place_population_recovery.csv")
    calibration = pd.read_csv(TABLES / "02_city_population_support_comparison.csv")
    partial = frame.loc[~frame["fully_contained"].astype(bool)].copy()
    qa = {
        "status": "PASS",
        "n_cities": int(frame["result_city_key"].nunique()),
        "n_intersecting_block_groups": int(len(frame)),
        "n_fully_contained_block_groups": int(frame["fully_contained"].astype(bool).sum()),
        "n_positive_allocated_analysis_units": int(frame["block_allocated_analysis_unit"].sum()),
        "states_downloaded": int(inventory["state_fips"].nunique()),
        "minimum_place_population_recovery_ratio": float(checks["recovery_ratio"].min()),
        "median_place_population_recovery_ratio": float(checks["recovery_ratio"].median()),
        "population_raking_applied": False,
        "acs_place_to_block_allocated_ratio_range": [
            float(calibration["acs_place_to_block_allocated_ratio"].min()),
            float(calibration["acs_place_to_block_allocated_ratio"].max()),
        ],
        "partial_units": int(len(partial)),
        "partial_units_using_area_fallback": int(
            partial["allocation_method"].eq("area_fraction_zero_2020_population").sum()
        ),
        "partial_units_with_zero_allocated_population": int(
            partial["population_block_allocated"].le(0).sum()
        ),
        "canopy_values_outside_0_100": int(
            (~frame["canopy_pct_block_allocated_frame"].between(0, 100)).sum()
        ),
        "partial_canopy_denominator_fallback_units": int(
            frame["canopy_denominator_method"].eq(
                "mapped_valid_area_fallback_for_nonphysical_land_estimate"
            ).sum()
        ),
        "important_interpretation": (
            "The 2020 block counts allocate 2024 ACS population weights within boundary-crossing "
            "block groups. Block-group social compositions remain observed at block-group support; "
            "the frame is a boundary sensitivity, not reconstructed city-only demographics."
        ),
        "source": (
            "U.S. Census Bureau 2020 Census Redistricting Data (P.L. 94-171), "
            "state legacy summary files, block total population P1_001N and PLACEFP geography."
        ),
        "generated_unix_time": time.time(),
    }
    recovery = pd.to_numeric(checks["recovery_ratio"], errors="coerce")
    if (
        qa["n_cities"] != 50
        or qa["n_intersecting_block_groups"] != 38_618
        or recovery.isna().any()
        or qa["minimum_place_population_recovery_ratio"] < 0.995
        or qa["canopy_values_outside_0_100"] > 0
    ):
        qa["status"] = "FAIL"
    (QA / "block_population_boundary_frame_audit.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "download", "build", "audit"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    support = pd.read_csv(SUPPORT_FILE, dtype={"GEOID_str": str}, low_memory=False)
    support["GEOID_str"] = normalize_geoid(support["GEOID_str"])
    inventory = read_city_inventory(support)
    if args.stage in ("all", "download"):
        download_states(inventory)
        if args.stage == "download":
            return 0
    if args.stage in ("all", "build"):
        frame = build_allocations(inventory, support)
        qa = audit_frame(frame, inventory)
        print(json.dumps(qa, indent=2, ensure_ascii=False))
        return 0
    frame = pd.read_csv(
        DATA / "block_population_boundary_frame.csv.gz",
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    qa = audit_frame(frame, inventory)
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
