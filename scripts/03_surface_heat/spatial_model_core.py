from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
import statsmodels.api as sm
from pyproj import Transformer
from pygam import LinearGAM, l, te
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy import ndimage, optimize, stats
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(str(analysis_store_root()))
RESULT_ROOT = PROJECT_ROOT / "Result"
R3_ROOT = RESULT_ROOT / "Result3" / "Result3_temperature_canopy_50city"
R4_ROOT = RESULT_ROOT / "Result4" / "versions" / "counterfactual_density_strata"
LOWRES_ROOT = PROJECT_ROOT / "inputs" / "low_resolution_products"
OUT_ROOT = RESULT_ROOT / "Result3" / "Result3_spatial_residual_audit"

R3_MODEL = R3_ROOT / "data" / "model" / "result3_model_input_block_groups.csv"
R3_GPKG = R3_ROOT / "data" / "block_group" / "result3_temperature_canopy_block_group.gpkg"
R3_CITY_SLOPES = R3_ROOT / "data" / "model" / "result3_city_canopy_temperature_slopes.csv"
R4_HVCG = R4_ROOT / "module_05_hvcg_priority" / "result4_module_05_hvcg_priority_block_groups_valid.csv"
R4_SLOPE_AUDIT = R4_ROOT / "module_04_counterfactual_cooling" / "result4_module_04_cooling_slope_audit.csv"
LOWRES_MANIFEST = LOWRES_ROOT / "metadata" / "download_manifest.csv"

FROZEN_CSV = OUT_ROOT / "data" / "frozen_analysis_sample_38447.csv"
FROZEN_GPKG = OUT_ROOT / "data" / "frozen_analysis_sample_38447.gpkg"
LAND_COVARIATES = OUT_ROOT / "data" / "covariates" / "worldcover_built_water_covariates.csv"
TOPO_COVARIATES = OUT_ROOT / "data" / "covariates" / "copernicus_dem_covariates.csv"
MODEL_FRAME = OUT_ROOT / "data" / "model_analysis_frame.csv"

SEED = 20260716
N_SPATIAL_FOLDS = 5
K_NEIGHBOURS = 8
MORAN_PERMUTATIONS = 199
GAM_N_SPLINES = 6
GAM_LAMBDA = 3.0

MODEL1_COVARIATES = ["log_population_density"]
MODEL2_COVARIATES = [
    "log_population_density",
    "z_worldcover_built_pct",
    "z_log_distance_to_water_m",
    "z_elevation_m",
    "z_log_terrain_relief_1km_m",
]
GAM_LINEAR_COLUMNS = ["canopy_pct_10pp", *MODEL2_COVARIATES]


@dataclass
class FitBundle:
    model_id: str
    national: dict[str, object]
    city: pd.DataFrame
    predictions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and spatially re-estimate the canopy-LST association.")
    parser.add_argument(
        "--stage",
        choices=("all", "freeze", "covariates", "models"),
        default="all",
        help="Run one resumable stage or the full workflow.",
    )
    parser.add_argument("--overwrite-covariates", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--moran-permutations", type=int, default=MORAN_PERMUTATIONS)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in (
        OUT_ROOT / "data" / "covariates",
        OUT_ROOT / "tables",
        OUT_ROOT / "diagnostics",
        OUT_ROOT / "qa",
        OUT_ROOT / "scripts",
        OUT_ROOT / "text",
    ):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def canonical_city(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower()).replace("city", "")


def normalize_geoid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2:
        return float("nan")
    x = values[valid]
    w = weights[valid]
    mean = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mean) ** 2, weights=w)))


def add_within_city_z(df: pd.DataFrame, source: str, target: str) -> None:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("city_key", sort=False).groups.items():
        values = pd.to_numeric(df.loc[idx, source], errors="coerce").to_numpy(float)
        weights = pd.to_numeric(df.loc[idx, "population_model_weight"], errors="coerce").fillna(0).to_numpy(float)
        mean = weighted_mean(values, weights)
        sd = weighted_sd(values, weights)
        if not np.isfinite(sd) or sd <= 1e-12:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (values - mean) / sd
    df[target] = out


def freeze_analysis_sample() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    model = pd.read_csv(R3_MODEL, low_memory=False, dtype={"GEOID_str": str})
    model["GEOID_str"] = normalize_geoid(model["GEOID_str"])
    if len(model) != 38447 or model["city_key"].nunique() != 50:
        raise RuntimeError(f"Expected 38,447 block groups in 50 cities, found {len(model):,} in {model['city_key'].nunique()} cities.")
    if model[["city_key", "GEOID_str"]].duplicated().any():
        raise RuntimeError("Frozen Result 3 input contains duplicate city-GEOID records.")

    geometry = gpd.read_file(R3_GPKG)
    geometry["GEOID_str"] = normalize_geoid(geometry["GEOID_str"])
    geometry = geometry[["city_key", "GEOID_str", "geometry"]].drop_duplicates(["city_key", "GEOID_str"])
    frozen = geometry.merge(model, on=["city_key", "GEOID_str"], how="inner", validate="one_to_one")
    if len(frozen) != len(model):
        raise RuntimeError(f"Only {len(frozen):,} of {len(model):,} frozen records have geometry.")
    if frozen.crs is None:
        raise RuntimeError("Frozen block-group geometry has no CRS.")
    frozen = frozen.to_crs("EPSG:5070")
    points = frozen.geometry.representative_point()
    frozen["x_5070_m"] = points.x
    frozen["y_5070_m"] = points.y
    frozen["analysis_record_id"] = frozen["city_key"].astype(str) + "|" + frozen["GEOID_str"]
    frozen["mask_source"] = "predict_mosaic_clip2 city-clipped binary U-Net mask"
    frozen["mask_postprocessing_for_analysis"] = "none"

    canopy_raw = pd.to_numeric(frozen["canopy_pct"], errors="coerce")
    canopy_analysis = pd.to_numeric(frozen["canopy_pct_analysis"], errors="coerce")
    canopy_diff = (canopy_raw - canopy_analysis).abs()

    hvcg = pd.read_csv(R4_HVCG, low_memory=False, dtype={"GEOID_str": str})
    hvcg["GEOID_str"] = normalize_geoid(hvcg["GEOID_str"])
    poc_expected = 100.0 - pd.to_numeric(hvcg["white_alone_pct"], errors="coerce")
    poc_diff = (pd.to_numeric(hvcg["people_of_color_pct"], errors="coerce") - poc_expected).abs()

    slopes = pd.read_csv(R4_SLOPE_AUDIT)
    slope_count = slopes.loc[slopes["slope_type"].eq("population_weighted"), "city_key"].nunique()
    if slope_count != 50:
        raise RuntimeError(f"Legacy HVCG slope audit contains {slope_count} cities, expected 50.")

    frozen.drop(columns="geometry").to_csv(FROZEN_CSV, index=False, encoding="utf-8-sig")
    frozen.to_file(FROZEN_GPKG, layer="frozen_sample", driver="GPKG")
    slopes.to_csv(OUT_ROOT / "tables" / "legacy_hvcg_city_slope_ledger.csv", index=False, encoding="utf-8-sig")

    freeze_qa = {
        "analysis_sample": {
            "source": str(R3_MODEL),
            "rows": int(len(frozen)),
            "cities": int(frozen["city_key"].nunique()),
            "unique_city_geoid_records": int(frozen[["city_key", "GEOID_str"]].drop_duplicates().shape[0]),
            "geometry_crs": str(frozen.crs),
        },
        "canopy_mask": {
            "analytical_raster_root": str(PROJECT_ROOT / "predict_mosaic_clip2"),
            "interpretation": "Raw thresholded U-Net binary masks clipped to city boundaries; no morphological filtering entered block-group statistics.",
            "canopy_pct_vs_canopy_pct_analysis_different_rows": int((canopy_diff > 1e-9).sum()),
            "maximum_absolute_difference_percentage_points": float(canopy_diff.max()),
        },
        "people_of_color": {
            "unique_formula": "100 - White alone (%)",
            "code_location": str(RESULT_ROOT / "Result4" / "scripts" / "prepare_result4_hvcg.py"),
            "rows_checked": int(poc_diff.notna().sum()),
            "maximum_absolute_formula_difference": float(poc_diff.max()),
        },
        "hvcg_tiers": {
            "decision": "Mutually exclusive, precedence-ordered tiers are used in this audit.",
            "tier_1": "intensity percentile >= 0.90 and burden percentile >= 0.90",
            "tier_2a": "intensity percentile >= 0.90 and burden percentile < 0.90",
            "tier_2b": "intensity percentile < 0.90 and burden percentile >= 0.90",
            "tier_3": "not tiers 1-2; canopy-gap percentile >= 0.90 and relievable-heat percentile < 0.50",
            "lower": "all remaining records",
        },
        "legacy_city_slopes": {
            "cities": int(slope_count),
            "rule": "Use unadjusted city population-weighted slope only when negative and P <= 0.10; otherwise use pooled population-weighted FE+density coefficient.",
            "warning": "This rule differs from the current Result 3 national reference model and is explicitly separated from the three re-estimated model tiers.",
        },
    }
    write_json(OUT_ROOT / "qa" / "analysis_freeze.json", freeze_qa)
    return model, frozen


def load_frozen() -> gpd.GeoDataFrame:
    if not FROZEN_GPKG.exists():
        freeze_analysis_sample()
    frozen = gpd.read_file(FROZEN_GPKG, layer="frozen_sample")
    frozen["GEOID_str"] = normalize_geoid(frozen["GEOID_str"])
    return frozen


def worldcover_paths_by_city() -> dict[str, Path]:
    manifest = pd.read_csv(LOWRES_MANIFEST)
    manifest = manifest.loc[manifest["product_key"].eq("worldcover")].copy()
    if manifest["city"].nunique() != 50:
        raise RuntimeError("WorldCover manifest is not complete for all 50 cities.")
    return {canonical_city(row.city): Path(row.output_file) for row in manifest.itertuples(index=False)}


def extract_worldcover_covariates(frozen: gpd.GeoDataFrame, overwrite: bool = False) -> pd.DataFrame:
    if LAND_COVARIATES.exists() and not overwrite:
        return pd.read_csv(LAND_COVARIATES, dtype={"GEOID_str": str})
    paths = worldcover_paths_by_city()
    records: list[pd.DataFrame] = []
    for number, (city_key, sub) in enumerate(frozen.groupby("city_key", sort=True), start=1):
        key = canonical_city(city_key)
        if key not in paths:
            raise RuntimeError(f"Cannot match WorldCover raster for {city_key}.")
        raster_path = paths[key]
        print(f"[WorldCover {number:02d}/50] {city_key}", flush=True)
        with rasterio.open(raster_path) as src:
            work = sub.to_crs(src.crs).copy()
            if (~work.geometry.is_valid).any():
                work.geometry = work.geometry.buffer(0)
            labels = rasterize(
                ((geom, i + 1) for i, geom in enumerate(work.geometry) if geom is not None and not geom.is_empty),
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                all_touched=False,
                dtype="int32",
            )
            values = src.read(1)
            valid = values != src.nodata
            flat_labels = labels.ravel()
            n = len(work)
            valid_counts = np.bincount(flat_labels[valid.ravel()], minlength=n + 1)[1 : n + 1]
            built_counts = np.bincount(flat_labels[(valid & (values == 50)).ravel()], minlength=n + 1)[1 : n + 1]
            water_counts = np.bincount(flat_labels[(valid & (values == 80)).ravel()], minlength=n + 1)[1 : n + 1]
            bare_counts = np.bincount(flat_labels[(valid & (values == 60)).ravel()], minlength=n + 1)[1 : n + 1]

            water_mask = valid & (values == 80)
            if water_mask.any():
                distance = ndimage.distance_transform_edt(~water_mask, sampling=(abs(src.transform.e), src.transform.a))
                points = work.geometry.representative_point()
                rows, cols = rasterio.transform.rowcol(src.transform, points.x.to_numpy(), points.y.to_numpy())
                rows = np.clip(np.asarray(rows, dtype=int), 0, src.height - 1)
                cols = np.clip(np.asarray(cols, dtype=int), 0, src.width - 1)
                water_distance = distance[rows, cols]
            else:
                water_distance = np.full(n, np.nan)

        denom = np.where(valid_counts > 0, valid_counts, np.nan)
        records.append(
            pd.DataFrame(
                {
                    "city_key": city_key,
                    "GEOID_str": work["GEOID_str"].astype(str).to_numpy(),
                    "worldcover_valid_pixels": valid_counts,
                    "worldcover_built_pct": built_counts / denom * 100.0,
                    "worldcover_water_pct": water_counts / denom * 100.0,
                    "worldcover_bare_pct": bare_counts / denom * 100.0,
                    "distance_to_worldcover_water_m": water_distance,
                    "worldcover_raster": str(raster_path),
                }
            )
        )
    out = pd.concat(records, ignore_index=True)
    out.to_csv(LAND_COVARIATES, index=False, encoding="utf-8-sig")
    return out


def point_sample_dem(href: str, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(href) as src:
        # Use the COG overview at approximately 300 m. This preserves the broad
        # topographic gradients relevant to a neighbourhood LST model while
        # avoiding tens of thousands of remote 30 m range requests.
        factor = 10
        out_height = max(1, int(math.ceil(src.height / factor)))
        out_width = max(1, int(math.ceil(src.width / factor)))
        array = src.read(1, out_shape=(out_height, out_width), masked=True, resampling=Resampling.average)
        transform = src.transform * src.transform.scale(src.width / out_width, src.height / out_height)
        rows, cols = rasterio.transform.rowcol(transform, lon, lat)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        sampled = np.full((len(lon), 2), np.nan, dtype=float)
        for index, (row, col) in enumerate(zip(rows, cols)):
            if row < 0 or row >= out_height or col < 0 or col >= out_width:
                continue
            center = array[row, col]
            if np.ma.is_masked(center):
                continue
            sampled[index, 0] = float(center)
            r0, r1 = max(0, row - 1), min(out_height, row + 2)
            c0, c1 = max(0, col - 1), min(out_width, col + 2)
            local = np.asarray(array[r0:r1, c0:c1].filled(np.nan), dtype=float)
            sampled[index, 1] = float(np.nanmax(local) - np.nanmin(local))
        nodata = src.nodata
    if nodata is not None:
        sampled[np.isclose(sampled, nodata)] = np.nan
    sampled[(sampled < -500) | (sampled > 9000)] = np.nan
    elevation = sampled[:, 0]
    relief = sampled[:, 1]
    return elevation, relief


def copdem_tile_id(lon_floor: int, lat_floor: int) -> str:
    latitude = f"N{lat_floor:02d}" if lat_floor >= 0 else f"S{abs(lat_floor):02d}"
    longitude = f"E{lon_floor:03d}" if lon_floor >= 0 else f"W{abs(lon_floor):03d}"
    return f"Copernicus_DSM_COG_10_{latitude}_00_{longitude}_00_DEM"


def fill_missing_dem_primary_tiles(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "requested_dem_item_id" not in frame.columns:
        frame["requested_dem_item_id"] = [
            copdem_tile_id(math.floor(float(x)), math.floor(float(y))) for x, y in zip(frame["lon"], frame["lat"])
        ]
    if "dem_fill_method" not in frame.columns:
        frame["dem_fill_method"] = np.where(frame["elevation_m"].notna(), "primary_tile", "pending")
    missing = frame["elevation_m"].isna()
    if not missing.any():
        return frame
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    collection = catalog.get_collection("cop-dem-glo-30")
    groups = list(frame.loc[missing].groupby("requested_dem_item_id", sort=True).groups.items())
    for number, (item_id, indices) in enumerate(groups, start=1):
        index = np.asarray(list(indices), dtype=int)
        item = collection.get_item(item_id)
        if item is None:
            continue
        item = pc.sign(item)
        try:
            elevation, relief = point_sample_dem(
                item.assets["data"].href,
                frame.loc[index, "lon"].to_numpy(float),
                frame.loc[index, "lat"].to_numpy(float),
            )
        except Exception as exc:
            print(f"[Copernicus DEM refill warning] {item_id}: {exc}", file=sys.stderr, flush=True)
            continue
        valid = np.isfinite(elevation)
        valid_index = index[valid]
        frame.loc[valid_index, "elevation_m"] = elevation[valid]
        frame.loc[valid_index, "terrain_relief_1km_m"] = relief[valid]
        frame.loc[valid_index, "dem_item_id"] = item_id
        frame.loc[valid_index, "dem_fill_method"] = "primary_tile_refill"
        if number % 10 == 0 or number == len(groups):
            remaining = int(frame["elevation_m"].isna().sum())
            print(f"[Copernicus DEM refill] {number}/{len(groups)} tiles; {remaining:,} points remain", flush=True)
    return frame


def repair_missing_dem_boundary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "dem_fill_method" not in frame.columns:
        frame["dem_fill_method"] = "primary_tile"
    missing = frame.index[frame["elevation_m"].isna()].tolist()
    if not missing:
        return frame
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    collection = catalog.get_collection("cop-dem-glo-30")
    for position, index in enumerate(missing, start=1):
        lon = float(frame.at[index, "lon"])
        lat = float(frame.at[index, "lat"])
        lon_floor = math.floor(lon)
        lat_floor = math.floor(lat)
        lon_fraction = lon - lon_floor
        lat_fraction = lat - lat_floor
        lon_order = [lon_floor + 1, lon_floor, lon_floor - 1] if lon_fraction > 0.5 else [lon_floor - 1, lon_floor, lon_floor + 1]
        lat_order = [lat_floor - 1, lat_floor, lat_floor + 1] if lat_fraction < 0.5 else [lat_floor + 1, lat_floor, lat_floor - 1]
        primary = str(frame.at[index, "requested_dem_item_id"])
        candidates = [copdem_tile_id(x, y) for y in lat_order for x in lon_order]
        candidates = [item_id for item_id in candidates if item_id != primary]
        for item_id in candidates:
            item = collection.get_item(item_id)
            if item is None:
                continue
            item = pc.sign(item)
            try:
                elevation, relief = point_sample_dem(item.assets["data"].href, np.asarray([lon]), np.asarray([lat]))
            except Exception:
                continue
            if np.isfinite(elevation[0]):
                frame.at[index, "elevation_m"] = elevation[0]
                frame.at[index, "terrain_relief_1km_m"] = relief[0]
                frame.at[index, "dem_item_id"] = item_id
                frame.at[index, "dem_fill_method"] = "adjacent_tile_boundary"
                break
        print(f"[Copernicus DEM boundary repair] {position}/{len(missing)}", flush=True)

    unresolved = frame["elevation_m"].isna()
    if unresolved.any():
        city_elevation = frame.groupby("city_key")["elevation_m"].transform("median")
        city_relief = frame.groupby("city_key")["terrain_relief_1km_m"].transform("median")
        frame.loc[unresolved, "elevation_m"] = city_elevation.loc[unresolved]
        frame.loc[unresolved, "terrain_relief_1km_m"] = city_relief.loc[unresolved]
        frame.loc[unresolved, "dem_fill_method"] = "city_median_after_adjacent_tile_failure"
    return frame


def extract_dem_covariates(frozen: gpd.GeoDataFrame, overwrite: bool = False) -> pd.DataFrame:
    if TOPO_COVARIATES.exists() and not overwrite:
        cached = pd.read_csv(TOPO_COVARIATES, dtype={"GEOID_str": str})
        original_missing = int(cached["elevation_m"].isna().sum())
        had_fill_method = "dem_fill_method" in cached.columns
        cached = fill_missing_dem_primary_tiles(cached)
        repaired = repair_missing_dem_boundary_rows(cached)
        if int(repaired["elevation_m"].isna().sum()) < original_missing or not had_fill_method:
            repaired.to_csv(TOPO_COVARIATES, index=False, encoding="utf-8-sig")
        return repaired
    work = frozen[["city_key", "GEOID_str", "geometry"]].to_crs("EPSG:4326").copy()
    points = work.geometry.representative_point()
    work["lon"] = points.x
    work["lat"] = points.y
    work["elevation_m"] = np.nan
    work["terrain_relief_1km_m"] = np.nan
    work["dem_item_id"] = ""

    work["requested_dem_item_id"] = [copdem_tile_id(math.floor(x), math.floor(y)) for x, y in zip(work["lon"], work["lat"])]
    tile_groups = list(work.groupby("requested_dem_item_id", sort=True).groups.items())
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    collection = catalog.get_collection("cop-dem-glo-30")
    print(f"[Copernicus DEM] {len(tile_groups)} occupied one-degree tiles", flush=True)
    for number, (item_id, candidate_index) in enumerate(tile_groups, start=1):
        candidate = np.asarray(list(candidate_index), dtype=int)
        item = collection.get_item(item_id)
        if item is None:
            print(f"[Copernicus DEM warning] missing STAC item {item_id}", file=sys.stderr, flush=True)
            continue
        item = pc.sign(item)
        href = item.assets["data"].href
        try:
            elevation, relief = point_sample_dem(
                href,
                work.loc[candidate, "lon"].to_numpy(float),
                work.loc[candidate, "lat"].to_numpy(float),
            )
        except Exception as exc:
            print(f"[Copernicus DEM warning] {item.id}: {exc}", file=sys.stderr, flush=True)
            continue
        work.loc[candidate, "elevation_m"] = elevation
        work.loc[candidate, "terrain_relief_1km_m"] = relief
        work.loc[candidate, "dem_item_id"] = item.id
        if number % 10 == 0 or number == len(tile_groups):
            completed = int(work["elevation_m"].notna().sum())
            print(f"[Copernicus DEM] {number}/{len(tile_groups)} tiles; {completed:,}/{len(work):,} elevations", flush=True)

    out = repair_missing_dem_boundary_rows(pd.DataFrame(work.drop(columns="geometry")))
    out.to_csv(TOPO_COVARIATES, index=False, encoding="utf-8-sig")
    return out


def build_model_frame(overwrite_covariates: bool = False) -> pd.DataFrame:
    frozen = load_frozen()
    land = extract_worldcover_covariates(frozen, overwrite=overwrite_covariates)
    topo = extract_dem_covariates(frozen, overwrite=overwrite_covariates)
    land["GEOID_str"] = normalize_geoid(land["GEOID_str"])
    topo["GEOID_str"] = normalize_geoid(topo["GEOID_str"])
    frame = pd.DataFrame(frozen.drop(columns="geometry"))
    frame = frame.merge(land, on=["city_key", "GEOID_str"], how="left", validate="one_to_one")
    frame = frame.merge(topo, on=["city_key", "GEOID_str"], how="left", validate="one_to_one")
    frame["log_distance_to_water_m"] = np.log1p(pd.to_numeric(frame["distance_to_worldcover_water_m"], errors="coerce"))
    frame["log_terrain_relief_1km_m"] = np.log1p(pd.to_numeric(frame["terrain_relief_1km_m"], errors="coerce").clip(lower=0))
    add_within_city_z(frame, "worldcover_built_pct", "z_worldcover_built_pct")
    add_within_city_z(frame, "worldcover_water_pct", "z_worldcover_water_pct")
    add_within_city_z(frame, "log_distance_to_water_m", "z_log_distance_to_water_m")
    add_within_city_z(frame, "elevation_m", "z_elevation_m")
    add_within_city_z(frame, "log_terrain_relief_1km_m", "z_log_terrain_relief_1km_m")

    for _, idx in frame.groupby("city_key", sort=False).groups.items():
        x = frame.loc[idx, "x_5070_m"].to_numpy(float)
        y = frame.loc[idx, "y_5070_m"].to_numpy(float)
        x_range = max(float(np.nanmax(x) - np.nanmin(x)), 1.0)
        y_range = max(float(np.nanmax(y) - np.nanmin(y)), 1.0)
        frame.loc[idx, "x_city_0_1"] = (x - np.nanmin(x)) / x_range
        frame.loc[idx, "y_city_0_1"] = (y - np.nanmin(y)) / y_range

    complete = frame[["heat_anomaly_C", "canopy_pct_10pp", "population_model_weight", *MODEL2_COVARIATES]].notna().all(axis=1)
    complete &= pd.to_numeric(frame["population_model_weight"], errors="coerce").gt(0)
    frame["complete_spatial_model"] = complete
    frame.to_csv(MODEL_FRAME, index=False, encoding="utf-8-sig")
    qa = {
        "rows": int(len(frame)),
        "cities": int(frame["city_key"].nunique()),
        "complete_spatial_model_rows": int(complete.sum()),
        "missing_by_covariate": frame[["worldcover_built_pct", "distance_to_worldcover_water_m", "elevation_m", "terrain_relief_1km_m"]].isna().sum().astype(int).to_dict(),
        "worldcover_definition": {
            "built_up": "ESA WorldCover 2020 class 50; used as a built-surface proxy, not a continuous imperviousness percentage.",
            "water": "ESA WorldCover 2020 class 80; centroid distance is measured to the nearest mapped water pixel within the city-clipped raster.",
        },
        "terrain_definition": {
            "source": "Copernicus DEM GLO-30 via Microsoft Planetary Computer",
            "elevation": "Approximately 300 m COG-overview elevation at the block-group representative point.",
            "relief": "Elevation range in the surrounding 3 x 3 overview cells (approximately 0.9 km window).",
        },
    }
    write_json(OUT_ROOT / "qa" / "covariate_extraction.json", qa)
    return frame


def load_model_frame() -> pd.DataFrame:
    if not MODEL_FRAME.exists():
        return build_model_frame()
    frame = pd.read_csv(MODEL_FRAME, low_memory=False, dtype={"GEOID_str": str})
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    return frame


def design_matrix(df: pd.DataFrame, covariates: list[str], city_levels: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if city_levels is None:
        city_levels = sorted(df["city_key"].astype(str).unique())
    city = pd.Categorical(df["city_key"].astype(str), categories=city_levels)
    dummies = pd.get_dummies(city, prefix="city", drop_first=True, dtype=float)
    numeric = df[["canopy_pct_10pp", *covariates]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    x = pd.concat([pd.Series(1.0, index=numeric.index, name="const"), numeric, dummies.reset_index(drop=True)], axis=1)
    return x, city_levels


def fit_national_fe(df: pd.DataFrame, model_id: str, covariates: list[str]) -> tuple[dict[str, object], pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    needed = ["heat_anomaly_C", "canopy_pct_10pp", "population_model_weight", "city_key", *covariates]
    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work.loc[pd.to_numeric(work["population_model_weight"], errors="coerce").gt(0)].copy()
    x, levels = design_matrix(work, covariates)
    y = pd.to_numeric(work["heat_anomaly_C"], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(work["population_model_weight"], errors="coerce").to_numpy(float)
    fitted = sm.WLS(y, x, weights=weights).fit()
    clustered = fitted.get_robustcov_results(cov_type="cluster", groups=work["city_key"].astype(str).to_numpy(), use_correction=True)
    names = list(x.columns)
    canopy_index = names.index("canopy_pct_10pp")
    beta = float(clustered.params[canopy_index])
    se = float(clustered.bse[canopy_index])
    cluster_df = max(work["city_key"].nunique() - 1, 1)
    critical = float(stats.t.ppf(0.975, df=cluster_df))
    result = {
        "model_id": model_id,
        "model_family": "population-weighted city fixed effects",
        "outcome": "within-city summer mean LST anomaly (C)",
        "canopy_unit": "10 percentage points",
        "covariates": " + ".join(covariates),
        "coefficient_C_per_10pp_canopy": beta,
        "standard_error_city_clustered": se,
        "ci_low": beta - critical * se,
        "ci_high": beta + critical * se,
        "p_value": float(2 * stats.t.sf(abs(beta / se), df=cluster_df)),
        "n_block_groups": int(len(work)),
        "n_cities": int(work["city_key"].nunique()),
        "r_squared": float(fitted.rsquared),
        "inference": "city-clustered 95% confidence interval",
    }
    pred = pd.DataFrame(
        {
            "city_key": work["city_key"].astype(str).to_numpy(),
            "row_index": work.index.to_numpy(),
            "observed": y,
            "predicted": fitted.predict(x),
        }
    )
    pred["residual"] = pred["observed"] - pred["predicted"]
    pred["model_id"] = model_id
    return result, pred, fitted


def fit_city_wls(df: pd.DataFrame, model_id: str, covariates: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for city_key, sub in df.groupby("city_key", sort=True):
        needed = ["heat_anomaly_C", "canopy_pct_10pp", "population_model_weight", *covariates]
        work = sub[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
        work = work.loc[pd.to_numeric(work["population_model_weight"], errors="coerce").gt(0)].copy()
        x = sm.add_constant(work[["canopy_pct_10pp", *covariates]].astype(float), has_constant="add")
        y = work["heat_anomaly_C"].astype(float)
        weights = work["population_model_weight"].astype(float)
        fit = sm.WLS(y, x, weights=weights).fit(cov_type="HC3")
        beta = float(fit.params["canopy_pct_10pp"])
        se = float(fit.bse["canopy_pct_10pp"])
        records.append(
            {
                "model_id": model_id,
                "city_key": city_key,
                "city_display": sub["city_display"].iloc[0],
                "coefficient_C_per_10pp_canopy": beta,
                "standard_error": se,
                "ci_low": beta - 1.96 * se,
                "ci_high": beta + 1.96 * se,
                "p_value": float(fit.pvalues["canopy_pct_10pp"]),
                "n_block_groups": int(len(work)),
                "r_squared": float(fit.rsquared),
                "covariates": " + ".join(covariates),
                "coefficient_type": "city-specific WLS, HC3 interval",
            }
        )
    return pd.DataFrame(records)


def gam_terms(n_linear: int, n_splines: int = 6):
    terms = l(0)
    for index in range(1, n_linear):
        terms = terms + l(index)
    return terms + te(n_linear, n_linear + 1, n_splines=[n_splines, n_splines], spline_order=[3, 3])


def fit_one_city_gam(
    sub: pd.DataFrame,
    *,
    n_splines: int = GAM_N_SPLINES,
    lam: float = GAM_LAMBDA,
) -> tuple[LinearGAM, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    needed = ["heat_anomaly_C", "population_model_weight", *GAM_LINEAR_COLUMNS, "x_city_0_1", "y_city_0_1"]
    work = sub[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    x = work[[*GAM_LINEAR_COLUMNS, "x_city_0_1", "y_city_0_1"]].to_numpy(float)
    y = work["heat_anomaly_C"].to_numpy(float)
    weights = work["population_model_weight"].to_numpy(float)
    weights = np.clip(weights / np.nanmean(weights), 1e-6, None)
    gam = LinearGAM(
        gam_terms(len(GAM_LINEAR_COLUMNS), n_splines=n_splines),
        lam=[0.0] * len(GAM_LINEAR_COLUMNS) + [[lam, lam]],
        max_iter=250,
        tol=1e-5,
    ).fit(x, y, weights=weights)
    coef_index = gam.terms.get_coef_indices(0)[0]
    beta = float(gam.coef_[coef_index])
    se = float(np.sqrt(max(gam.statistics_["cov"][coef_index, coef_index], 0)))
    prediction = gam.predict(x)
    return gam, work.index.to_numpy(), prediction, np.asarray([beta, se]), x


def reml_random_effects(beta: np.ndarray, se: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    beta = beta[valid]
    se = se[valid]
    if len(beta) < 3:
        raise RuntimeError("At least three city slopes are required for partial pooling.")

    def objective(log_tau2: float) -> float:
        tau2 = math.exp(log_tau2)
        variance = se**2 + tau2
        weights = 1.0 / variance
        mu = np.sum(weights * beta) / np.sum(weights)
        return 0.5 * (
            np.sum(np.log(variance))
            + math.log(np.sum(weights))
            + np.sum((beta - mu) ** 2 / variance)
            + (len(beta) - 1) * math.log(2 * math.pi)
        )

    result = optimize.minimize_scalar(objective, bounds=(-20, 5), method="bounded")
    tau2 = float(math.exp(result.x))
    variance = se**2 + tau2
    weights = 1.0 / variance
    mu = float(np.sum(weights * beta) / np.sum(weights))
    mu_se = float(np.sqrt(1.0 / np.sum(weights)))
    if tau2 <= 1e-12:
        shrunk = np.full_like(beta, mu)
        shrunk_se = np.full_like(se, mu_se)
    else:
        precision = 1.0 / se**2 + 1.0 / tau2
        shrunk = (beta / se**2 + mu / tau2) / precision
        shrunk_se = np.sqrt(1.0 / precision)
    q = float(np.sum((beta - np.average(beta, weights=1 / se**2)) ** 2 / se**2))
    i2 = max(0.0, (q - (len(beta) - 1)) / q) if q > 0 else 0.0
    return {
        "mu": mu,
        "mu_se": mu_se,
        "tau2": tau2,
        "tau": math.sqrt(tau2),
        "i2": i2,
        "shrunk": shrunk,
        "shrunk_se": shrunk_se,
        "valid_mask": valid,
        "optimizer_success": bool(result.success),
    }


def fit_hierarchical_spatial_gam(df: pd.DataFrame) -> FitBundle:
    city_records: list[dict[str, object]] = []
    raw_predictions: list[pd.DataFrame] = []
    for number, (city_key, sub) in enumerate(df.groupby("city_key", sort=True), start=1):
        print(f"[Spatial GAM {number:02d}/50] {city_key}", flush=True)
        gam, index, prediction, coef, x = fit_one_city_gam(sub)
        beta, se = coef
        city_records.append(
            {
                "model_id": "M3_hierarchical_spatial_GAM",
                "city_key": city_key,
                "city_display": sub["city_display"].iloc[0],
                "raw_coefficient_C_per_10pp_canopy": beta,
                "raw_standard_error": se,
                "raw_ci_low": beta - 1.96 * se,
                "raw_ci_high": beta + 1.96 * se,
                "n_block_groups": int(len(index)),
                "spatial_smooth": f"city-specific tensor-product spline s(x,y), {GAM_N_SPLINES} x {GAM_N_SPLINES} basis, lambda={GAM_LAMBDA:g}; linear terms unpenalized",
            }
        )
        raw_predictions.append(
            pd.DataFrame(
                {
                    "row_index": index,
                    "city_key": city_key,
                    "observed": df.loc[index, "heat_anomaly_C"].to_numpy(float),
                    "raw_predicted": prediction,
                    "canopy_pct_10pp": df.loc[index, "canopy_pct_10pp"].to_numpy(float),
                    "population_model_weight": df.loc[index, "population_model_weight"].to_numpy(float),
                }
            )
        )

    city = pd.DataFrame(city_records)
    pool = reml_random_effects(city["raw_coefficient_C_per_10pp_canopy"].to_numpy(float), city["raw_standard_error"].to_numpy(float))
    city["coefficient_C_per_10pp_canopy"] = pool["shrunk"]
    city["standard_error"] = pool["shrunk_se"]
    city["ci_low"] = city["coefficient_C_per_10pp_canopy"] - 1.96 * city["standard_error"]
    city["ci_high"] = city["coefficient_C_per_10pp_canopy"] + 1.96 * city["standard_error"]
    city["coefficient_type"] = "empirical-Bayes city slope after REML partial pooling"

    pred = pd.concat(raw_predictions, ignore_index=True)
    slope_map = city.set_index("city_key")[["raw_coefficient_C_per_10pp_canopy", "coefficient_C_per_10pp_canopy"]]
    pred = pred.join(slope_map, on="city_key")
    city_canopy_mean = pred.groupby("city_key").apply(
        lambda g: weighted_mean(g["canopy_pct_10pp"].to_numpy(float), g["population_model_weight"].to_numpy(float)),
        include_groups=False,
    )
    pred["canopy_city_weighted_mean"] = pred["city_key"].map(city_canopy_mean)
    pred["predicted"] = pred["raw_predicted"] + (
        pred["coefficient_C_per_10pp_canopy"] - pred["raw_coefficient_C_per_10pp_canopy"]
    ) * (pred["canopy_pct_10pp"] - pred["canopy_city_weighted_mean"])
    pred["residual"] = pred["observed"] - pred["predicted"]
    pred["model_id"] = "M3_hierarchical_spatial_GAM"

    critical = float(stats.t.ppf(0.975, df=max(len(city) - 1, 1)))
    national = {
        "model_id": "M3_hierarchical_spatial_GAM",
        "model_family": "two-stage hierarchical city GAM with REML random-effects pooling",
        "outcome": "within-city summer mean LST anomaly (C)",
        "canopy_unit": "10 percentage points",
        "covariates": " + ".join(MODEL2_COVARIATES),
        "spatial_smooth": f"city-specific tensor-product spline s(x,y), {GAM_N_SPLINES} x {GAM_N_SPLINES} basis, lambda={GAM_LAMBDA:g}; linear terms unpenalized",
        "coefficient_C_per_10pp_canopy": pool["mu"],
        "standard_error_city_clustered": pool["mu_se"],
        "ci_low": pool["mu"] - critical * pool["mu_se"],
        "ci_high": pool["mu"] + critical * pool["mu_se"],
        "p_value": float(2 * stats.t.sf(abs(pool["mu"] / pool["mu_se"]), df=max(len(city) - 1, 1))),
        "n_block_groups": int(len(pred)),
        "n_cities": int(city["city_key"].nunique()),
        "tau_between_city": pool["tau"],
        "i2_city_heterogeneity": pool["i2"],
        "inference": "REML random-effects 95% confidence interval",
    }
    return FitBundle("M3_hierarchical_spatial_GAM", national, city, pred)


def spatial_folds(df: pd.DataFrame) -> pd.Series:
    fold = pd.Series(-1, index=df.index, dtype=int)
    for city_key, idx in df.groupby("city_key", sort=True).groups.items():
        coords = df.loc[idx, ["x_5070_m", "y_5070_m"]].to_numpy(float)
        labels = KMeans(n_clusters=N_SPATIAL_FOLDS, random_state=SEED, n_init=20).fit_predict(coords)
        centers = np.asarray([coords[labels == j].mean(axis=0) for j in range(N_SPATIAL_FOLDS)])
        order = np.lexsort((centers[:, 1], centers[:, 0]))
        relabel = {old: new for new, old in enumerate(order)}
        fold.loc[idx] = [relabel[v] for v in labels]
    if (fold < 0).any():
        raise RuntimeError("Spatial fold assignment failed.")
    return fold


def fit_predict_fe(train: pd.DataFrame, test: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    levels = sorted(train["city_key"].astype(str).unique())
    x_train, _ = design_matrix(train, covariates, levels)
    x_test, _ = design_matrix(test, covariates, levels)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    fit = sm.WLS(
        train["heat_anomaly_C"].to_numpy(float),
        x_train,
        weights=train["population_model_weight"].to_numpy(float),
    ).fit()
    return np.asarray(fit.predict(x_test), dtype=float)


def fit_predict_spatial_gam_fold(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    records: list[dict[str, object]] = []
    fitted: dict[str, tuple[LinearGAM, float, float, float]] = {}
    for city_key in sorted(train["city_key"].unique()):
        tr = train.loc[train["city_key"].eq(city_key)].copy()
        gam, index, _, coef, _ = fit_one_city_gam(tr)
        beta, se = coef
        canopy_mean = weighted_mean(tr["canopy_pct_10pp"].to_numpy(float), tr["population_model_weight"].to_numpy(float))
        records.append({"city_key": city_key, "beta": beta, "se": se})
        fitted[city_key] = (gam, beta, se, canopy_mean)
    slope = pd.DataFrame(records)
    pool = reml_random_effects(slope["beta"].to_numpy(float), slope["se"].to_numpy(float))
    slope["shrunk"] = pool["shrunk"]
    shrunk_map = slope.set_index("city_key")["shrunk"].to_dict()
    prediction = pd.Series(np.nan, index=test.index, dtype=float)
    for city_key, sub in test.groupby("city_key", sort=True):
        gam, raw_beta, _, canopy_mean = fitted[city_key]
        x = sub[[*GAM_LINEAR_COLUMNS, "x_city_0_1", "y_city_0_1"]].to_numpy(float)
        raw_prediction = gam.predict(x)
        adjusted = raw_prediction + (shrunk_map[city_key] - raw_beta) * (sub["canopy_pct_10pp"].to_numpy(float) - canopy_mean)
        prediction.loc[sub.index] = adjusted
    return prediction.loc[test.index].to_numpy(float)


def metric_record(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray, model_id: str, fold: int | str, city_key: str = "ALL") -> dict[str, object]:
    valid = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(weights) & (weights > 0)
    y = y[valid]
    prediction = prediction[valid]
    weights = weights[valid]
    error = y - prediction
    mean_y = np.average(y, weights=weights)
    denominator = np.sum(weights * (y - mean_y) ** 2)
    return {
        "model_id": model_id,
        "fold": fold,
        "city_key": city_key,
        "n": int(len(y)),
        "weighted_rmse_C": float(np.sqrt(np.average(error**2, weights=weights))),
        "weighted_mae_C": float(np.average(np.abs(error), weights=weights)),
        "weighted_r2": float(1 - np.sum(weights * error**2) / denominator) if denominator > 0 else float("nan"),
        "unweighted_rmse_C": float(np.sqrt(mean_squared_error(y, prediction))),
        "unweighted_mae_C": float(mean_absolute_error(y, prediction)),
    }


def run_spatial_block_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.loc[df["complete_spatial_model"].astype(bool)].copy()
    work["spatial_fold"] = spatial_folds(work)
    work[["analysis_record_id", "city_key", "GEOID_str", "spatial_fold"]].to_csv(
        OUT_ROOT / "data" / "spatial_block_fold_assignment.csv", index=False, encoding="utf-8-sig"
    )
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for fold in range(N_SPATIAL_FOLDS):
        print(f"[Spatial block CV {fold + 1}/{N_SPATIAL_FOLDS}]", flush=True)
        train = work.loc[work["spatial_fold"].ne(fold)].copy()
        test = work.loc[work["spatial_fold"].eq(fold)].copy()
        predictions = {
            "M1_current_FE_density": fit_predict_fe(train, test, MODEL1_COVARIATES),
            "M2_built_water_terrain_FE": fit_predict_fe(train, test, MODEL2_COVARIATES),
            "M3_hierarchical_spatial_GAM": fit_predict_spatial_gam_fold(train, test),
        }
        for model_id, pred in predictions.items():
            y = test["heat_anomaly_C"].to_numpy(float)
            w = test["population_model_weight"].to_numpy(float)
            metric_rows.append(metric_record(y, pred, w, model_id, fold))
            fold_frame = test[["analysis_record_id", "city_key", "GEOID_str", "heat_anomaly_C", "population_model_weight"]].copy()
            fold_frame["model_id"] = model_id
            fold_frame["fold"] = fold
            fold_frame["predicted"] = pred
            fold_frame["residual"] = fold_frame["heat_anomaly_C"] - fold_frame["predicted"]
            prediction_rows.append(fold_frame)
            for city_key, idx in fold_frame.groupby("city_key").groups.items():
                sub = fold_frame.loc[idx]
                metric_rows.append(
                    metric_record(
                        sub["heat_anomaly_C"].to_numpy(float),
                        sub["predicted"].to_numpy(float),
                        sub["population_model_weight"].to_numpy(float),
                        model_id,
                        fold,
                        city_key,
                    )
                )
    prediction = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    prediction.to_csv(OUT_ROOT / "diagnostics" / "spatial_block_cv_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_ROOT / "diagnostics" / "spatial_block_cv_metrics_by_fold_city.csv", index=False, encoding="utf-8-sig")
    summary = (
        metrics.loc[metrics["city_key"].eq("ALL")]
        .groupby("model_id", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_weighted_rmse_C=("weighted_rmse_C", "mean"),
            sd_weighted_rmse_C=("weighted_rmse_C", "std"),
            mean_weighted_mae_C=("weighted_mae_C", "mean"),
            mean_weighted_r2=("weighted_r2", "mean"),
            mean_unweighted_rmse_C=("unweighted_rmse_C", "mean"),
        )
    )
    summary.to_csv(OUT_ROOT / "tables" / "spatial_block_cv_summary.csv", index=False, encoding="utf-8-sig")
    return prediction, summary


def moran_i(residual: np.ndarray, coords: np.ndarray, permutations: int, seed: int) -> tuple[float, float]:
    valid = np.isfinite(residual) & np.all(np.isfinite(coords), axis=1)
    z = residual[valid].astype(float)
    xy = coords[valid].astype(float)
    n = len(z)
    if n < 4 or np.nanstd(z) <= 0:
        return float("nan"), float("nan")
    z = z - z.mean()
    k = min(K_NEIGHBOURS, n - 1)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(return_distance=False)[:, 1:]

    def statistic(values: np.ndarray) -> float:
        denominator = float(np.sum(values**2))
        if denominator <= 0:
            return float("nan")
        return float(np.sum(values * values[neighbours].mean(axis=1)) / denominator)

    observed = statistic(z)
    if permutations <= 0:
        return observed, float("nan")
    rng = np.random.default_rng(seed)
    null = np.asarray([statistic(rng.permutation(z)) for _ in range(permutations)])
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (permutations + 1)
    return observed, float(p_value)


def residual_moran_diagnostics(df: pd.DataFrame, predictions: pd.DataFrame, permutations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    coord = df.set_index("analysis_record_id")[["x_5070_m", "y_5070_m"]]
    pred = predictions.copy()
    if "analysis_record_id" not in pred.columns:
        index_to_id = df["analysis_record_id"].to_dict()
        pred["analysis_record_id"] = pred["row_index"].map(index_to_id)
    pred = pred.join(coord, on="analysis_record_id")
    records: list[dict[str, object]] = []
    pooled_components: list[dict[str, object]] = []
    for model_number, (model_id, model_df) in enumerate(pred.groupby("model_id", sort=True), start=1):
        pooled_numerator = 0.0
        pooled_denominator = 0.0
        pooled_n = 0
        for city_number, (city_key, sub) in enumerate(model_df.groupby("city_key", sort=True), start=1):
            residual = sub["residual"].to_numpy(float)
            coords = sub[["x_5070_m", "y_5070_m"]].to_numpy(float)
            value, p_value = moran_i(residual, coords, permutations, SEED + model_number * 1000 + city_number)
            records.append(
                {
                    "model_id": model_id,
                    "city_key": city_key,
                    "n_block_groups": int(len(sub)),
                    "moran_i_8nn": value,
                    "permutation_p_value_two_sided": p_value,
                    "permutations": permutations,
                }
            )
            valid = np.isfinite(residual) & np.all(np.isfinite(coords), axis=1)
            z = residual[valid] - np.nanmean(residual[valid])
            xy = coords[valid]
            if len(z) > K_NEIGHBOURS:
                neighbours = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1).fit(xy).kneighbors(return_distance=False)[:, 1:]
                pooled_numerator += float(np.sum(z * z[neighbours].mean(axis=1)))
                pooled_denominator += float(np.sum(z**2))
                pooled_n += len(z)
        pooled_components.append(
            {
                "model_id": model_id,
                "pooled_block_diagonal_moran_i_8nn": pooled_numerator / pooled_denominator if pooled_denominator > 0 else np.nan,
                "n_block_groups": pooled_n,
            }
        )
    city = pd.DataFrame(records)
    pooled = pd.DataFrame(pooled_components)
    summary = (
        city.groupby("model_id", as_index=False)
        .agg(
            cities=("city_key", "nunique"),
            median_city_moran_i=("moran_i_8nn", "median"),
            q25_city_moran_i=("moran_i_8nn", lambda s: s.quantile(0.25)),
            q75_city_moran_i=("moran_i_8nn", lambda s: s.quantile(0.75)),
            maximum_city_moran_i=("moran_i_8nn", "max"),
            positive_city_count=("moran_i_8nn", lambda s: int((s > 0).sum())),
            significant_city_count=("permutation_p_value_two_sided", lambda s: int((s < 0.05).sum())),
        )
        .merge(pooled, on="model_id", how="left")
    )
    city.to_csv(OUT_ROOT / "diagnostics" / "residual_moran_by_city.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_ROOT / "tables" / "residual_moran_summary.csv", index=False, encoding="utf-8-sig")
    return city, summary


def coefficient_change_table(national: pd.DataFrame, city: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = national.set_index("model_id").loc["M1_current_FE_density", "coefficient_C_per_10pp_canopy"]
    national_out = national.copy()
    national_out["change_vs_M1_C_per_10pp"] = national_out["coefficient_C_per_10pp_canopy"] - base
    national_out["magnitude_retained_vs_M1"] = np.where(base != 0, national_out["coefficient_C_per_10pp_canopy"] / base, np.nan)

    wide = city.pivot(index=["city_key", "city_display"], columns="model_id", values="coefficient_C_per_10pp_canopy").reset_index()
    for model_id in ["M2_built_water_terrain_FE", "M3_hierarchical_spatial_GAM"]:
        wide[f"change_{model_id}_vs_M1"] = wide[model_id] - wide["M1_current_FE_density"]
        wide[f"magnitude_retained_{model_id}_vs_M1"] = wide[model_id] / wide["M1_current_FE_density"].replace(0, np.nan)
        wide[f"sign_changed_{model_id}_vs_M1"] = np.sign(wide[model_id]) != np.sign(wide["M1_current_FE_density"])
    return national_out, wide


def priority_tier(i: pd.Series, b: pd.Series, gap: pd.Series, heat: pd.Series) -> pd.Series:
    tier1 = (i >= 0.90) & (b >= 0.90)
    tier2a = (~tier1) & (i >= 0.90) & (b < 0.90)
    tier2b = (~tier1) & (~tier2a) & (i < 0.90) & (b >= 0.90)
    tier3 = (~tier1) & (~tier2a) & (~tier2b) & (gap >= 0.90) & (heat < 0.50)
    out = pd.Series("Lower priority", index=i.index, dtype=object)
    out.loc[tier3] = "Tier 3 high canopy gap but low relievable heat"
    out.loc[tier2b] = "Tier 2B high burden public-health scale priority"
    out.loc[tier2a] = "Tier 2A high intensity environmental-justice priority"
    out.loc[tier1] = "Tier 1 high intensity + high burden"
    return out


def recompute_hvcg_for_models(city_coefficients: pd.DataFrame) -> pd.DataFrame:
    hvcg = pd.read_csv(R4_HVCG, low_memory=False, dtype={"GEOID_str": str})
    hvcg["GEOID_str"] = normalize_geoid(hvcg["GEOID_str"])
    stored = hvcg[
        [
            "city_key", "city_display", "GEOID_str", "population",
            "slope_C_per_10pp_main", "cooling_C_per_10pp_main",
            "canopy_relievable_heat_q75_C", "hvcg_intensity_C",
            "hvcg_burden_person_C", "hvcg_pareto_score_0_100", "hvcg_priority_tier",
        ]
    ].copy()
    stored = stored.rename(
        columns={
            "slope_C_per_10pp_main": "model_city_slope_C_per_10pp",
            "cooling_C_per_10pp_main": "model_cooling_magnitude_C_per_10pp",
            "canopy_relievable_heat_q75_C": "model_canopy_relievable_heat_q75_C",
            "hvcg_intensity_C": "model_hvcg_intensity_C",
            "hvcg_burden_person_C": "model_hvcg_burden_person_C",
            "hvcg_pareto_score_0_100": "model_hvcg_pareto_score_0_100",
            "hvcg_priority_tier": "model_hvcg_priority_tier",
        }
    )
    stored["model_id"] = "M0_legacy_HVCG_stored"
    stored["model_top10_pareto"] = False
    for _, idx in stored.groupby("city_key").groups.items():
        k = max(1, int(math.ceil(len(idx) * 0.10)))
        selected = stored.loc[idx, "model_hvcg_pareto_score_0_100"].nlargest(k).index
        stored.loc[selected, "model_top10_pareto"] = True

    legacy = hvcg[["city_key", "slope_C_per_10pp_main"]].drop_duplicates("city_key").rename(columns={"slope_C_per_10pp_main": "coefficient_C_per_10pp_canopy"})
    legacy["model_id"] = "M0_legacy_rule_harmonized_sample"
    slopes = pd.concat(
        [legacy, city_coefficients[["model_id", "city_key", "coefficient_C_per_10pp_canopy"]]],
        ignore_index=True,
    )
    records: list[pd.DataFrame] = []
    for model_id, slope in slopes.groupby("model_id", sort=True):
        mapping = slope.drop_duplicates("city_key").set_index("city_key")["coefficient_C_per_10pp_canopy"]
        out = hvcg.copy()
        out["model_id"] = model_id
        out["model_city_slope_C_per_10pp"] = out["city_key"].map(mapping)
        out["model_cooling_magnitude_C_per_10pp"] = (-out["model_city_slope_C_per_10pp"]).clip(lower=0)
        out["model_counterfactual_cooling_q75_C"] = out["model_cooling_magnitude_C_per_10pp"] * out["canopy_gap_q75_pct"] / 10.0
        out["model_canopy_relievable_heat_q75_C"] = np.minimum(
            out["heat_excess_reference_C"].clip(lower=0), out["model_counterfactual_cooling_q75_C"].clip(lower=0)
        )
        out["model_hvcg_intensity_C"] = out["model_canopy_relievable_heat_q75_C"] * out["vulnerability_amplification_factor"]
        out["model_hvcg_burden_person_C"] = out["model_hvcg_intensity_C"] * out["population"]
        out["model_intensity_city_percentile"] = out.groupby("city_key")["model_hvcg_intensity_C"].rank(method="average", pct=True)
        out["model_burden_city_percentile"] = out.groupby("city_key")["model_hvcg_burden_person_C"].rank(method="average", pct=True)
        out["model_gap_city_percentile"] = out.groupby("city_key")["canopy_gap_q75_pct"].rank(method="average", pct=True)
        out["model_heat_city_percentile"] = out.groupby("city_key")["model_canopy_relievable_heat_q75_C"].rank(method="average", pct=True)
        i = out["model_intensity_city_percentile"].fillna(0)
        b = out["model_burden_city_percentile"].fillna(0)
        out["model_hvcg_pareto_score_0_100"] = 100.0 * (1.0 - np.sqrt((1.0 - i) ** 2 + (1.0 - b) ** 2) / math.sqrt(2.0))
        out["model_hvcg_pareto_score_0_100"] = out["model_hvcg_pareto_score_0_100"].clip(0, 100)
        out["model_hvcg_priority_tier"] = priority_tier(i, b, out["model_gap_city_percentile"], out["model_heat_city_percentile"])
        out["model_top10_pareto"] = False
        for _, idx in out.groupby("city_key").groups.items():
            k = max(1, int(math.ceil(len(idx) * 0.10)))
            selected = out.loc[idx, "model_hvcg_pareto_score_0_100"].nlargest(k).index
            out.loc[selected, "model_top10_pareto"] = True
        records.append(
            out[
                [
                    "city_key", "city_display", "GEOID_str", "population", "model_id",
                    "model_city_slope_C_per_10pp", "model_cooling_magnitude_C_per_10pp",
                    "model_canopy_relievable_heat_q75_C", "model_hvcg_intensity_C",
                    "model_hvcg_burden_person_C", "model_hvcg_pareto_score_0_100",
                    "model_hvcg_priority_tier", "model_top10_pareto",
                ]
            ]
        )
    result = pd.concat([stored, *records], ignore_index=True)
    result.to_csv(OUT_ROOT / "diagnostics" / "hvcg_membership_by_model.csv", index=False, encoding="utf-8-sig")
    return result


def set_retention(base: pd.DataFrame, candidate: pd.DataFrame, selector: str, city_key: str = "ALL") -> dict[str, object]:
    if selector == "tier1":
        choose_base = base["model_hvcg_priority_tier"].eq("Tier 1 high intensity + high burden")
        choose_candidate = candidate["model_hvcg_priority_tier"].eq("Tier 1 high intensity + high burden")
    elif selector == "tiers1_2":
        labels = {
            "Tier 1 high intensity + high burden",
            "Tier 2A high intensity environmental-justice priority",
            "Tier 2B high burden public-health scale priority",
        }
        choose_base = base["model_hvcg_priority_tier"].isin(labels)
        choose_candidate = candidate["model_hvcg_priority_tier"].isin(labels)
    elif selector == "top10_pareto":
        choose_base = base["model_top10_pareto"].astype(bool)
        choose_candidate = candidate["model_top10_pareto"].astype(bool)
    else:
        raise ValueError(selector)
    base_ids = set(base.loc[choose_base, "GEOID_str"])
    candidate_ids = set(candidate.loc[choose_candidate, "GEOID_str"])
    intersection = base_ids & candidate_ids
    union = base_ids | candidate_ids
    base_pop = base.loc[base["GEOID_str"].isin(base_ids), "population"].sum()
    retained_pop = base.loc[base["GEOID_str"].isin(intersection), "population"].sum()
    return {
        "city_key": city_key,
        "priority_definition": selector,
        "baseline_n": len(base_ids),
        "candidate_n": len(candidate_ids),
        "intersection_n": len(intersection),
        "baseline_retention_fraction": len(intersection) / len(base_ids) if base_ids else np.nan,
        "jaccard": len(intersection) / len(union) if union else np.nan,
        "baseline_population_retained_fraction": retained_pop / base_pop if base_pop > 0 else np.nan,
    }


def hvcg_retention_table(hvcg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = hvcg.loc[hvcg["model_id"].eq("M0_legacy_HVCG_stored")].copy()
    national_records: list[dict[str, object]] = []
    city_records: list[dict[str, object]] = []
    for model_id in sorted(set(hvcg["model_id"]) - {"M0_legacy_HVCG_stored"}):
        candidate = hvcg.loc[hvcg["model_id"].eq(model_id)].copy()
        for selector in ("tier1", "tiers1_2", "top10_pareto"):
            record = set_retention(base, candidate, selector)
            record["candidate_model_id"] = model_id
            national_records.append(record)
            for city_key in sorted(base["city_key"].unique()):
                city_record = set_retention(
                    base.loc[base["city_key"].eq(city_key)],
                    candidate.loc[candidate["city_key"].eq(city_key)],
                    selector,
                    city_key,
                )
                city_record["candidate_model_id"] = model_id
                city_records.append(city_record)
    national = pd.DataFrame(national_records)
    city = pd.DataFrame(city_records)
    national.to_csv(OUT_ROOT / "tables" / "hvcg_high_priority_retention_summary.csv", index=False, encoding="utf-8-sig")
    city.to_csv(OUT_ROOT / "diagnostics" / "hvcg_high_priority_retention_by_city.csv", index=False, encoding="utf-8-sig")
    return national, city


def write_analysis_report(
    national: pd.DataFrame,
    city: pd.DataFrame,
    moran: pd.DataFrame,
    cv: pd.DataFrame | None,
    retention: pd.DataFrame,
) -> None:
    n = national.set_index("model_id")
    m = moran.set_index("model_id")
    lines = [
        "# Canopy-LST spatial residual audit",
        "",
        "## Frozen analytical definition",
        "",
        "- The analysis is fixed to the 38,447 Result 3 block groups in 50 cities.",
        "- Canopy exposure comes from the city-clipped binary U-Net masks in `predict_mosaic_clip2`; no morphological filtering entered block-group calculations.",
        "- Not White alone is fixed as 100 minus the ACS White-alone percentage.",
        "- HVCG tiers are precedence-ordered and mutually exclusive.",
        "- Legacy HVCG slopes and their P-value fallback are retained only as an audit comparator.",
        "",
        "## National canopy coefficients",
        "",
    ]
    for model_id, row in n.iterrows():
        lines.append(
            f"- **{model_id}:** {row['coefficient_C_per_10pp_canopy']:.3f} C per 10 percentage points "
            f"(95% interval {row['ci_low']:.3f} to {row['ci_high']:.3f}; n={int(row['n_block_groups']):,})."
        )
    lines.extend(["", "## Residual spatial structure", ""])
    for model_id, row in m.iterrows():
        lines.append(
            f"- **{model_id}:** pooled block-diagonal 8-NN Moran's I = {row['pooled_block_diagonal_moran_i_8nn']:.3f}; "
            f"median city I = {row['median_city_moran_i']:.3f}; positive in {int(row['positive_city_count'])}/50 cities."
        )
    if cv is not None and not cv.empty:
        lines.extend(["", "## Five-fold spatial-block prediction", ""])
        for row in cv.itertuples(index=False):
            lines.append(
                f"- **{row.model_id}:** weighted RMSE {row.mean_weighted_rmse_C:.3f} C, "
                f"weighted MAE {row.mean_weighted_mae_C:.3f} C, mean held-block R2 {row.mean_weighted_r2:.3f}."
            )
    lines.extend(["", "## HVCG membership stability", ""])
    for row in retention.loc[retention["priority_definition"].eq("tier1")].itertuples(index=False):
        lines.append(
            f"- **{row.candidate_model_id}:** retained {row.baseline_retention_fraction:.1%} of legacy Tier 1 block groups "
            f"(Jaccard {row.jaccard:.3f}; {row.baseline_population_retained_fraction:.1%} of legacy Tier 1 population)."
        )

    m1 = float(n.loc["M1_current_FE_density", "coefficient_C_per_10pp_canopy"])
    m3 = float(n.loc["M3_hierarchical_spatial_GAM", "coefficient_C_per_10pp_canopy"])
    i1 = float(m.loc["M1_current_FE_density", "pooled_block_diagonal_moran_i_8nn"])
    i3 = float(m.loc["M3_hierarchical_spatial_GAM", "pooled_block_diagonal_moran_i_8nn"])
    reduction = 1 - i3 / i1 if i1 != 0 else np.nan
    stable = np.sign(m1) == np.sign(m3) and float(n.loc["M3_hierarchical_spatial_GAM", "ci_high"]) < 0
    lines.extend(
        [
            "",
            "## Decision for Results 3-5",
            "",
            f"- Spatial smoothing changed the national canopy coefficient from {m1:.3f} to {m3:.3f} C per 10 percentage points.",
            f"- Pooled residual Moran's I changed from {i1:.3f} to {i3:.3f}, a relative reduction of {reduction:.1%}.",
            "- " + (
                "The negative association remains statistically distinguishable from zero after spatial adjustment. This supports retaining Result 3 as a cross-sectional association, conditional on the residual diagnostics below."
                if stable
                else "The spatially adjusted association is not stable enough to support canopy-relievable heat as a quantitative downstream input. Results 4-5 should be reframed around descriptive co-occurrence rather than modelled cooling."
            ),
            "- Even a stable coefficient does not identify a planting effect; HVCG and allocation quantities remain association-based screening metrics.",
        ]
    )
    (OUT_ROOT / "text" / "spatial_residual_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_models(skip_cv: bool, moran_permutations: int) -> None:
    frame = load_model_frame()
    frame = frame.loc[frame["complete_spatial_model"].astype(bool)].copy()
    if len(frame) != 38447:
        raise RuntimeError(f"The complete three-model frame has {len(frame):,} rows; all 38,447 frozen rows are required.")

    national_records: list[dict[str, object]] = []
    city_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_id, covariates in (
        ("M1_current_FE_density", MODEL1_COVARIATES),
        ("M2_built_water_terrain_FE", MODEL2_COVARIATES),
    ):
        national, prediction, _ = fit_national_fe(frame, model_id, covariates)
        national_records.append(national)
        city_frames.append(fit_city_wls(frame, model_id, covariates))
        prediction["analysis_record_id"] = prediction["row_index"].map(frame["analysis_record_id"].to_dict())
        prediction_frames.append(prediction)

    hierarchical = fit_hierarchical_spatial_gam(frame)
    national_records.append(hierarchical.national)
    city_frames.append(hierarchical.city)
    hierarchical.predictions["analysis_record_id"] = hierarchical.predictions["row_index"].map(frame["analysis_record_id"].to_dict())
    prediction_frames.append(hierarchical.predictions)

    national = pd.DataFrame(national_records)
    city = pd.concat(city_frames, ignore_index=True, sort=False)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    national, city_changes = coefficient_change_table(national, city)
    national.to_csv(OUT_ROOT / "tables" / "national_canopy_lst_model_comparison.csv", index=False, encoding="utf-8-sig")
    city.to_csv(OUT_ROOT / "tables" / "city_canopy_lst_coefficients_all_models.csv", index=False, encoding="utf-8-sig")
    city_changes.to_csv(OUT_ROOT / "diagnostics" / "city_coefficient_change_vs_current.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_ROOT / "diagnostics" / "in_sample_predictions_and_residuals.csv", index=False, encoding="utf-8-sig")

    moran_city, moran_summary = residual_moran_diagnostics(frame, predictions, moran_permutations)
    if skip_cv:
        cv_summary = None
    else:
        _, cv_summary = run_spatial_block_cv(frame)

    hvcg = recompute_hvcg_for_models(city)
    retention, _ = hvcg_retention_table(hvcg)
    write_analysis_report(national, city, moran_summary, cv_summary, retention)

    qa = {
        "analysis_rows": int(len(frame)),
        "cities": int(frame["city_key"].nunique()),
        "models": national["model_id"].tolist(),
        "model_definitions": {
            "M1_current_FE_density": "Current population-weighted city fixed-effects model with log population density.",
            "M2_built_water_terrain_FE": "M1 plus ESA WorldCover built-up fraction, distance to mapped water, Copernicus DEM elevation and 1-km relief.",
            "M3_hierarchical_spatial_GAM": "M2 covariates plus city-specific 2D tensor-product spatial smooths; city canopy slopes are partially pooled by REML random-effects meta-analysis.",
        },
        "hvcg_slope_policy": "No P-value threshold or pooled fallback is applied to M1-M3. Positive city coefficients imply zero modelled cooling; all city slopes and intervals remain visible.",
        "moran": moran_summary.to_dict(orient="records"),
        "spatial_cv_completed": not skip_cv,
        "hvcg_retention": retention.to_dict(orient="records"),
    }
    write_json(OUT_ROOT / "qa" / "model_audit.json", qa)


def write_readme() -> None:
    text = """# Result 3 spatial residual audit

This directory is an additive audit of the 38,447-block-group Result 3 sample. It does not overwrite the existing Result 3, HVCG or scenario outputs.

## Three model tiers

1. M1 reproduces the population-weighted city fixed-effects plus density specification.
2. M2 adds independently measured built-surface, water-proximity and terrain covariates.
3. M3 adds a city-specific two-dimensional spatial smooth and partially pools city canopy slopes through a REML random-effects layer.

## Interpretation boundary

All coefficients remain cross-sectional associations with Landsat land-surface temperature. The spatial model reduces residual structure and omitted broad spatial trends; it does not identify a causal planting effect, air-temperature response, individual exposure or health outcome.
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_dirs()
    write_readme()
    if args.stage in {"all", "freeze"}:
        freeze_analysis_sample()
    if args.stage in {"all", "covariates"}:
        build_model_frame(overwrite_covariates=args.overwrite_covariates)
    if args.stage in {"all", "models"}:
        run_models(skip_cv=args.skip_cv, moran_permutations=args.moran_permutations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
