"""Build the browser city catalogue and simplified city-boundary GeoJSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import mapping


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def clean_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value).replace(":", "-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cog-root", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--tract-frame",
        type=Path,
        required=True,
        help="Frozen fully-contained tract frame with a result_city_key column",
    )
    parser.add_argument("--imagery-dates", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "city_registry.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "catalog",
    )
    args = parser.parse_args()

    registry = read_registry(args.registry)
    metrics = pd.read_csv(args.metrics).set_index("city_key")
    tract_frame = pd.read_csv(args.tract_frame, usecols=["result_city_key"])
    tract_counts = tract_frame["result_city_key"].value_counts()
    if len(tract_frame) != 11_984 or tract_counts.size != 50:
        raise RuntimeError("Expected the frozen 11,984-tract, 50-city frame")
    dates = (
        pd.read_csv(args.imagery_dates).set_index("city_task_name")
        if args.imagery_dates and args.imagery_dates.exists()
        else pd.DataFrame()
    )
    cities = []
    features = []

    for row in registry:
        cog = args.cog_root / f"{row['slug']}.tif"
        source = cog if cog.exists() else None
        if source:
            with rasterio.open(source) as dataset:
                bounds = [dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top]
                raster = {
                    "file": source.name,
                    "width": dataset.width,
                    "height": dataset.height,
                    "crs": str(dataset.crs),
                    "overviews": dataset.overviews(1),
                    "available": True,
                }
        else:
            boundary_preview = gpd.read_file(args.boundary_root / row["boundary_file"]).to_crs(4326)
            west, south, east, north = boundary_preview.total_bounds
            bounds = [west, south, east, north]
            raster = {"file": f"{row['slug']}.tif", "available": False}

        metric = metrics.loc[row["city_key"]]
        date = dates.loc[row["city_key"]] if not dates.empty and row["city_key"] in dates.index else None
        city = {
            "rank": int(row["rank"]),
            "slug": row["slug"],
            "cityKey": row["city_key"],
            "name": row["name"],
            "state": row["state"],
            "bounds": [round(float(value), 7) for value in bounds],
            "center": [
                round(float((bounds[0] + bounds[2]) / 2), 7),
                round(float((bounds[1] + bounds[3]) / 2), 7),
            ],
            "areaWeightedCanopyPct": round(float(metric["area_weighted_canopy_pct"]), 2),
            "populationWeightedCanopyPct": round(float(metric["population_weighted_canopy_pct"]), 2),
            "population": int(round(float(metric["population"]))),
            "tracts": int(tract_counts[row["city_key"]]),
            "blockGroups": int(metric["n_atlas_intersections"]),
            "imagery": {
                "earliest": clean_date(date["earliest_acquisition_date"]) if date is not None else None,
                "median": clean_date(date["median_acquisition_date"]) if date is not None else None,
                "latest": clean_date(date["latest_acquisition_date"]) if date is not None else None,
                "uniqueDates": int(date["unique_acquisition_dates"]) if date is not None else None,
            },
            "raster": raster,
        }
        cities.append(city)

        boundary = gpd.read_file(args.boundary_root / row["boundary_file"]).to_crs(4326)
        # This national overview boundary is display-only; city analysis uses the frozen source geometry.
        geometry = boundary.geometry.union_all().simplify(0.0005, preserve_topology=True)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "slug": row["slug"],
                    "name": row["name"],
                    "state": row["state"],
                    "rank": int(row["rank"]),
                },
                "geometry": mapping(geometry),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cities.json").write_text(json.dumps(cities, indent=2), encoding="utf-8")
    geojson = {"type": "FeatureCollection", "features": features}
    (args.output / "city_boundaries.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    print(f"Wrote {len(cities)} cities and {len(features)} boundaries to {args.output}")


if __name__ == "__main__":
    main()
