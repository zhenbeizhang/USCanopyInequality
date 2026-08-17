"""Audit the generated city catalogue, boundaries and canopy COGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]


def audit_cog(path: Path) -> list[str]:
    issues: list[str] = []
    with rasterio.open(path) as dataset:
        if dataset.driver != "GTiff":
            issues.append("driver is not GTiff")
        if dataset.crs is None or dataset.crs.to_epsg() != 4326:
            issues.append("CRS is not EPSG:4326")
        if dataset.count != 1 or dataset.dtypes[0] != "uint8":
            issues.append("expected one uint8 band")
        if dataset.nodata != 254:
            issues.append("nodata is not 254")
        if not dataset.is_tiled:
            issues.append("image is not tiled")
        if not dataset.overviews(1):
            issues.append("overview pyramid is missing")
        if dataset.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG":
            issues.append("GDAL does not report COG layout")
        mask_preview = dataset.read_masks(
            1,
            out_shape=(min(512, dataset.height), min(512, dataset.width)),
            resampling=Resampling.average,
        )
        if not mask_preview.any():
            issues.append("overview mask has no valid pixels")
        width = min(512, dataset.width)
        height = min(512, dataset.height)
        sample = dataset.read(
            1,
            window=Window((dataset.width - width) // 2, (dataset.height - height) // 2, width, height),
            masked=False,
        )
        unexpected = np.setdiff1d(np.unique(sample), np.array([0, 254, 255], dtype=np.uint8))
        if unexpected.size:
            issues.append(f"unexpected native values: {unexpected[:8].tolist()}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "public" / "catalog" / "cities.json",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=ROOT / "public" / "catalog" / "city_boundaries.geojson",
    )
    parser.add_argument("--cog-root", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cities = json.loads(args.catalog.read_text(encoding="utf-8"))
    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))["features"]
    city_slugs = [city["slug"] for city in cities]
    boundary_slugs = [feature["properties"]["slug"] for feature in boundaries]

    global_issues: list[str] = []
    if len(cities) != 50:
        global_issues.append(f"catalogue has {len(cities)} cities, expected 50")
    if len(set(city_slugs)) != len(city_slugs):
        global_issues.append("city slugs are not unique")
    if sorted(city["rank"] for city in cities) != list(range(1, 51)):
        global_issues.append("population ranks are not exactly 1-50")
    if set(city_slugs) != set(boundary_slugs):
        global_issues.append("catalogue and boundary slugs differ")

    city_reports = []
    for city in cities:
        path = args.cog_root / city["raster"]["file"]
        if not path.exists():
            issues = ["COG is missing"] if args.require_all else []
            status = "missing"
        else:
            issues = audit_cog(path)
            status = "ok" if not issues else "failed"
        city_reports.append(
            {
                "slug": city["slug"],
                "status": status,
                "bytes": path.stat().st_size if path.exists() else 0,
                "issues": issues,
            }
        )

    report = {
        "catalogueCities": len(cities),
        "boundaries": len(boundaries),
        "availableCogs": sum(item["status"] != "missing" for item in city_reports),
        "validCogs": sum(item["status"] == "ok" for item in city_reports),
        "globalIssues": global_issues,
        "cities": city_reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cities"}, indent=2))

    failures = global_issues or any(item["issues"] for item in city_reports)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
