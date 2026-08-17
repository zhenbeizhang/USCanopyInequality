"""Build a local GeoJSON outline from the official 2024 Census boundary KML."""

from __future__ import annotations

import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree


SOURCE_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2024/kml/"
    "cb_2024_us_nation_20m.zip"
)
OUTPUT = Path(__file__).resolve().parents[1] / "public" / "catalog" / "us_nation_20m.geojson"
KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def coordinates(text: str | None) -> list[list[float]]:
    if not text:
        return []
    points = []
    for item in text.split():
        longitude, latitude, *_ = item.split(",")
        points.append([round(float(longitude), 6), round(float(latitude), 6)])
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "boundary.zip"
        urllib.request.urlretrieve(SOURCE_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            kml_name = next(name for name in archive.namelist() if name.lower().endswith(".kml"))
            root = ElementTree.fromstring(archive.read(kml_name))

    polygons: list[list[list[list[float]]]] = []
    for polygon in root.findall(".//kml:Polygon", KML_NS):
        outer = polygon.find(
            "./kml:outerBoundaryIs/kml:LinearRing/kml:coordinates",
            KML_NS,
        )
        rings = [coordinates(outer.text if outer is not None else None)]
        for inner in polygon.findall(
            "./kml:innerBoundaryIs/kml:LinearRing/kml:coordinates",
            KML_NS,
        ):
            rings.append(coordinates(inner.text))
        rings = [ring for ring in rings if len(ring) >= 4]
        if rings:
            polygons.append(rings)

    if not polygons:
        raise RuntimeError("No polygons were found in the Census boundary file")

    feature_collection = {
        "type": "FeatureCollection",
        "metadata": {
            "title": "2024 United States national cartographic boundary",
            "source": SOURCE_URL,
            "publisher": "U.S. Census Bureau",
            "resolution": "1:20,000,000",
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "United States"},
                "geometry": {"type": "MultiPolygon", "coordinates": polygons},
            }
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(feature_collection, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} with {len(polygons)} polygons")


if __name__ == "__main__":
    main()
