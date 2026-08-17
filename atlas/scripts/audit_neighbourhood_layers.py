"""Audit the versioned neighbourhood evidence release used by the atlas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = {"A", "B", "C", "D", "E", "F"}
ALLOCATIONS = ("a_u", "a_g", "a_h", "a_v")
TYPE_PATTERN = re.compile(r"^G([123])H([123])V([123])$")
EXPECTED_CONTEXT_COUNTS = {
    "A": 3_602,
    "B": 5_006,
    "C": 7_259,
    "D": 9_978,
    "E": 3_590,
    "F": 5_131,
}
EXPECTED_JOINT_POPULATION = 4_592_253


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_context(g: int, h: int, v: int) -> str:
    if g >= 2 and h == 3 and v == 3:
        return "A"
    if g >= 2 and h == 3:
        return "B"
    if g >= 2 and v == 3:
        return "C"
    if g >= 2:
        return "D"
    if h == 3 or v == 3:
        return "E"
    return "F"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "public" / "neighbourhoods")
    parser.add_argument("--registry", type=Path, default=ROOT / "config" / "city_registry.csv")
    args = parser.parse_args()

    with args.registry.open(newline="", encoding="utf-8") as stream:
        registry = list(csv.DictReader(stream))
    slugs = {row["slug"] for row in registry}
    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "release_manifest.json").read_text(encoding="utf-8"))

    issues: list[str] = []
    if len(registry) != 50 or set(summary["cities"]) != slugs:
        issues.append("registry and summary do not contain the same 50 cities")
    if summary.get("freezeId") != "GHV27-2026-08-09":
        issues.append("unexpected typology freeze identifier")

    manifest_files = {item["slug"]: item for item in manifest.get("cityFiles", [])}
    if set(manifest_files) != slugs:
        issues.append("manifest city files do not match the registry")

    total = 0
    all_geoids: set[str] = set()
    context_counts = {context: 0 for context in CONTEXTS}
    for slug in sorted(slugs):
        path = args.root / f"{slug}.geojson"
        if not path.exists():
            issues.append(f"{slug}: GeoJSON is missing")
            continue
        entry = manifest_files.get(slug, {})
        if entry.get("sha256") != sha256(path):
            issues.append(f"{slug}: SHA-256 differs from the manifest")
        collection = json.loads(path.read_text(encoding="utf-8"))
        features = collection.get("features", [])
        total += len(features)
        if len(features) != summary["cities"][slug]["neighbourhoods"]:
            issues.append(f"{slug}: feature count differs from summary")

        city_contexts = {context: 0 for context in CONTEXTS}
        for feature in features:
            properties = feature.get("properties", {})
            geoid = str(properties.get("geoid", ""))
            if not re.fullmatch(r"\d{12}", geoid) or geoid in all_geoids:
                issues.append(f"{slug}: invalid or duplicate GEOID {geoid!r}")
                break
            all_geoids.add(geoid)
            match = TYPE_PATTERN.fullmatch(str(properties.get("type", "")))
            if not match:
                issues.append(f"{slug}/{geoid}: invalid 27-type code")
                break
            levels = tuple(int(value) for value in match.groups())
            if levels != (properties.get("gl"), properties.get("hl"), properties.get("vl")):
                issues.append(f"{slug}/{geoid}: type code and level fields disagree")
                break
            context = properties.get("context")
            if context not in CONTEXTS or context != expected_context(*levels):
                issues.append(f"{slug}/{geoid}: invalid six-context assignment")
                break
            if float(properties.get("g", -1)) < 0 or float(properties.get("h", -1)) < 0:
                issues.append(f"{slug}/{geoid}: negative G or H value")
                break
            if not 0 <= float(properties.get("v", -1)) <= 1:
                issues.append(f"{slug}/{geoid}: V lies outside [0, 1]")
                break
            if not 0 <= float(properties.get("canopy", -1)) <= 100:
                issues.append(f"{slug}/{geoid}: canopy lies outside [0, 100]")
                break
            if any(not 0 <= float(properties.get(field, -1)) <= 1 for field in ALLOCATIONS):
                issues.append(f"{slug}/{geoid}: allocation fraction lies outside [0, 1]")
                break
            geometry_type = feature.get("geometry", {}).get("type")
            if geometry_type not in {"Polygon", "MultiPolygon"}:
                issues.append(f"{slug}/{geoid}: geometry is not polygonal")
                break
            city_contexts[context] += 1
            context_counts[context] += 1

        expected_counts = {
            context: summary["cities"][slug]["contexts"][context]["neighbourhoods"]
            for context in CONTEXTS
        }
        if city_contexts != expected_counts:
            issues.append(f"{slug}: context counts differ from summary")

    if total != 34_566 or len(all_geoids) != 34_566:
        issues.append(f"release contains {total:,} features and {len(all_geoids):,} unique GEOIDs")
    if summary.get("totalNeighbourhoods") != total or manifest.get("totalFeatures") != total:
        issues.append("summary or manifest total differs from GeoJSON files")
    if context_counts != EXPECTED_CONTEXT_COUNTS:
        issues.append(f"context counts differ from the frozen evidence layer: {context_counts}")
    joint_population = sum(
        city["jointContext"]["population"] for city in summary.get("cities", {}).values()
    )
    if joint_population != EXPECTED_JOINT_POPULATION:
        issues.append(
            "joint-context population differs from the frozen evidence layer: "
            f"{joint_population:,}"
        )

    report = {
        "freezeId": summary.get("freezeId"),
        "cities": len(slugs),
        "features": total,
        "uniqueGeoids": len(all_geoids),
        "contexts": dict(sorted(context_counts.items())),
        "jointContextPopulation": joint_population,
        "issues": issues,
    }
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
