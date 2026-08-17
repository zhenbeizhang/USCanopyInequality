"""Convert city canopy masks to web-ready Cloud Optimized GeoTIFFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def convert_one(
    source: Path,
    destination: Path,
    metadata: dict[str, str],
    overwrite: bool,
    gdal_threads: str,
    compression_level: int,
) -> dict[str, object]:
    started = time.perf_counter()
    if destination.exists() and not overwrite:
        status = "existing"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial.tif")
        temporary.unlink(missing_ok=True)
        try:
            rio_copy(
                source,
                temporary,
                driver="COG",
                BLOCKSIZE="512",
                COMPRESS="DEFLATE",
                LEVEL=str(compression_level),
                BIGTIFF="IF_SAFER",
                OVERVIEWS="AUTO",
                OVERVIEW_RESAMPLING="AVERAGE",
                NUM_THREADS=gdal_threads,
            )
            with rasterio.open(temporary) as dataset:
                if not dataset.is_tiled or not dataset.overviews(1):
                    raise RuntimeError(f"COG validation failed for {temporary}")
                mask_preview = dataset.read_masks(
                    1,
                    out_shape=(min(512, dataset.height), min(512, dataset.width)),
                    resampling=Resampling.average,
                )
                if not mask_preview.any():
                    raise RuntimeError(f"COG overview mask has no valid pixels for {temporary}")
            temporary.replace(destination)
        finally:
            for artifact in temporary.parent.glob(f"{temporary.name}*"):
                artifact.unlink(missing_ok=True)
        status = "created"

    with rasterio.open(destination) as dataset:
        report = {
            "slug": metadata["slug"],
            "city": metadata["name"],
            "source_file": source.name,
            "output_file": destination.name,
            "status": status,
            "width": dataset.width,
            "height": dataset.height,
            "crs": str(dataset.crs),
            "dtype": dataset.dtypes[0],
            "nodata": dataset.nodata,
            "is_tiled": dataset.is_tiled,
            "block_shape": list(dataset.block_shapes[0]),
            "overviews": dataset.overviews(1),
            "compression": "DEFLATE",
            "compression_level": compression_level,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
    if not report["is_tiled"] or not report["overviews"]:
        raise RuntimeError(f"COG validation failed for {destination}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "city_registry.csv",
    )
    parser.add_argument("--cities", nargs="*", help="Optional city slugs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gdal-threads", default="4")
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    all_rows = load_registry(args.registry)
    rows = all_rows
    selected = set(args.cities or [])
    if selected:
        rows = [row for row in rows if row["slug"] in selected]
        missing = selected - {row["slug"] for row in rows}
        if missing:
            raise SystemExit(f"Unknown city slugs: {', '.join(sorted(missing))}")

    jobs = []
    for row in rows:
        source = args.source_root / f"{row['mask_stem']}.tif"
        if not source.exists():
            raise FileNotFoundError(source)
        jobs.append(
            (
                source,
                args.output_root / f"{row['slug']}.tif",
                row,
                args.overwrite,
                args.gdal_threads,
                args.compression_level,
            )
        )

    reports: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    if args.workers == 1:
        for index, job in enumerate(jobs, start=1):
            try:
                report = convert_one(*job)
                reports.append(report)
                print(
                    f"[{index}/{len(jobs)}] {report['city']}: {report['status']} "
                    f"({report['elapsed_seconds']} s)",
                    flush=True,
                )
            except Exception as error:
                failure = {"slug": job[2]["slug"], "city": job[2]["name"], "error": str(error)}
                failures.append(failure)
                print(
                    f"[{index}/{len(jobs)}] {failure['city']}: FAILED ({failure['error']})",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(convert_one, *job): job[2] for job in jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                city = row["name"]
                try:
                    report = future.result()
                    reports.append(report)
                    print(
                        f"[{index}/{len(jobs)}] {report['city']}: {report['status']} "
                        f"({report['elapsed_seconds']} s)",
                        flush=True,
                    )
                except Exception as error:
                    failure = {"slug": row["slug"], "city": city, "error": str(error)}
                    failures.append(failure)
                    print(
                        f"[{index}/{len(jobs)}] {city}: FAILED ({error})",
                        file=sys.stderr,
                        flush=True,
                    )

    manifest = args.output_root / "cog_manifest.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    merged = {item["slug"]: item for item in previous}
    merged.update({item["slug"]: item for item in reports})
    rank = {row["slug"]: int(row["rank"]) for row in all_rows}
    ordered = sorted(merged.values(), key=lambda item: rank[item["slug"]])
    manifest.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"Wrote {manifest}")
    failure_manifest = args.output_root / "cog_failures.json"
    if failures:
        failure_manifest.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise SystemExit(f"{len(failures)} city COG conversion(s) failed; see {failure_manifest}")
    failure_manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
