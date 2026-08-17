"""Build city-specific neighbourhood evidence layers for the public atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import set_precision


CONTEXT_LABELS = {
    "A": "Joint canopy-deficit, surface-heat and social context",
    "B": "Surface-heat canopy deficit",
    "C": "Social-condition canopy deficit",
    "D": "Canopy-deficit dominant",
    "E": "Heat or social context with no measured canopy deficit",
    "F": "Lower or mixed reference",
}

ALLOCATION_COLUMNS = {
    "uniform": "a_u",
    "canopy_gap": "a_g",
    "surface_heat": "a_h",
    "social_constraint": "a_v",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path)
    registry["rank"] = registry["rank"].astype(int)
    if registry["city_key"].duplicated().any() or registry["slug"].duplicated().any():
        raise RuntimeError("City registry contains duplicate keys.")
    return registry


def find_city_gpkg(root: Path, rank: int) -> Path:
    folders = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(f"{rank:02d}_")]
    if len(folders) != 1:
        raise RuntimeError(f"Expected one ACS folder for rank {rank}, found {len(folders)}.")
    gpkg = folders[0] / "gis" / "block_group_acs2024_5yr.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(gpkg)
    return gpkg


def hard_allocate(capacity: np.ndarray, budget: float, score: np.ndarray) -> np.ndarray:
    allocation = np.zeros_like(capacity, dtype=float)
    order = np.lexsort((np.arange(len(capacity), dtype=np.int64), -score))
    remaining = float(budget)
    for index in order:
        if remaining <= 1e-8:
            break
        take = min(float(capacity[index]), remaining)
        allocation[index] = take
        remaining -= take
    tolerance = max(1e-5, 1e-10 * max(budget, 1.0))
    if remaining > tolerance:
        raise RuntimeError(f"Equal-area budget was not exhausted: {remaining:.6g} m2.")
    return allocation


def build_allocation_wide(frame: pd.DataFrame, budget_pct: float) -> pd.DataFrame:
    fraction = budget_pct / 100.0
    rows: list[pd.DataFrame] = []
    for _, city in frame.sort_values(["city_key", "GEOID_str"]).groupby("city_key", sort=True):
        land = city["land_area_analysis_m2"].to_numpy(float)
        gap = city["G"].to_numpy(float)
        capacity = land * gap / 100.0
        budget = fraction * float(capacity.sum())
        if budget <= 0:
            raise RuntimeError(f"{city.iloc[0]['city_key']} has no positive analytical capacity.")
        allocations = {
            "a_u": capacity * fraction,
            "a_g": hard_allocate(capacity, budget, gap),
            "a_h": hard_allocate(capacity, budget, city["H"].to_numpy(float)),
            "a_v": hard_allocate(capacity, budget, city["V"].to_numpy(float)),
        }
        totals = np.asarray([values.sum() for values in allocations.values()])
        if (totals.max() - totals.min()) / totals.mean() > 1e-10:
            raise RuntimeError(f"{city.iloc[0]['city_key']} does not preserve an equal area budget.")
        out = city[["city_key", "GEOID_str"]].copy()
        for name, values in allocations.items():
            out[name] = np.divide(
                values,
                capacity,
                out=np.zeros_like(values),
                where=capacity > 0,
            )
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def round_or_none(value: object, digits: int) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def city_summary(
    city: pd.DataFrame,
    slug: str,
    name: str,
) -> dict[str, object]:
    population = city["population"].fillna(0).clip(lower=0)
    context_rows: dict[str, object] = {}
    for context_id in CONTEXT_LABELS:
        selected = city["context_id"].eq(context_id)
        context_rows[context_id] = {
            "neighbourhoods": int(selected.sum()),
            "population": int(round(population.loc[selected].sum())),
        }

    strict = city["type_code"].eq("G3H3V3")
    joint = city["context_id"].eq("A")
    def ratio(strict_event: bool) -> float | None:
        g = city["G_level"].eq(3) if strict_event else city["G_level"].ge(2)
        h = city["H_level"].eq(3)
        v = city["V_level"].eq(3)
        n = float(len(city))
        if n < 2:
            return None
        gh = g & h
        gv = g & v
        hv = h & v
        ghv = g & h & v
        expected_count = (
            (n - 3.0) * g.sum() * h.sum() * v.sum()
            + g.sum() * hv.sum()
            + h.sum() * gv.sum()
            + v.sum() * gh.sum()
            - ghv.sum()
        ) / (n - 1.0) ** 3
        expected = expected_count / n
        observed = float(ghv.mean())
        return round_or_none(observed / expected if expected > 0 else None, 3)

    return {
        "slug": slug,
        "name": name,
        "neighbourhoods": int(len(city)),
        "population": int(round(population.sum())),
        "jointContext": {
            "neighbourhoods": int(joint.sum()),
            "population": int(round(population.loc[joint].sum())),
            "observedExpectedRatio": ratio(False),
        },
        "strictTripleHigh": {
            "neighbourhoods": int(strict.sum()),
            "population": int(round(population.loc[strict].sum())),
            "observedExpectedRatio": ratio(True),
        },
        "contexts": context_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--acs-root", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "city_registry.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "neighbourhoods",
    )
    parser.add_argument("--budget-pct", type=float, default=10.0)
    parser.add_argument("--simplify-degrees", type=float, default=0.00002)
    parser.add_argument("--coordinate-precision", type=float, default=0.000001)
    args = parser.parse_args()

    registry = read_registry(args.registry)
    frame = pd.read_csv(args.classification, dtype={"GEOID_str": str}, low_memory=False)
    frame["GEOID_str"] = frame["GEOID_str"].str.zfill(12)
    if len(frame) != 34_566 or frame["city_key"].nunique() != 50:
        raise RuntimeError(
            f"Unexpected frozen support: {len(frame):,} rows and "
            f"{frame['city_key'].nunique()} cities."
        )
    if frame.duplicated(["city_key", "GEOID_str"]).any():
        raise RuntimeError("Classification contains duplicate city-neighbourhood keys.")
    required = {
        "G", "H", "V", "G_level", "H_level", "V_level", "type_code",
        "context_id", "land_area_analysis_m2", "population", "canopy_pct_analysis",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Classification is missing fields: {sorted(missing)}")
    if not frame[["G_level", "H_level", "V_level"]].isin([1, 2, 3]).all().all():
        raise RuntimeError("G, H or V level is outside the frozen three-level definition.")
    expected_type = (
        "G" + frame["G_level"].astype(str)
        + "H" + frame["H_level"].astype(str)
        + "V" + frame["V_level"].astype(str)
    )
    if not expected_type.eq(frame["type_code"]).all():
        raise RuntimeError("One or more 27-type labels disagree with their G/H/V levels.")

    def expected_context(row: pd.Series) -> str:
        g, h, v = int(row.G_level), int(row.H_level), int(row.V_level)
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

    expected_contexts = frame.apply(expected_context, axis=1)
    if not expected_contexts.eq(frame["context_id"]).all():
        raise RuntimeError("One or more six-context labels disagree with their G/H/V levels.")
    if (frame[["G", "H"]] < 0).any().any() or not frame["V"].between(0, 1).all():
        raise RuntimeError("G/H must be non-negative and V must lie in [0, 1].")
    if set(frame["city_key"]) != set(registry["city_key"]):
        raise RuntimeError("Classification and city registry do not contain the same 50 cities.")

    allocation = build_allocation_wide(frame, args.budget_pct)
    frame = frame.merge(
        allocation,
        on=["city_key", "GEOID_str"],
        how="left",
        validate="one_to_one",
    )
    for column in ALLOCATION_COLUMNS.values():
        frame[column] = frame[column].fillna(0.0).clip(0, 1)

    args.output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    release_cities = []
    total_features = 0

    keep = [
        "GEOID_str",
        "population",
        "canopy_pct_analysis",
        "G",
        "H",
        "V",
        "G_level",
        "H_level",
        "V_level",
        "type_code",
        "context_id",
        *ALLOCATION_COLUMNS.values(),
    ]
    property_names = {
        "GEOID_str": "geoid",
        "population": "pop",
        "canopy_pct_analysis": "canopy",
        "G": "g",
        "H": "h",
        "V": "v",
        "G_level": "gl",
        "H_level": "hl",
        "V_level": "vl",
        "type_code": "type",
        "context_id": "context",
    }

    for row in registry.itertuples(index=False):
        city = frame.loc[frame["city_key"].eq(row.city_key)].copy()
        if city.empty:
            raise RuntimeError(f"No classified neighbourhoods for {row.city_key}.")
        gpkg = find_city_gpkg(args.acs_root, int(row.rank))
        geometry = gpd.read_file(gpkg, columns=["GEOID", "geometry"])
        geometry["GEOID_str"] = geometry["GEOID"].astype(str).str.zfill(12)
        joined = geometry[["GEOID_str", "geometry"]].merge(
            city[keep],
            on="GEOID_str",
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(city):
            raise RuntimeError(
                f"Geometry mismatch for {row.city_key}: {len(joined):,}/{len(city):,}."
            )
        joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=geometry.crs).to_crs(4326)
        if (~joined.geometry.is_valid).any():
            joined.geometry = joined.geometry.make_valid()
        joined.geometry = joined.geometry.simplify(
            args.simplify_degrees,
            preserve_topology=True,
        )
        joined.geometry = set_precision(joined.geometry.array, args.coordinate_precision)
        if joined.geometry.is_empty.any() or joined.geometry.isna().any():
            raise RuntimeError(f"Empty geometry produced for {row.city_key}.")

        joined = joined.rename(columns=property_names)
        for column, digits in {
            "canopy": 2,
            "g": 3,
            "h": 3,
            "v": 4,
            "a_u": 4,
            "a_g": 4,
            "a_h": 4,
            "a_v": 4,
        }.items():
            joined[column] = joined[column].round(digits)
        joined["pop"] = joined["pop"].fillna(0).round().astype(int)
        for column in ["gl", "hl", "vl"]:
            joined[column] = joined[column].astype(int)

        output = args.output / f"{row.slug}.geojson"
        output.write_text(joined.to_json(drop_id=True, separators=(",", ":")), encoding="utf-8")
        total_features += len(joined)
        summaries[row.slug] = city_summary(city, row.slug, row.name)
        release_cities.append(
            {
                "slug": row.slug,
                "features": int(len(joined)),
                "file": output.name,
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
            }
        )
        print(f"[{int(row.rank):02d}/50] {row.name}: {len(joined):,} neighbourhoods")

    if total_features != len(frame):
        raise RuntimeError(f"Final feature count mismatch: {total_features:,}/{len(frame):,}.")

    summary_payload = {
        "freezeId": "GHV27-2026-08-09",
        "spatialUnit": "2020 census block group",
        "totalNeighbourhoods": int(total_features),
        "cities": summaries,
        "contextLabels": CONTEXT_LABELS,
        "primaryBudgetPct": args.budget_pct,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    release = {
        "freezeId": summary_payload["freezeId"],
        "classificationSha256": sha256(args.classification),
        "allocationPolicy": "Four display rules recomputed from frozen G, H and V at a city-specific 10% area budget",
        "totalFeatures": int(total_features),
        "cityFiles": release_cities,
    }
    (args.output / "release_manifest.json").write_text(
        json.dumps(release, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {total_features:,} features across {len(release_cities)} cities to {args.output}")


if __name__ == "__main__":
    main()
