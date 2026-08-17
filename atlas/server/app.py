"""Whitelisted raster-tile and point-query service for the canopy atlas."""

from __future__ import annotations

import io
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from morecantile import Tile, tms
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import sieve
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, transform as transform_coordinates, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(os.environ.get("ATLAS_CATALOG", ROOT / "public" / "catalog" / "cities.json"))
COG_ROOT = Path(os.environ.get("ATLAS_COG_ROOT", ROOT / "runtime" / "cogs"))
COG_BASE_URL = os.environ.get("ATLAS_COG_BASE_URL", "").rstrip("/")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ATLAS_ALLOWED_ORIGINS",
        "http://127.0.0.1:5174,http://localhost:5174,"
        "http://127.0.0.1:5175,http://localhost:5175",
    ).split(",")
    if origin.strip()
]
TILE_CACHE_SIZE = int(os.environ.get("ATLAS_TILE_CACHE_SIZE", "2048"))
MIN_CANOPY_COMPONENT_PIXELS = max(
    1,
    int(os.environ.get("ATLAS_MIN_CANOPY_COMPONENT_PIXELS", "6")),
)
CANOPY_FILTER_CONNECTIVITY = 8
CANOPY_FILTER_HALO = MIN_CANOPY_COMPONENT_PIXELS - 1
CANOPY_FILTER_MIN_ZOOM = int(os.environ.get("ATLAS_CANOPY_FILTER_MIN_ZOOM", "15"))
CANOPY_NODATA = 254
CANOPY_VALUE = 255
WEB_MERCATOR_TMS = tms.get("WebMercatorQuad")
WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WGS84_CRS = CRS.from_epsg(4326)

app = FastAPI(title="US Urban Canopy Evidence Atlas", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def city_index() -> dict[str, dict[str, object]]:
    cities = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {city["slug"]: city for city in cities}


def city_cog(slug: str) -> str | Path:
    city = city_index().get(slug)
    if city is None:
        raise HTTPException(status_code=404, detail="Unknown city")
    filename = str(city["raster"]["file"])
    if Path(filename).name != filename:
        raise HTTPException(status_code=500, detail="Invalid catalogue filename")
    if COG_BASE_URL:
        return f"{COG_BASE_URL}/{filename}"
    path = (COG_ROOT / filename).resolve()
    root = COG_ROOT.resolve()
    if root not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Canopy map is not available")
    return path


def transparent_png(size: int = 256) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (size, size), (0, 0, 0, 0)).save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def suppress_small_canopy_components(values: np.ndarray) -> np.ndarray:
    """Retain only 8-connected canopy components with at least six pixels."""
    valid = values != CANOPY_NODATA
    canopy = valid & (values >= 128)
    if MIN_CANOPY_COMPONENT_PIXELS > 1 and canopy.any():
        sieved = sieve(
            canopy.astype(np.uint8),
            size=MIN_CANOPY_COMPONENT_PIXELS,
            connectivity=CANOPY_FILTER_CONNECTIVITY,
        )
        canopy &= sieved > 0
    return np.where(valid, np.where(canopy, CANOPY_VALUE, 0), CANOPY_NODATA).astype(np.uint8)


def filtered_native_tile(
    path: str | Path,
    x: int,
    y: int,
    z: int,
    tilesize: int = 256,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Filter native pixels with a halo, then project them to one web tile."""
    tile = Tile(x=x, y=y, z=z)
    geographic_bounds = WEB_MERCATOR_TMS.bounds(tile)
    projected_bounds = WEB_MERCATOR_TMS.xy_bounds(tile)

    with rasterio.open(path) as source:
        source_bounds = transform_bounds(
            WGS84_CRS,
            source.crs,
            geographic_bounds.left,
            geographic_bounds.bottom,
            geographic_bounds.right,
            geographic_bounds.top,
            densify_pts=21,
        )
        if (
            source_bounds[2] <= source.bounds.left
            or source_bounds[0] >= source.bounds.right
            or source_bounds[3] <= source.bounds.bottom
            or source_bounds[1] >= source.bounds.top
        ):
            return None

        requested = window_from_bounds(*source_bounds, transform=source.transform)
        col_start = math.floor(requested.col_off) - CANOPY_FILTER_HALO
        row_start = math.floor(requested.row_off) - CANOPY_FILTER_HALO
        col_stop = math.ceil(requested.col_off + requested.width) + CANOPY_FILTER_HALO
        row_stop = math.ceil(requested.row_off + requested.height) + CANOPY_FILTER_HALO
        read_window = Window(
            col_start,
            row_start,
            col_stop - col_start,
            row_stop - row_start,
        )
        values = source.read(
            1,
            window=read_window,
            boundless=True,
            fill_value=CANOPY_NODATA,
        )
        cleaned = suppress_small_canopy_components(values)
        destination = np.full((tilesize, tilesize), -9999, dtype=np.float32)
        reproject(
            source=cleaned,
            destination=destination,
            src_transform=window_transform(read_window, source.transform),
            src_crs=source.crs,
            src_nodata=CANOPY_NODATA,
            dst_transform=transform_from_bounds(
                projected_bounds.left,
                projected_bounds.bottom,
                projected_bounds.right,
                projected_bounds.top,
                tilesize,
                tilesize,
            ),
            dst_crs=WEB_MERCATOR_CRS,
            dst_nodata=-9999,
            resampling=Resampling.nearest if z >= 17 else Resampling.average,
        )

    mask = np.where(destination == -9999, 0, 255).astype(np.uint8)
    data = np.where(mask > 0, destination, 0)
    return data, mask


def filtered_point_value(path: str | Path, lon: float, lat: float) -> int:
    """Return the display-filtered native value at a WGS84 point."""
    with rasterio.open(path) as source:
        x, y = lon, lat
        if source.crs != WGS84_CRS:
            xs, ys = transform_coordinates(WGS84_CRS, source.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        if not (source.bounds.left <= x <= source.bounds.right and source.bounds.bottom <= y <= source.bounds.top):
            return CANOPY_NODATA

        row, col = source.index(x, y)
        width = 2 * CANOPY_FILTER_HALO + 1
        values = source.read(
            1,
            window=Window(
                col - CANOPY_FILTER_HALO,
                row - CANOPY_FILTER_HALO,
                width,
                width,
            ),
            boundless=True,
            fill_value=CANOPY_NODATA,
        )
    cleaned = suppress_small_canopy_components(values)
    return int(cleaned[CANOPY_FILTER_HALO, CANOPY_FILTER_HALO])


def canopy_png(data: np.ndarray, mask: np.ndarray, zoom: int) -> bytes:
    values = np.clip(data.astype(np.float32) / 255.0, 0, 1)
    valid = mask > 0
    if zoom >= 17:
        strength = (values >= 0.5).astype(np.float32)
    else:
        strength = values

    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 15
    rgba[..., 1] = 116
    rgba[..., 2] = 83
    rgba[..., 3] = np.where(valid, np.round(strength * 218), 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "cities": len(city_index()),
        "cogSource": "remote" if COG_BASE_URL else "local",
        "canopyDisplayFilter": {
            "minimumPixelsRetained": MIN_CANOPY_COMPONENT_PIXELS,
            "connectivity": CANOPY_FILTER_CONNECTIVITY,
            "nativePixelFilterFromZoom": CANOPY_FILTER_MIN_ZOOM,
        },
    }


@app.get("/api/cities")
def cities() -> list[dict[str, object]]:
    return list(city_index().values())


@lru_cache(maxsize=TILE_CACHE_SIZE)
def render_tile(slug: str, z: int, x: int, y: int) -> bytes:
    path = city_cog(slug)
    if z >= CANOPY_FILTER_MIN_ZOOM:
        filtered = filtered_native_tile(path, x, y, z)
        if filtered is None:
            return transparent_png()
        data, mask = filtered
        return canopy_png(data, mask, z)
    try:
        with Reader(path) as source:
            tile_data = source.tile(
                x,
                y,
                z,
                tilesize=256,
                resampling_method="nearest" if z >= 17 else "average",
            )
    except TileOutsideBounds:
        return transparent_png()
    return canopy_png(tile_data.data[0], tile_data.mask, z)


@app.get("/tiles/{slug}/{z}/{x}/{y}.png")
def tile(slug: str, z: int, x: int, y: int) -> Response:
    limit = 1 << z if 0 <= z <= 20 else 0
    if not limit or not (0 <= x < limit and 0 <= y < limit):
        raise HTTPException(status_code=400, detail="Invalid Web Mercator tile coordinate")
    payload = render_tile(slug, z, x, y)
    return Response(
        payload,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-Canopy-Display-Filter": "8-connected native components of at least 6 pixels",
        },
    )


@app.get("/api/value/{slug}")
def value(slug: str, lon: float, lat: float) -> dict[str, object]:
    path = city_cog(slug)
    try:
        raw = filtered_point_value(path, lon, lat)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Canopy map query failed") from error
    if raw == CANOPY_NODATA:
        classification = "nodata"
    elif raw >= 128:
        classification = "canopy"
    else:
        classification = "not_canopy"
    return {
        "city": slug,
        "longitude": lon,
        "latitude": lat,
        "value": raw,
        "classification": classification,
    }
