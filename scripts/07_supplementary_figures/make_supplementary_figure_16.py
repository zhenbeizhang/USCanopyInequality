from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box


PROJECT = analysis_store_root()
RESULT_ROOT = PROJECT / "Result"
OUT_ROOT = (
    RESULT_ROOT
    / "Supplementary figure and table"
    / "Supplementary Fig. 16"
    / "versions"
    / "spatial_harmonisation"
)
FIG_DIR = OUT_ROOT / "figures"
SOURCE_DIR = OUT_ROOT / "source_data"
METHOD_DIR = OUT_ROOT / "methods"
QA_DIR = OUT_ROOT / "qa"
TEXT_DIR = OUT_ROOT / "text"
for directory in (FIG_DIR, SOURCE_DIR, METHOD_DIR, QA_DIR, TEXT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ACS_ROOT = PROJECT / "inputs" / "acs"
BG_GPKG = ACS_ROOT / "04_Houston_city_TX" / "gis" / "block_group_acs2024_5yr.gpkg"
TRACT_GPKG = ACS_ROOT / "04_Houston_city_TX" / "gis" / "tract_acs2024_5yr.gpkg"
RGB_RASTER = PROJECT / "inputs" / "canopy" / "source_imagery" / "Houstoncity.tif"
CANOPY_RASTER = PROJECT / "predict_mosaic_clip2" / "Houstoncity_mosaic_clip2.tif"
LST_RASTER = (
    PROJECT
    / "Landsat_Temperature"
    / "Houston_city_Landsat_C2_L2_LST_2020_2024_JJA_PC30.tif"
)
RESULT2_BG = (
    RESULT_ROOT
    / "Result2"
    / "Result2_city_interior_reanalysis"
    / "source_data"
    / "result2_city_interior80_block_group_analysis_dataset.csv"
)
RESULT2_TRACT = (
    RESULT_ROOT
    / "Result2"
    / "Result2_city_interior_reanalysis"
    / "source_data"
    / "result2_city_interior80_tract_analysis_dataset.csv"
)
RESULT3_FRAME = (
    RESULT_ROOT
    / "Result3"
    / "Result3_support_aligned"
    / "data"
    / "support_aligned_model_frame_34566.csv"
)

SELECTED_GEOIDS = ["482015108011", "482015107022", "482015107021"]
STEM = FIG_DIR / "Supplementary_Fig16"

BLACK = "#171717"
CANOPY_GREEN = "#13825f"
BG_RED = "#b9433b"
TRACT_PURPLE = "#65507f"
CONTEXT_GREY = "#f2f2ef"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.2,
            "axes.titlesize": 6.8,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "axes.linewidth": 0.55,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
        }
    )


def normalize_geoid(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def robust_rgb(array: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(array.astype(float), 0, -1)
    out = np.empty_like(rgb, dtype=float)
    for channel in range(3):
        values = rgb[..., channel]
        low, high = np.nanpercentile(values, [1.0, 99.5])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = 0.0, 255.0
        out[..., channel] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.power(out, 0.94)


def fit_bounds_to_aspect(
    original: tuple[float, float, float, float], target_ratio: float = 1.48
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = original
    latitude = 0.5 * (ymin + ymax)
    cos_lat = math.cos(math.radians(latitude))
    dx = (xmax - xmin) * 1.12
    dy = (ymax - ymin) * 1.12
    physical_ratio = dx * cos_lat / dy
    if physical_ratio > target_ratio:
        dy = dx * cos_lat / target_ratio
    else:
        dx = target_ratio * dy / cos_lat
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    return (cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2)


def read_rgb(bounds: tuple[float, float, float, float], output_width: int = 1450) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    latitude = 0.5 * (ymin + ymax)
    ratio = (ymax - ymin) / ((xmax - xmin) * math.cos(math.radians(latitude)))
    output_height = max(500, int(output_width * ratio))
    with rasterio.open(RGB_RASTER) as source:
        if source.crs.to_epsg() != 4326:
            raise RuntimeError(f"Unexpected RGB CRS: {source.crs}")
        window = from_bounds(xmin, ymin, xmax, ymax, source.transform)
        data = source.read(
            [1, 2, 3],
            window=window,
            out_shape=(3, output_height, output_width),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=255,
        )
    return robust_rgb(data)


def read_canopy(
    bounds: tuple[float, float, float, float], shape: tuple[int, int]
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    with rasterio.open(CANOPY_RASTER) as source:
        if source.crs.to_epsg() != 4326:
            raise RuntimeError(f"Unexpected canopy CRS: {source.crs}")
        window = from_bounds(xmin, ymin, xmax, ymax, source.transform)
        data = source.read(
            1,
            window=window,
            out_shape=shape,
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=0,
        )
    return data > 0


def load_geometries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, tuple[float, float, float, float]]:
    block_groups = gpd.read_file(BG_GPKG, layer="block_group").to_crs(4326)
    block_groups["GEOID"] = normalize_geoid(block_groups["GEOID"])
    tracts = gpd.read_file(TRACT_GPKG, layer="tract").to_crs(4326)
    tracts["GEOID"] = normalize_geoid(tracts["GEOID"]).str[:11]
    selected = block_groups.loc[block_groups["GEOID"].isin(SELECTED_GEOIDS)].copy()
    if set(selected["GEOID"]) != set(SELECTED_GEOIDS):
        raise RuntimeError("The three frozen Houston block groups were not recovered")
    display_bounds = fit_bounds_to_aspect(tuple(float(v) for v in selected.total_bounds))
    crop = gpd.GeoDataFrame(geometry=[box(*display_bounds)], crs="EPSG:4326")
    bg_crop = gpd.clip(block_groups, crop, keep_geom_type=True)
    tract_crop = gpd.clip(tracts, crop, keep_geom_type=True)
    return selected, bg_crop, tract_crop, display_bounds


def read_lst_native(
    bounds_wgs84: tuple[float, float, float, float]
) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float], rasterio.crs.CRS, dict[str, float | int | str]]:
    with rasterio.open(LST_RASTER) as source:
        if source.descriptions[0] != "lst_median_c":
            raise RuntimeError(f"Unexpected LST band 1: {source.descriptions[0]}")
        projected = transform_bounds("EPSG:4326", source.crs, *bounds_wgs84, densify_pts=21)
        window = from_bounds(*projected, source.transform).round_offsets().round_lengths()
        data = source.read(1, window=window, masked=True)
        extent = window_bounds(window, source.transform)
        pixel_x = abs(float(source.transform.a))
        pixel_y = abs(float(source.transform.e))
        if not math.isclose(pixel_x, 30.0, abs_tol=1e-8) or not math.isclose(pixel_y, 30.0, abs_tol=1e-8):
            raise RuntimeError("LST raster is not on a 30 m grid")
        valid_city = source.read(1, masked=True).compressed()
        city_low, city_high = np.percentile(valid_city, [2.0, 98.0])
        stats: dict[str, float | int | str] = {
            "band": source.descriptions[0],
            "pixel_size_m": pixel_x,
            "scene_count": int(source.tags().get("scene_count", "0")),
            "years": source.tags().get("years", ""),
            "months": source.tags().get("months", ""),
            "window_valid_pixels": int(data.count()),
            "window_rows": int(data.shape[0]),
            "window_columns": int(data.shape[1]),
            "city_p02_C": float(city_low),
            "city_p98_C": float(city_high),
        }
        return data, tuple(float(v) for v in extent), source.crs, stats


def load_support_counts() -> pd.DataFrame:
    result2_bg = pd.read_csv(RESULT2_BG, usecols=["result_city_key", "spatial_unit_id"], dtype=str)
    result2_tract = pd.read_csv(RESULT2_TRACT, usecols=["result_city_key", "spatial_unit_id"], dtype=str)
    result3 = pd.read_csv(
        RESULT3_FRAME,
        usecols=["city_key", "GEOID_str", "canopy_pct_analysis", "heat_anomaly_C", "mean_LST_C"],
        dtype={"GEOID_str": str},
    )
    rows = [
        {
            "analysis": "Result 1 descriptive atlas",
            "spatial_unit": "atlas-intersecting block group",
            "units": 38618,
            "cities": 50,
        },
        {
            "analysis": "Result 2 primary",
            "spatial_unit": "city-interior block group",
            "units": int(len(result2_bg)),
            "cities": int(result2_bg["result_city_key"].nunique()),
        },
        {
            "analysis": "Result 2 scale sensitivity",
            "spatial_unit": "city-interior tract",
            "units": int(len(result2_tract)),
            "cities": int(result2_tract["result_city_key"].nunique()),
        },
        {
            "analysis": "Results 3-5 fixed frame",
            "spatial_unit": "fully contained block group with LST support",
            "units": int(len(result3)),
            "cities": int(result3["city_key"].nunique()),
        },
    ]
    support = pd.DataFrame(rows)
    expected = [38618, 35612, 12632, 34566]
    if support["units"].tolist() != expected or not (support["cities"] == 50).all():
        raise RuntimeError(f"Unexpected analytical supports:\n{support}")
    selected_frame = result3.loc[result3["GEOID_str"].isin(SELECTED_GEOIDS)].copy()
    if len(selected_frame) != 3 or selected_frame["city_key"].nunique() != 1:
        raise RuntimeError("Houston illustration does not match the fixed Result 3-5 frame")
    selected_frame.to_csv(SOURCE_DIR / "Supplementary_Fig16_selected_units.csv", index=False)
    return support


def lon_formatter(value: float, _position: int | None = None) -> str:
    return f"{abs(value):.2f}°W"


def lat_formatter(value: float, _position: int | None = None) -> str:
    return f"{abs(value):.3f}°N"


def map_ticks_wgs84(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bounds
    ax.xaxis.set_major_locator(FixedLocator([xmin + 0.22 * (xmax - xmin), xmax - 0.22 * (xmax - xmin)]))
    ax.yaxis.set_major_locator(FixedLocator([ymin + 0.25 * (ymax - ymin), ymax - 0.25 * (ymax - ymin)]))
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.tick_params(axis="both", length=2.0, width=0.5, pad=1.0)
    ax.tick_params(axis="y", labelrotation=90)


def projected_lon_ticks(
    ax: plt.Axes,
    bounds_wgs84: tuple[float, float, float, float],
    raster_crs: rasterio.crs.CRS,
) -> None:
    xmin, ymin, xmax, ymax = bounds_wgs84
    to_projected = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    x_lons = [xmin + 0.22 * (xmax - xmin), xmax - 0.22 * (xmax - xmin)]
    x_values = [to_projected.transform(lon, ymin)[0] for lon in x_lons]
    y_lats = [ymin + 0.25 * (ymax - ymin), ymax - 0.25 * (ymax - ymin)]
    y_values = [to_projected.transform(xmin, lat)[1] for lat in y_lats]
    ax.set_xticks(x_values)
    ax.set_xticklabels([lon_formatter(lon) for lon in x_lons])
    ax.set_yticks(y_values)
    ax.set_yticklabels([lat_formatter(lat) for lat in y_lats], rotation=90, va="center")
    ax.tick_params(axis="both", length=2.0, width=0.5, pad=1.0)


def add_scale_bar_wgs84(ax: plt.Axes, bounds: tuple[float, float, float, float], kilometres: float = 0.5) -> None:
    xmin, ymin, xmax, ymax = bounds
    latitude = 0.5 * (ymin + ymax)
    degree_width = kilometres / (111.32 * math.cos(math.radians(latitude)))
    x0 = xmin + 0.055 * (xmax - xmin)
    y0 = ymin + 0.075 * (ymax - ymin)
    ax.plot([x0, x0 + degree_width], [y0, y0], color="white", lw=2.2, solid_capstyle="butt", zorder=30)
    ax.plot([x0, x0 + degree_width], [y0, y0], color=BLACK, lw=0.9, solid_capstyle="butt", zorder=31)
    ax.text(
        x0 + degree_width / 2,
        y0 + 0.025 * (ymax - ymin),
        f"{kilometres:g} km",
        ha="center",
        va="bottom",
        fontsize=5.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
        zorder=32,
    )


def draw_base(ax: plt.Axes, rgb: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bounds
    ax.imshow(rgb, extent=[xmin, xmax, ymin, ymax], origin="upper", interpolation="bilinear", zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(1.0 / math.cos(math.radians(0.5 * (ymin + ymax))), adjustable="box")
    map_ticks_wgs84(ax, bounds)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(BLACK)


def panel_a(
    ax: plt.Axes,
    rgb: np.ndarray,
    canopy: np.ndarray,
    selected: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> None:
    grey = np.dot(rgb[..., :3], np.asarray([0.2126, 0.7152, 0.0722]))
    muted_rgb = 0.22 * rgb + 0.78 * grey[..., None]
    muted_rgb = np.clip(0.94 * muted_rgb + 0.06, 0.0, 1.0)
    draw_base(ax, muted_rgb, bounds)
    rgba = np.zeros((*canopy.shape, 4), dtype=float)
    rgba[..., :3] = mpl.colors.to_rgb(CANOPY_GREEN)
    rgba[..., 3] = np.where(canopy, 0.70, 0.0)
    ax.imshow(rgba, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], origin="upper", interpolation="nearest", zorder=2)
    selected.boundary.plot(ax=ax, color=BG_RED, linewidth=1.0, zorder=5)
    add_scale_bar_wgs84(ax, bounds)


def panel_b(
    ax: plt.Axes,
    rgb: np.ndarray,
    selected: gpd.GeoDataFrame,
    bg_crop: gpd.GeoDataFrame,
    tract_crop: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> None:
    draw_base(ax, rgb, bounds)
    bg_crop.boundary.plot(ax=ax, color="white", linewidth=0.45, alpha=0.9, zorder=3)
    tract_crop.boundary.plot(ax=ax, color=TRACT_PURPLE, linewidth=1.0, linestyle=(0, (3.0, 1.8)), zorder=4)
    selected.boundary.plot(ax=ax, color=BG_RED, linewidth=1.1, zorder=5)


def panel_c(
    ax: plt.Axes,
    lst: np.ma.MaskedArray,
    lst_extent: tuple[float, float, float, float],
    lst_crs: rasterio.crs.CRS,
    selected: gpd.GeoDataFrame,
    bounds_wgs84: tuple[float, float, float, float],
    vmin: float,
    vmax: float,
) -> mpl.image.AxesImage:
    xmin, ymin, xmax, ymax = lst_extent
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad(CONTEXT_GREY, 1.0)
    image = ax.imshow(
        lst,
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    selected.to_crs(lst_crs).boundary.plot(ax=ax, color=BG_RED, linewidth=1.1, zorder=4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    projected_lon_ticks(ax, bounds_wgs84, lst_crs)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(BLACK)
    return image


def save_exports(fig: plt.Figure) -> None:
    fig.savefig(STEM.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), dpi=600, facecolor="white")
    fig.savefig(STEM.with_suffix(".tif"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})


def fingerprint(path: Path) -> dict[str, str | int]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    block = 1024 * 1024
    with path.open("rb") as handle:
        if size <= 512 * 1024 * 1024:
            for chunk in iter(lambda: handle.read(block), b""):
                digest.update(chunk)
            method = "full_file_sha256"
        else:
            digest.update(handle.read(block))
            handle.seek(max(0, size - block))
            digest.update(handle.read(block))
            digest.update(str(size).encode("ascii"))
            method = "sha256_first_last_1MiB_plus_size"
    return {"path": str(path), "size_bytes": size, "fingerprint_method": method, "fingerprint": digest.hexdigest()}


def main() -> int:
    apply_style()
    selected, bg_crop, tract_crop, bounds = load_geometries()
    rgb = read_rgb(bounds)
    canopy = read_canopy(bounds, rgb.shape[:2])
    lst, lst_extent, lst_crs, lst_stats = read_lst_native(bounds)
    support = load_support_counts()

    fig = plt.figure(figsize=(7.20, 2.16), facecolor="white")
    lefts = [0.052, 0.373, 0.694]
    width = 0.276
    bottom = 0.275
    height = 0.620
    axes = [fig.add_axes([left, bottom, width, height]) for left in lefts]
    panel_a(axes[0], rgb, canopy, selected, bounds)
    panel_b(axes[1], rgb, selected, bg_crop, tract_crop, bounds)
    image = panel_c(
        axes[2],
        lst,
        lst_extent,
        lst_crs,
        selected,
        bounds,
        float(lst_stats["city_p02_C"]),
        float(lst_stats["city_p98_C"]),
    )

    titles = [
        "a  0.6 m canopy classification",
        "b  Complete block groups and parent tracts",
        "c  30 m median summer LST",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontweight="bold", pad=3.0, color=BLACK)

    legend_handles = [
        Patch(facecolor=CANOPY_GREEN, edgecolor="none", label="mapped canopy"),
        Line2D([0], [0], color=BG_RED, lw=1.1, label="illustrated block groups"),
        Line2D([0], [0], color=TRACT_PURPLE, lw=1.1, linestyle=(0, (3.0, 1.8)), label="parent tracts"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.348, 0.120),
        ncol=3,
        frameon=False,
        fontsize=5.5,
        handlelength=1.45,
        handletextpad=0.45,
        columnspacing=1.1,
        borderaxespad=0,
    )
    cbar_ax = fig.add_axes([0.708, 0.140, 0.247, 0.036])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal")
    low = float(lst_stats["city_p02_C"])
    high = float(lst_stats["city_p98_C"])
    cbar.set_ticks([round(low), round((low + high) / 2), round(high)])
    cbar.ax.tick_params(labelsize=5.2, length=1.8, width=0.45, pad=1.0)
    cbar.set_label("Median summer LST (°C)", fontsize=5.4, labelpad=1.2)
    cbar.outline.set_linewidth(0.45)

    save_exports(fig)
    plt.close(fig)

    selected_out = selected[["GEOID", "geometry"]].copy()
    selected_out["parent_tract_GEOID"] = selected_out["GEOID"].str[:11]
    selected_out.drop(columns="geometry").to_csv(
        SOURCE_DIR / "Supplementary_Fig16_illustrated_Houston_block_groups.csv", index=False
    )
    support.to_csv(SOURCE_DIR / "Supplementary_Fig16_analysis_support.csv", index=False)
    pd.DataFrame(
        [
            {
                "xmin_WGS84": bounds[0],
                "ymin_WGS84": bounds[1],
                "xmax_WGS84": bounds[2],
                "ymax_WGS84": bounds[3],
                **lst_stats,
            }
        ]
    ).to_csv(SOURCE_DIR / "Supplementary_Fig16_spatial_window.csv", index=False)

    caption = (
        "Supplementary Fig. 16 | Native canopy, census and surface-temperature supports are harmonised without treating their resolutions as interchangeable. "
        "a, Mapped 0.6 m canopy pixels over three adjacent, fully contained Houston census block groups. "
        "b, The same footprint with the illustrated block-group boundaries and their parent census-tract boundaries. "
        "c, Native 30 m pixels from the median June-August 2020-2024 Landsat land-surface-temperature composite over the same footprint; the colour limits are the Houston-wide second and 98th percentiles and are used only to preserve local pixel contrast. "
        "The three block groups were selected before redesign because they form a contiguous, legible spatial-unit example and all occur in the fixed Results 3-5 frame; they are not presented as representative estimates. "
        "Canopy was aggregated to census units, whereas Landsat pixels contributed to census-unit surface-temperature summaries; thermal information was not upsampled to 0.6 m. "
        "The descriptive atlas comprised 38,618 atlas-intersecting block groups, Result 2 used 35,612 city-interior block groups and 12,632 tracts, and Results 3-5 used one fixed set of 34,566 fully contained block groups across all 50 cities. "
        "LST denotes remotely sensed surface thermal context, not air temperature, personal heat exposure or a health outcome."
    )
    (TEXT_DIR / "Supplementary_Fig16_caption.txt").write_text(caption + "\n", encoding="utf-8")

    provenance = {
        "figure": "Supplementary Fig. 16",
        "analysis_role": "spatial harmonisation audit",
        "figure_contract": {
            "core_claim": "Canopy, census and LST layers share a spatial frame but retain distinct native supports.",
            "panel_roles": {
                "a": "0.6 m measurement support",
                "b": "census aggregation support",
                "c": "native 30 m thermal support",
            },
            "principal_limitation": "A schematic grid or a sample-flow slide could obscure whether actual Landsat pixels and frozen census units were used.",
            "response": "Panel c reads the actual 30 m Houston LST raster, while sample counts are confined to the caption and source table.",
        },
        "analysis_support": support.to_dict(orient="records"),
        "selected_GEOIDs": SELECTED_GEOIDS,
        "display_bounds_WGS84": bounds,
        "lst_window": lst_stats,
        "source_files": {
            "Houston_RGB": fingerprint(RGB_RASTER),
            "Houston_canopy": fingerprint(CANOPY_RASTER),
            "Houston_LST": fingerprint(LST_RASTER),
            "Houston_block_groups": fingerprint(BG_GPKG),
            "Houston_tracts": fingerprint(TRACT_GPKG),
            "Result2_block_groups": fingerprint(RESULT2_BG),
            "Result2_tracts": fingerprint(RESULT2_TRACT),
            "Result3_fixed_frame": fingerprint(RESULT3_FRAME),
        },
    }
    (OUT_ROOT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    qa = {
        "status": "CANDIDATE",
        "analysis_units": {row["analysis"]: int(row["units"]) for row in support.to_dict(orient="records")},
        "cities_all_stages": bool((support["cities"] == 50).all()),
        "selected_block_groups": int(len(selected)),
        "actual_lst_grid": bool(float(lst_stats["pixel_size_m"]) == 30.0),
        "actual_lst_band": lst_stats["band"],
        "minimum_font_pt": 5.2,
        "sample_flow_panel_removed": True,
        "obsolete_38_12_gate_removed": True,
        "obsolete_capacity_total_removed": True,
        "exports": [str(STEM.with_suffix(suffix)) for suffix in (".png", ".pdf", ".svg", ".tif")],
    }
    (QA_DIR / "Supplementary_Fig16_build_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
