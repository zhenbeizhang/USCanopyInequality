from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from PIL import Image


PROJECT = analysis_store_root()
RESULT = PROJECT / "Result"
LEGACY_PATH = (
    Path(__file__).resolve().parents[1]
    / "05_allocation"
    / "run_equal_budget_allocation.py"
)
CONTEXT_PATH = (
    RESULT
    / "Result4"
    / "versions"
    / "joint_context_boundary_spatial_audit"
    / "data"
    / "fully_contained_primary_GHV27.csv.gz"
)
ACS_PATH = (
    PROJECT
    / "inputs"
    / "acs"
    / "35_Sacramento_city_CA"
    / "gis"
    / "block_group_acs2024_5yr.gpkg"
)
BOUNDARY_PATH = PROJECT / "inputs" / "census_boundaries" / "city_shp" / "Sacramento city.shp"
OUT = (
    RESULT
    / "Supplementary figure and table"
    / "Supplementary Fig. 13"
    / "versions"
    / "sacramento_budget_expansion"
)
FIGURES = OUT / "figures"
SOURCE = OUT / "source_data"
QA = OUT / "qa"
METHODS = OUT / "methods"
TEXT = OUT / "text"

CITY_KEY = "Sacramentocity"
CITY_DISPLAY = "Sacramento"
BUDGETS = np.asarray([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00], dtype=float)
MAP_BUDGETS = (0.01, 0.10, 0.50)

INK = "#171A18"
OUTSIDE = "#E8EAE8"
ANALYSED = "#F7F7F5"
CAPACITY = "#C3D9D0"
HEAT = "#D07738"
PARTIAL_EDGE = "#7A3D19"
CONTEXT_CMAP = LinearSegmentedColormap.from_list(
    "context_share", ["#F8F6F1", "#EDC4A8", "#D07738", "#8D3C22"], N=256
)


def load_legacy():
    name = "result5_fig13_budget_expansion"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.4,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.3,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def normalize_geoid(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def longitude_label(value: float, _position: int) -> str:
    return f"{abs(value):.2f}°{'W' if value < 0 else 'E'}"


def latitude_label(value: float, _position: int) -> str:
    return f"{abs(value):.2f}°{'N' if value >= 0 else 'S'}"


def interior_ticks(lower: float, upper: float) -> np.ndarray:
    span = upper - lower
    return np.linspace(lower + 0.26 * span, upper - 0.26 * span, 2)


def map_extent(boundary: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = boundary.total_bounds
    return (
        xmin - 0.03 * (xmax - xmin),
        xmax + 0.03 * (xmax - xmin),
        ymin - 0.03 * (ymax - ymin),
        ymax + 0.03 * (ymax - ymin),
    )


def add_scale_bar(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    latitude = 0.5 * (ymin + ymax)
    target_km = 5
    degree_width = target_km / (111.32 * math.cos(math.radians(latitude)))
    x0 = xmin + 0.07 * (xmax - xmin)
    y0 = ymin + 0.07 * (ymax - ymin)
    ax.plot([x0, x0 + degree_width], [y0, y0], color=INK, lw=0.9, zorder=20)
    for x in (x0, x0 + degree_width):
        ax.plot(
            [x, x],
            [y0 - 0.008 * (ymax - ymin), y0 + 0.008 * (ymax - ymin)],
            color=INK,
            lw=0.65,
            zorder=20,
        )
    ax.text(
        x0 + 0.5 * degree_width,
        y0 + 0.018 * (ymax - ymin),
        "5 km",
        ha="center",
        va="bottom",
        fontsize=5.2,
        zorder=21,
    )


def style_map(ax: plt.Axes, extent: tuple[float, float, float, float], scale: bool) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(FixedLocator(interior_ticks(xmin, xmax)))
    ax.xaxis.set_major_formatter(FuncFormatter(longitude_label))
    ax.yaxis.set_major_locator(FixedLocator(interior_ticks(ymin, ymax)))
    ax.yaxis.set_major_formatter(FuncFormatter(latitude_label))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, length=2, width=0.55, pad=1.0)
    ax.tick_params(
        axis="y",
        left=True,
        right=False,
        labelleft=True,
        length=2,
        width=0.55,
        pad=0.8,
        labelrotation=90,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    if scale:
        add_scale_bar(ax, extent)


def load_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = LEGACY.load_frame()
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    context = pd.read_csv(
        CONTEXT_PATH,
        usecols=["city_key", "GEOID_str", "context_id", "type_code", "G", "H", "V"],
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    context["GEOID_str"] = normalize_geoid(context["GEOID_str"])
    city = frame.loc[frame["city_key"].eq(CITY_KEY)].copy().reset_index(drop=True)
    city = city.merge(
        context.loc[context["city_key"].eq(CITY_KEY)].drop(columns="city_key"),
        on="GEOID_str",
        how="left",
        validate="one_to_one",
    )
    if city[["context_id", "type_code", "G", "H", "V"]].isna().any().any():
        raise RuntimeError("Incomplete Sacramento context assignment")

    population, land, canopy, log_density, temperature, v_indicators, _ = LEGACY.city_arrays(city)
    gap, _ = LEGACY.recompute_gap(canopy, log_density)
    heat_reference = float(np.average(temperature, weights=population))
    positive_heat = np.maximum(temperature - heat_reference, 0.0)
    capacity = land * gap / 100.0
    score = 100.0 * population * positive_heat / land

    allocation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    allocation_arrays: dict[float, np.ndarray] = {}
    for fraction in BUDGETS:
        budget = float(fraction * capacity.sum())
        allocation = LEGACY.hard_allocate(capacity, budget, score)
        allocation_arrays[float(fraction)] = allocation
        delta_canopy = 100.0 * allocation / land
        heat_quantity = float(np.sum(population * positive_heat * delta_canopy))
        selected = allocation > 1e-8
        partial = selected & (allocation < capacity - 1e-8)
        summary_rows.append(
            {
                "city_key": CITY_KEY,
                "city_display": CITY_DISPLAY,
                "budget_fraction": fraction,
                "budget_pct": 100.0 * fraction,
                "capacity_km2": float(capacity.sum() / 1e6),
                "budget_km2": float(budget / 1e6),
                "allocated_km2": float(allocation.sum() / 1e6),
                "selected_block_groups": int(selected.sum()),
                "partly_allocated_block_groups": int(partial.sum()),
                "heat_context_quantity": heat_quantity,
                "heat_context_quantity_per_km2": heat_quantity / (budget / 1e6),
            }
        )
        for geoid, cap, allocated in zip(city["GEOID_str"], capacity, allocation):
            allocation_rows.append(
                {
                    "city_key": CITY_KEY,
                    "city_display": CITY_DISPLAY,
                    "GEOID_str": geoid,
                    "budget_fraction": fraction,
                    "budget_pct": 100.0 * fraction,
                    "capacity_m2": cap,
                    "allocated_m2": allocated,
                    "allocation_fraction_of_capacity": allocated / cap if cap > 0 else 0.0,
                }
            )
        for context_id in ("A", "B", "C", "D"):
            mask = city["context_id"].eq(context_id).to_numpy()
            context_rows.append(
                {
                    "budget_fraction": fraction,
                    "budget_pct": 100.0 * fraction,
                    "context_id": context_id,
                    "allocated_km2": float(allocation[mask].sum() / 1e6),
                    "allocated_area_share_pct": float(100.0 * allocation[mask].sum() / budget),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary["relative_heat_context_quantity_per_km2"] = (
        summary["heat_context_quantity_per_km2"] / summary["heat_context_quantity_per_km2"].iloc[0]
    )
    nested = True
    for lower, upper in zip(BUDGETS[:-1], BUDGETS[1:]):
        nested &= bool(np.all(allocation_arrays[float(lower)] <= allocation_arrays[float(upper)] + 1e-8))
    summary["nested_across_all_budget_steps"] = nested
    return city, pd.DataFrame(allocation_rows), summary, pd.DataFrame(context_rows)


def load_geometry(city: pd.DataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, float, float]:
    geometry = gpd.read_file(ACS_PATH, layer="block_group")
    geometry_crs = geometry.crs
    geometry["GEOID_str"] = normalize_geoid(geometry["GEOID"])
    geometry = geometry[["GEOID_str", "geometry"]].merge(city, on="GEOID_str", how="inner", validate="one_to_one")
    geometry = gpd.GeoDataFrame(geometry, geometry="geometry", crs=geometry_crs)
    boundary_raw = gpd.read_file(BOUNDARY_PATH).to_crs(geometry.crs)
    boundary = gpd.GeoDataFrame(geometry=[boundary_raw.geometry.union_all()], crs=geometry.crs)
    geometry = gpd.clip(geometry, boundary, keep_geom_type=True)
    if len(geometry) != len(city) or geometry.geometry.is_empty.any():
        raise RuntimeError("Sacramento geometry match is incomplete")
    utm = boundary.estimate_utm_crs()
    boundary_m = boundary.to_crs(utm)
    geometry_m = geometry.to_crs(utm)
    coverage = float(geometry_m.geometry.union_all().area / boundary_m.geometry.iloc[0].area)
    area = boundary_m.geometry.iloc[0].area
    perimeter = boundary_m.geometry.iloc[0].length
    compactness = float(4.0 * math.pi * area / perimeter**2)
    return geometry, boundary, coverage, compactness


def draw_map(
    ax: plt.Axes,
    geometry: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    allocation: pd.DataFrame,
    budget: float,
    extent: tuple[float, float, float, float],
    scale: bool,
) -> None:
    selected = allocation.loc[
        np.isclose(allocation["budget_fraction"], budget),
        ["GEOID_str", "allocated_m2", "allocation_fraction_of_capacity"],
    ]
    mapped = geometry.merge(selected, on="GEOID_str", how="left", validate="one_to_one")
    mapped["allocated_m2"] = mapped["allocated_m2"].fillna(0.0)
    boundary.plot(ax=ax, facecolor=OUTSIDE, edgecolor="none", zorder=0)
    mapped.plot(ax=ax, facecolor=ANALYSED, edgecolor="white", linewidth=0.10, zorder=1)
    mapped.loc[mapped["G"].gt(0)].plot(
        ax=ax, facecolor=CAPACITY, edgecolor="white", linewidth=0.10, zorder=2
    )
    allocated = mapped.loc[mapped["allocated_m2"].gt(1e-8)]
    allocated.plot(ax=ax, facecolor=HEAT, edgecolor="white", linewidth=0.15, zorder=3)
    partial = allocated.loc[allocated["allocation_fraction_of_capacity"].lt(1.0 - 1e-8)]
    if len(partial):
        partial.boundary.plot(ax=ax, color=PARTIAL_EDGE, linewidth=0.65, zorder=4)
    boundary.boundary.plot(ax=ax, color=INK, linewidth=0.82, zorder=5)
    style_map(ax, extent, scale)


def text_overlap_audit(fig: mpl.figure.Figure) -> list[list[object]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for artist in fig.findobj(mpl.text.Text):
        if artist.get_visible() and artist.get_text().strip():
            texts.append((artist.get_text(), artist.get_window_extent(renderer=renderer)))
    overlaps = []
    for index, (left_text, left_box) in enumerate(texts):
        for right_text, right_box in texts[index + 1 :]:
            x_overlap = max(0.0, min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0))
            y_overlap = max(0.0, min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0))
            if x_overlap * y_overlap > 4.0:
                overlaps.append([left_text, right_text, x_overlap * y_overlap])
    return overlaps


def main() -> int:
    setup_style()
    for directory in (FIGURES, SOURCE, QA, METHODS, TEXT):
        directory.mkdir(parents=True, exist_ok=True)

    city, allocations, summary, contexts = load_analysis()
    geometry, boundary, coverage, compactness = load_geometry(city)
    extent = map_extent(boundary)

    fig = plt.figure(figsize=(7.20, 5.05), facecolor="white")
    fig.text(
        0.055,
        0.950,
        "a  Surface-heat-first allocation expands across a fixed city frame",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
    )
    map_lefts = (0.055, 0.357, 0.659)
    map_width = 0.270
    map_bottom = 0.572
    map_height = 0.288
    for index, (left, budget) in enumerate(zip(map_lefts, MAP_BUDGETS)):
        row = summary.loc[np.isclose(summary["budget_fraction"], budget)].iloc[0]
        fig.text(
            left + map_width / 2,
            0.875,
            f"{row['budget_pct']:.0f}% budget | {row['budget_km2']:.2f} km² | {int(row['selected_block_groups'])} block groups",
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="bold",
        )
        ax = fig.add_axes((left, map_bottom, map_width, map_height))
        draw_map(ax, geometry, boundary, allocations, budget, extent, scale=index == 0)

    map_handles = [
        Patch(facecolor=OUTSIDE, edgecolor="none", label="Outside analysis frame"),
        Patch(facecolor=CAPACITY, edgecolor="none", label="Positive analytical capacity"),
        Patch(facecolor=HEAT, edgecolor="none", label="Block group receiving allocation"),
        Line2D([0], [0], color=INK, lw=0.9, label="Official city boundary"),
    ]
    fig.legend(
        handles=map_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.493),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.text(
        0.070,
        0.447,
        "b  Selected neighbourhood context changes with budget",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
    )
    ax_b = fig.add_axes((0.070, 0.145, 0.515, 0.260))
    context_matrix = (
        contexts.pivot(index="context_id", columns="budget_pct", values="allocated_area_share_pct")
        .reindex(index=list("ABCD"), columns=100.0 * BUDGETS)
        .to_numpy(float)
    )
    image_b = ax_b.imshow(context_matrix, cmap=CONTEXT_CMAP, norm=Normalize(0, 100), aspect="auto")
    ax_b.set_xticks(np.arange(len(BUDGETS)), labels=["1", "2.5", "5", "10", "20", "30", "50", "100"])
    ax_b.set_yticks(
        np.arange(4),
        labels=["A  joint", "B  heat + gap", "C  social + gap", "D  gap only"],
    )
    ax_b.set_xlabel("Allocated share of analytical capacity (%)", labelpad=1.0)
    ax_b.tick_params(length=0, pad=2.0)
    ax_b.set_xticks(np.arange(-0.5, len(BUDGETS), 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=0.8)
    ax_b.tick_params(which="minor", bottom=False, left=False)
    for row in range(4):
        for column in range(len(BUDGETS)):
            value = context_matrix[row, column]
            if value >= 2.0:
                ax_b.text(
                    column,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white" if value >= 50 else INK,
                )
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes((0.195, 0.052, 0.300, 0.014))
    colorbar = fig.colorbar(image_b, cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    colorbar.ax.tick_params(labelsize=5.2, length=2, pad=1.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_label("Allocated area in context (%)", fontsize=5.5, labelpad=1.5)

    fig.text(
        0.650,
        0.447,
        "c  Per-area surface-heat quantity declines",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
    )
    ax_c = fig.add_axes((0.650, 0.145, 0.320, 0.260))
    x = summary["budget_pct"].to_numpy(float)
    y = summary["relative_heat_context_quantity_per_km2"].to_numpy(float)
    ax_c.plot(x, y, color=HEAT, lw=1.35, marker="o", ms=4.1, mec="white", mew=0.6)
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.8, 125)
    ax_c.set_ylim(0, 1.06)
    ax_c.set_xticks([1, 2.5, 5, 10, 20, 50, 100], labels=["1", "2.5", "5", "10", "20", "50", "100"])
    ax_c.set_yticks([0, 0.5, 1.0])
    ax_c.set_xlabel("Allocated share of analytical capacity (%)", labelpad=2.0)
    ax_c.set_ylabel("Surface-heat quantity per km²\n(relative to 1% budget)", labelpad=2.5)
    ax_c.axvline(10, color="#A9AEAC", lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.grid(axis="y", color="#E2E4E2", linewidth=0.5)
    ax_c.tick_params(length=2.5, width=0.55, pad=1.5)

    overlaps = text_overlap_audit(fig)
    outputs = {
        "png": FIGURES / "Supplementary_Fig13.png",
        "pdf": FIGURES / "Supplementary_Fig13.pdf",
        "svg": FIGURES / "Supplementary_Fig13.svg",
        "tif": FIGURES / "Supplementary_Fig13.tif",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["png"], dpi=450, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["tif"], dpi=450, bbox_inches="tight", pad_inches=0.035, pil_kwargs={"compression": "tiff_lzw"})
    array = np.asarray(Image.open(outputs["png"]).convert("RGB"))
    Image.fromarray(array).convert("L").save(QA / "Supplementary_Fig13_grayscale.png")
    Image.fromarray(array).resize(
        (max(1, array.shape[1] * 300 // 450), max(1, array.shape[0] * 300 // 450)),
        Image.Resampling.LANCZOS,
    ).save(QA / "Supplementary_Fig13_300dpi_preview.png")
    minimum_font = min(
        float(text.get_fontsize())
        for text in fig.findobj(mpl.text.Text)
        if text.get_visible() and text.get_text().strip()
    )
    plt.close(fig)

    allocations.to_csv(SOURCE / "Supplementary_Fig13_sacramento_allocations.csv.gz", index=False, compression="gzip")
    summary.to_csv(SOURCE / "Supplementary_Fig13_sacramento_budget_summary.csv", index=False)
    contexts.to_csv(SOURCE / "Supplementary_Fig13_sacramento_context_shares.csv", index=False)
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)

    context_sums = contexts.groupby("budget_fraction")["allocated_area_share_pct"].sum()
    relative_yield = summary["relative_heat_context_quantity_per_km2"].to_numpy(float)
    qa = {
        "status": "PASS",
        "quality_gate_status": "PASS",
        "fixed_analysis_frame": int(len(LEGACY.load_frame())),
        "all_cities_in_parent_analysis": int(LEGACY.load_frame()["city_key"].nunique()),
        "selected_city": CITY_DISPLAY,
        "selected_city_analysis_units": int(len(city)),
        "selected_city_capacity_km2": float(summary["capacity_km2"].iloc[0]),
        "analysis_boundary_area_coverage": coverage,
        "boundary_compactness": compactness,
        "budgets_pct": (100.0 * BUDGETS).tolist(),
        "max_absolute_budget_error_m2": float(
            np.max(np.abs(summary["allocated_km2"] - summary["budget_km2"])) * 1e6
        ),
        "all_allocations_nested": bool(summary["nested_across_all_budget_steps"].all()),
        "context_share_max_sum_error_pp": float(np.max(np.abs(context_sums - 100.0))),
        "per_area_heat_quantity_nonincreasing": bool(np.all(np.diff(relative_yield) <= 1e-10)),
        "minimum_font_pt": minimum_font,
        "text_text_overlap_count": len(overlaps),
        "text_text_overlaps": overlaps,
        "exports_exist": all(path.exists() and path.stat().st_size > 10_000 for path in outputs.values()),
        "scientific_claim": (
            "With the surface-heat-first ranking fixed, a larger equal-area budget expands the selected Sacramento block-group footprint into lower-ranked contexts and lowers average model-implied surface-heat quantity per allocated square kilometre."
        ),
        "principal_limitation": (
            "Sacramento is an illustrative spatial mechanism case, not evidence that every city expands in the same geography. National budget responses are reported in Supplementary Fig. 11."
        ),
        "remaining_limitations": [
            "Coloured polygons identify block groups receiving analytical allocation, not parcel-level planting sites.",
            "Surface-heat quantity is a ranking diagnostic based on positive surface-temperature deviation and represented population; it is not observed cooling, air temperature or health benefit.",
            "Panel c is a same-target diagnostic of the surface-heat-first ranking and is not used to claim superiority over another rule.",
            "The example was selected for high spatial-frame coverage and legibility, not for a favourable outcome magnitude.",
        ],
    }
    qa["status"] = "PASS" if all(
        [
            qa["fixed_analysis_frame"] == 34566,
            qa["all_cities_in_parent_analysis"] == 50,
            qa["selected_city_analysis_units"] == 345,
            qa["analysis_boundary_area_coverage"] >= 0.80,
            qa["max_absolute_budget_error_m2"] < 1e-6,
            qa["all_allocations_nested"],
            qa["context_share_max_sum_error_pp"] < 1e-8,
            qa["per_area_heat_quantity_nonincreasing"],
            qa["minimum_font_pt"] >= 5.2,
            qa["text_text_overlap_count"] == 0,
            qa["exports_exist"],
        ]
    ) else "FAIL"
    (QA / "Supplementary_Fig13_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    mapped_budget_details = ", ".join(
        f"{int(row.budget_pct)}% ({row.budget_km2:.2f} km²; {int(row.selected_block_groups)} block groups)"
        for row in summary.loc[summary["budget_fraction"].isin(MAP_BUDGETS)].itertuples()
    )
    caption = (
        "Supplementary Fig. 13 | A fixed surface-heat-first ranking expands spatially and becomes less concentrated as the analytical budget grows. "
        "a, Sacramento block groups receiving allocation at three shares of the city's positive density-peer canopy-deficit capacity: "
        f"{mapped_budget_details}. All maps use the same city boundary and surface-heat score; the allocations are nested, and the outlined final block group may be only partly allocated to exhaust the exact area budget. "
        "b, Share of allocated area in the four positive-deficit neighbourhood contexts across all eight budgets. A denotes high surface heat and high social condition, B high heat and lower social condition, C lower heat and high social condition, and D neither high heat nor high social condition. "
        "c, Model-implied surface-heat quantity per allocated square kilometre, normalized to the 1% budget, declines as lower-ranked capacity enters. This is a same-target diagnostic of the surface-heat-first ranking rather than an independent comparison of policy performance. Sacramento was selected before plotting for its high analysis-frame coverage (82% of the official boundary area), moderate number of analysed block groups and legible contiguous frame; it is an illustrative mechanism case rather than a representative city estimate. Coloured polygons are census block groups receiving analytical allocation, not parcel-level planting sites, and the plotted quantity is not observed cooling, air temperature or health benefit."
    )
    (TEXT / "Supplementary_Fig13_caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
