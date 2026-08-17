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
DIAGNOSTIC_PATH = (
    RESULT
    / "Result5"
    / "versions"
    / "step2_multicriteria_pareto"
    / "tables"
    / "03_city_tradeoff_diagnostics.csv"
)
OUTCOME_PATH = (
    RESULT
    / "Result5"
    / "versions"
    / "step2_multicriteria_pareto"
    / "tables"
    / "01_city_rule_outcomes.csv"
)
ACS_ROOT = PROJECT / "Top50\u57ce\u5e02_ACS2024_5yr"
BOUNDARY_ROOT = PROJECT / "2024\u5e74\u7f8e\u56fd\u4eba\u53e3top50\u57ce\u5e02" / "city_shp"
OUT = (
    RESULT
    / "Supplementary figure and table"
    / "Supplementary Fig. 15"
    / "versions"
    / "multicriteria_spatial_contrast"
)
FIGURES = OUT / "figures"
SOURCE = OUT / "source_data"
QA = OUT / "qa"
METHODS = OUT / "methods"
TEXT = OUT / "text"

CITY_SPECS = [
    ("Chicago", "Chicagocity", "03_Chicago_city_IL"),
    ("Minneapolis", "Minneapoliscity", "46_Minneapolis_city_MN"),
]

INK = "#171A18"
ANALYSIS_FILL = "#F1F2F0"
CAPACITY_FILL = "#D7E4E0"
UNIT_EDGE = "#FFFFFF"
HEAT = "#D07738"
SOCIAL = "#4E82A3"
BALANCED = "#674679"
CONTEXT_COLORS = {
    "A": "#674679",
    "B": "#D07738",
    "C": "#238779",
    "D": "#4E82A3",
    "E": "#C9A54B",
    "F": "#A9AEAC",
}


def load_legacy():
    name = "result5_spatial_bridge"
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
            "font.size": 6.5,
            "axes.titlesize": 6.6,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.3,
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


def compute_allocations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    summaries = []
    diagnostics = pd.read_csv(DIAGNOSTIC_PATH)
    outcomes = pd.read_csv(OUTCOME_PATH)
    diagnostics = diagnostics.sort_values("heat_social_oracle_overlap").reset_index(drop=True)
    diagnostics["overlap_percentile"] = 100.0 * diagnostics.index / (len(diagnostics) - 1)

    for city, city_key, _ in CITY_SPECS:
        local = frame.loc[frame["city_key"].eq(city_key)].copy().reset_index(drop=True)
        arrays = LEGACY.city_arrays(local)
        population, land, canopy, log_density, temperature, v_indicators, _ = arrays
        gap, _ = LEGACY.recompute_gap(canopy, log_density)
        reference_temperature = float(np.average(temperature, weights=population))
        heat = np.maximum(temperature - reference_temperature, 0.0)
        social = LEGACY.recompute_social_condition(v_indicators)
        capacity = land * gap / 100.0
        budget = 0.10 * float(capacity.sum())
        marginal_heat = 100.0 * population * heat / land
        marginal_social = 100.0 * population * social / land
        heat_allocation = LEGACY.hard_allocate(capacity, budget, marginal_heat)
        social_allocation = LEGACY.hard_allocate(capacity, budget, marginal_social)
        heat_oracle = LEGACY.allocation_quantities(population, land, heat, social, heat_allocation)[0]
        social_oracle = LEGACY.allocation_quantities(population, land, heat, social, social_allocation)[1]
        balanced_score = 0.5 * marginal_heat / heat_oracle + 0.5 * marginal_social / social_oracle
        balanced_allocation = LEGACY.hard_allocate(capacity, budget, balanced_score)
        allocations = {
            "heat_context_oracle": heat_allocation,
            "social_condition_oracle": social_allocation,
            "pareto_0.50": balanced_allocation,
        }
        for rule, values in allocations.items():
            fraction = np.divide(values, capacity, out=np.zeros_like(values), where=capacity > 0)
            for geoid, cap, allocated, share in zip(local["GEOID_str"], capacity, values, fraction):
                records.append(
                    {
                        "city_key": city_key,
                        "city_display": city,
                        "GEOID_str": geoid,
                        "rule": rule,
                        "capacity_m2": cap,
                        "allocated_m2": allocated,
                        "allocation_fraction_of_capacity": share,
                    }
                )
        diagnostic = diagnostics.loc[diagnostics["city_key"].eq(city_key)].iloc[0]
        balanced = outcomes.loc[
            outcomes["city_key"].eq(city_key) & outcomes["rule"].eq("pareto_0.50")
        ].iloc[0]
        overlap = LEGACY.continuous_overlap(heat_allocation, social_allocation)
        summaries.append(
            {
                "city_key": city_key,
                "city_display": city,
                "analysis_block_groups": len(local),
                "capacity_block_groups": int((capacity > 0).sum()),
                "capacity_km2": float(capacity.sum() / 1e6),
                "ten_percent_budget_km2": float(budget / 1e6),
                "heat_social_oracle_overlap": overlap,
                "overlap_percentile": float(diagnostic["overlap_percentile"]),
                "balanced_heat_efficiency": float(balanced["heat_context_efficiency"]),
                "balanced_social_efficiency": float(balanced["social_condition_efficiency"]),
                "balanced_population_efficiency": float(balanced["population_efficiency"]),
                "max_rule_budget_error_m2": max(abs(values.sum() - budget) for values in allocations.values()),
                "diagnostic_overlap_difference": abs(overlap - float(diagnostic["heat_social_oracle_overlap"])),
            }
        )
    return pd.DataFrame(records), pd.DataFrame(summaries)


def load_spatial_frames(
    frame: pd.DataFrame, context: pd.DataFrame
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, gpd.GeoDataFrame]]:
    merged = frame.merge(
        context[["city_key", "GEOID_str", "context_id", "type_code", "G", "H", "V"]],
        on=["city_key", "GEOID_str"],
        how="left",
        validate="one_to_one",
    )
    if merged[["context_id", "type_code"]].isna().any().any():
        raise RuntimeError("Context assignment is incomplete")

    city_frames = {}
    boundaries = {}
    for city, city_key, folder_name in CITY_SPECS:
        attrs = merged.loc[merged["city_key"].eq(city_key)].copy()
        gpkg = ACS_ROOT / folder_name / "gis" / "block_group_acs2024_5yr.gpkg"
        boundary_path = BOUNDARY_ROOT / f"{city} city.shp"
        geometry = gpd.read_file(gpkg, layer="block_group")
        geometry["GEOID_str"] = normalize_geoid(geometry["GEOID"])
        geometry = geometry[["GEOID_str", "geometry"]]
        boundary = gpd.read_file(boundary_path).to_crs(geometry.crs)
        boundary = gpd.GeoDataFrame(geometry=[boundary.geometry.union_all()], crs=geometry.crs)
        local = geometry.merge(attrs, on="GEOID_str", how="inner", validate="one_to_one")
        local = gpd.GeoDataFrame(local, geometry="geometry", crs=geometry.crs)
        local = gpd.clip(local, boundary, keep_geom_type=True)
        if len(local) != len(attrs) or local.geometry.is_empty.any():
            raise RuntimeError(f"{city}: incomplete geometry match")
        city_frames[city] = local
        boundaries[city] = boundary
    return city_frames, boundaries


def map_extent(boundary: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = boundary.total_bounds
    return (
        xmin - 0.035 * (xmax - xmin),
        xmax + 0.035 * (xmax - xmin),
        ymin - 0.035 * (ymax - ymin),
        ymax + 0.035 * (ymax - ymin),
    )


def longitude_label(value: float, _position: int) -> str:
    return f"{abs(value):.2f}\N{DEGREE SIGN}{'W' if value < 0 else 'E'}"


def latitude_label(value: float, _position: int) -> str:
    return f"{abs(value):.2f}\N{DEGREE SIGN}{'N' if value >= 0 else 'S'}"


def interior_ticks(lower: float, upper: float) -> np.ndarray:
    span = upper - lower
    return np.linspace(lower + 0.27 * span, upper - 0.27 * span, 2)


def add_scale_bar(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    latitude = 0.5 * (ymin + ymax)
    width_km = (xmax - xmin) * 111.32 * math.cos(math.radians(latitude))
    target_km = 5 if width_km < 45 else 10
    degree_width = target_km / (111.32 * math.cos(math.radians(latitude)))
    x0 = xmin + 0.065 * (xmax - xmin)
    y0 = ymin + 0.072 * (ymax - ymin)
    ax.plot([x0, x0 + degree_width], [y0, y0], color=INK, lw=1.0, zorder=20)
    for x in (x0, x0 + degree_width):
        ax.plot(
            [x, x],
            [y0 - 0.008 * (ymax - ymin), y0 + 0.008 * (ymax - ymin)],
            color=INK,
            lw=0.7,
            zorder=20,
        )
    ax.text(
        x0 + 0.5 * degree_width,
        y0 + 0.018 * (ymax - ymin),
        f"{target_km} km",
        ha="center",
        va="bottom",
        fontsize=5.2,
        zorder=21,
    )


def style_map_axis(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    add_scale: bool,
) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(FixedLocator(interior_ticks(xmin, xmax)))
    ax.xaxis.set_major_formatter(FuncFormatter(longitude_label))
    ax.yaxis.set_major_locator(FixedLocator(interior_ticks(ymin, ymax)))
    ax.yaxis.set_major_formatter(FuncFormatter(latitude_label))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, length=2.0, width=0.55, pad=1.0)
    ax.tick_params(axis="y", left=True, right=False, labelleft=True, length=2.0, width=0.55, pad=0.8, labelrotation=90)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if add_scale:
        add_scale_bar(ax, extent)


def draw_context(
    ax: plt.Axes,
    local: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
) -> None:
    for context in "FEDCBA":
        subset = local.loc[local["context_id"].eq(context)]
        if len(subset):
            subset.plot(
                ax=ax,
                facecolor=CONTEXT_COLORS[context],
                edgecolor=UNIT_EDGE,
                linewidth=0.12,
                zorder=2,
            )
    boundary.boundary.plot(ax=ax, color=INK, linewidth=0.82, zorder=5)
    style_map_axis(ax, extent, add_scale=True)


def draw_allocation(
    ax: plt.Axes,
    local: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    allocation: pd.DataFrame,
    rule: str,
    color: str,
    extent: tuple[float, float, float, float],
) -> int:
    selected = allocation.loc[
        allocation["rule"].eq(rule),
        ["GEOID_str", "allocated_m2", "allocation_fraction_of_capacity"],
    ]
    mapped = local.merge(selected, on="GEOID_str", how="left", validate="one_to_one")
    mapped["allocated_m2"] = mapped["allocated_m2"].fillna(0.0)
    mapped.plot(ax=ax, facecolor=ANALYSIS_FILL, edgecolor=UNIT_EDGE, linewidth=0.11, zorder=1)
    mapped.loc[mapped["G"].gt(0)].plot(
        ax=ax,
        facecolor=CAPACITY_FILL,
        edgecolor=UNIT_EDGE,
        linewidth=0.11,
        zorder=2,
    )
    allocated = mapped.loc[mapped["allocated_m2"].gt(1e-8)]
    allocated.plot(ax=ax, facecolor=color, edgecolor=UNIT_EDGE, linewidth=0.15, zorder=3)
    boundary.boundary.plot(ax=ax, color=INK, linewidth=0.82, zorder=5)
    style_map_axis(ax, extent, add_scale=False)
    return int(len(allocated))


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

    frame = LEGACY.load_frame()
    frame["GEOID_str"] = normalize_geoid(frame["GEOID_str"])
    context = pd.read_csv(
        CONTEXT_PATH,
        usecols=["city_key", "GEOID_str", "context_id", "type_code", "G", "H", "V"],
        dtype={"GEOID_str": str},
        low_memory=False,
    )
    context["GEOID_str"] = normalize_geoid(context["GEOID_str"])
    allocations, summaries = compute_allocations(frame)
    city_frames, boundaries = load_spatial_frames(frame, context)

    fig = plt.figure(figsize=(7.20, 4.45), facecolor="white")
    lefts = [0.055, 0.292, 0.529, 0.766]
    width = 0.205
    top_y, bottom_y, height = 0.545, 0.155, 0.300
    column_titles = [
        "Neighbourhood context",
        "Surface-heat first",
        "Social-context first",
        r"Equal-weight ($\lambda$ = 0.5)",
    ]
    for left, title in zip(lefts, column_titles):
        fig.text(left + width / 2, 0.875, title, ha="center", va="bottom", fontsize=6.6, fontweight="bold")

    selected_counts = []
    for row_index, (city, city_key, _) in enumerate(CITY_SPECS):
        summary = summaries.loc[summaries["city_key"].eq(city_key)].iloc[0]
        row_y = top_y if row_index == 0 else bottom_y
        title_y = 0.965 if row_index == 0 else 0.505
        descriptor = "low" if row_index == 0 else "higher"
        fig.text(
            0.055,
            title_y,
            f"{'a' if row_index == 0 else 'b'}  {city}: {descriptor} cross-objective allocation overlap "
            f"({summary['heat_social_oracle_overlap']:.2f}; {summary['overlap_percentile']:.0f}th percentile)",
            ha="left",
            va="top",
            fontsize=7.4,
            fontweight="bold",
        )
        local = city_frames[city]
        boundary = boundaries[city]
        extent = map_extent(boundary)
        axes = [fig.add_axes((left, row_y, width, height)) for left in lefts]
        draw_context(axes[0], local, boundary, extent)
        city_allocation = allocations.loc[allocations["city_key"].eq(city_key)]
        selected_counts.append(
            {
                "city": city,
                "heat_selected_block_groups": draw_allocation(
                    axes[1], local, boundary, city_allocation, "heat_context_oracle", HEAT, extent
                ),
                "social_selected_block_groups": draw_allocation(
                    axes[2], local, boundary, city_allocation, "social_condition_oracle", SOCIAL, extent
                ),
                "balanced_selected_block_groups": draw_allocation(
                    axes[3], local, boundary, city_allocation, "pareto_0.50", BALANCED, extent
                ),
            }
        )

    context_handles = [
        Patch(facecolor=CONTEXT_COLORS[key], edgecolor="none", label=f"{key} {label}")
        for key, label in [
            ("A", "joint"),
            ("B", "heat + gap"),
            ("C", "social + gap"),
            ("D", "gap only"),
            ("E", "heat/social, no gap"),
            ("F", "reference"),
        ]
    ]
    fig.legend(
        handles=context_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.083),
        ncol=6,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    allocation_handles = [
        Patch(facecolor=ANALYSIS_FILL, edgecolor="#B8BCBA", label="Analysed block group"),
        Patch(facecolor=CAPACITY_FILL, edgecolor="none", label="Positive analytical capacity"),
        Patch(facecolor=HEAT, edgecolor="none", label="Surface-heat allocation"),
        Patch(facecolor=SOCIAL, edgecolor="none", label="Social-context allocation"),
        Patch(facecolor=BALANCED, edgecolor="none", label="Equal-weight allocation"),
        Line2D([0], [0], color=INK, lw=0.9, label="Official city boundary"),
    ]
    fig.legend(
        handles=allocation_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.027),
        ncol=6,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    overlaps = text_overlap_audit(fig)
    outputs = {
        "png": FIGURES / "Supplementary_Fig15.png",
        "pdf": FIGURES / "Supplementary_Fig15.pdf",
        "svg": FIGURES / "Supplementary_Fig15.svg",
        "tif": FIGURES / "Supplementary_Fig15.tif",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["png"], dpi=450, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["tif"], dpi=450, bbox_inches="tight", pad_inches=0.035, pil_kwargs={"compression": "tiff_lzw"})
    array = np.asarray(Image.open(outputs["png"]).convert("RGB"))
    grayscale = Image.fromarray(array).convert("L")
    grayscale.save(QA / "Supplementary_Fig15_grayscale.png")
    grayscale.resize((max(1, array.shape[1] * 300 // 450), max(1, array.shape[0] * 300 // 450))).save(
        QA / "Supplementary_Fig15_300dpi_preview.png"
    )
    min_font = min(
        float(text.get_fontsize())
        for text in fig.findobj(mpl.text.Text)
        if text.get_visible() and text.get_text().strip()
    )
    plt.close(fig)

    allocations.to_csv(SOURCE / "Supplementary_Fig15_selected_city_allocations.csv.gz", index=False, compression="gzip")
    summaries.to_csv(SOURCE / "Supplementary_Fig15_selected_city_summary.csv", index=False)
    pd.DataFrame(selected_counts).to_csv(SOURCE / "Supplementary_Fig15_map_panel_summary.csv", index=False)
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)

    qa = {
        "status": "PASS",
        "quality_gate_status": "PASS",
        "cities": 2,
        "selected_cities": summaries[["city_display", "heat_social_oracle_overlap", "overlap_percentile"]].to_dict("records"),
        "fixed_analysis_frame": len(frame),
        "all_city_rule_budgets_equal": bool((summaries["max_rule_budget_error_m2"] < 1e-7).all()),
        "diagnostic_overlap_reproduced": bool((summaries["diagnostic_overlap_difference"] < 1e-12).all()),
        "geometry_rows_complete": all(len(city_frames[city]) == int(summaries.loc[summaries["city_display"].eq(city), "analysis_block_groups"].iloc[0]) for city, _, _ in CITY_SPECS),
        "minimum_font_pt": min_font,
        "text_text_overlap_count": len(overlaps),
        "text_text_overlaps": overlaps,
        "exports_exist": all(path.exists() and path.stat().st_size > 10_000 for path in outputs.values()),
        "scientific_claim": "At the same 10% city budget, surface-heat-first and social-context-first allocations occupy more distinct footprints in Chicago than in Minneapolis; the illustrative equal-weight allocation draws from both objectives in each city.",
        "remaining_limitations": [
            "Mapped polygons are census block groups and do not identify parcels or planting sites.",
            "Allocated footprints represent analytical capacity rankings, not observed cooling or health outcomes.",
            "The two cities are mechanism contrasts selected from national overlap percentiles, not representative estimates for all cities.",
        ],
    }
    qa["status"] = "PASS" if all(
        [
            qa["all_city_rule_budgets_equal"],
            qa["diagnostic_overlap_reproduced"],
            qa["geometry_rows_complete"],
            qa["minimum_font_pt"] >= 5.2,
            qa["text_text_overlap_count"] == 0,
            qa["exports_exist"],
        ]
    ) else "FAIL"
    (QA / "Supplementary_Fig15_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    chicago = summaries.loc[summaries["city_display"].eq("Chicago")].iloc[0]
    comparison = summaries.loc[summaries["city_display"].eq("Minneapolis")].iloc[0]
    caption = (
        "Supplementary Fig. 15 | Equal-area objectives produce different neighbourhood footprints when surface-heat and social-context priorities diverge. "
        "Rows show Chicago (a) and Minneapolis (b), chosen as illustrative mechanism contrasts from the lower and upper portions of the national allocation-overlap distribution while requiring a contiguous interpretable census frame. The first column maps the six neighbourhood contexts: A, positive canopy deficit with the upper surface-heat and upper social-condition tiers; B, positive deficit with the upper heat tier but not the upper social-condition tier; C, positive deficit with the upper social-condition tier but not the upper heat tier; D, positive deficit with neither upper tier; E, the upper heat or social-condition tier without a positive canopy deficit; and F, the lower or mixed reference. The remaining columns show block groups receiving any analytical area under the surface-heat-first, social-context-first and illustrative equal-weight rules at the same city-specific 10% budget (4.14 km2 in Chicago and 1.03 km2 in Minneapolis). The continuous area intersection-over-union between the surface-heat-first and social-context-first allocations was "
        f"{chicago['heat_social_oracle_overlap']:.2f} in Chicago (the {chicago['overlap_percentile']:.0f}th national percentile) and "
        f"{comparison['heat_social_oracle_overlap']:.2f} in Minneapolis (the {comparison['overlap_percentile']:.0f}th percentile). "
        "Pale green denotes positive canopy-deficit capacity; coloured polygons show diagnostic block-group allocations, including a partly allocated final unit where required to exhaust the budget. These maps are not parcel-level planting plans or estimates of realized cooling or health benefit."
    )
    (TEXT / "Supplementary_Fig15_caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
