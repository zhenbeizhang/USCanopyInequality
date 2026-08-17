from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from PIL import Image


PROJECT = analysis_store_root()
ROOT = (
    PROJECT
    / "Result"
    / "Result4"
    / "versions"
    / "joint_context_boundary_spatial_audit"
)
DATA = ROOT / "data" / "fully_contained_primary_GHV27.csv.gz"
COOCCURRENCE = ROOT / "tables" / "03_cooccurrence_benchmarks.csv"
RANGES = (
    PROJECT
    / "Result"
    / "Result3"
    / "Result3_support_aligned"
    / "spatial_block_bootstrap"
    / "tables"
    / "city_residual_range_and_block_sizes.csv"
)
STATES = (
    PROJECT
    / "Result"
    / "Result1"
    / "Result1_NatureCities_manuscript_Fig1"
    / "basemap"
    / "ne_50m_admin_1_states_provinces.shp"
)
FIGURES = ROOT / "figures"
SOURCE = ROOT / "source_data" / "Fig4"
QA = ROOT / "qa"
METHODS = ROOT / "methods"

INK = "#171A18"
MUTED = "#626967"
PURPLE = "#6D4A78"
PURPLE_DARK = "#4C3156"
TEAL = "#238779"
ORANGE = "#D27738"
LIGHT_LINE = "#D7DCDA"
STATE_FILL = "#F1F3F2"
STATE_EDGE = "#C8CECB"


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.6,
            "axes.titlesize": 7.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.7,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def panel_heading(fig: mpl.figure.Figure, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x, y, letter, fontsize=7.7, fontweight="bold", ha="left", va="bottom")
    fig.text(x + 0.018, y, title, fontsize=7.3, fontweight="bold", ha="left", va="bottom")


def density_panels(
    fig: mpl.figure.Figure,
    frame: pd.DataFrame,
) -> tuple[list[mpl.axes.Axes], pd.DataFrame, mpl.cm.ScalarMappable]:
    x_cap = float(frame["G"].quantile(0.99))
    y_cap = float(frame["H"].quantile(0.99))
    x_edges = np.linspace(0, x_cap, 33)
    y_edges = np.linspace(0, y_cap, 27)
    histograms: dict[int, np.ndarray] = {}
    source_rows: list[dict[str, float | int]] = []
    max_count = 1
    for level in (1, 2, 3):
        local = frame.loc[frame["V_level"].eq(level)]
        x = np.minimum(local["G"].to_numpy(float), np.nextafter(x_cap, 0))
        y = np.minimum(local["H"].to_numpy(float), np.nextafter(y_cap, 0))
        hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        histograms[level] = hist
        max_count = max(max_count, int(hist.max()))
        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                source_rows.append(
                    {
                        "V_level": level,
                        "G_bin_low_pp": x_edges[i],
                        "G_bin_high_pp": x_edges[i + 1],
                        "H_bin_low_C": y_edges[j],
                        "H_bin_high_C": y_edges[j + 1],
                        "block_groups": int(hist[i, j]),
                    }
                )
    cmap = LinearSegmentedColormap.from_list(
        "density_teal",
        ["#FFFFFF", "#DCEBE7", "#85BEB4", TEAL, "#13534B"],
    )
    cmap.set_under("white")
    norm = LogNorm(vmin=1, vmax=max_count)
    positions = ((0.075, 0.585, 0.245, 0.305), (0.350, 0.585, 0.245, 0.305), (0.625, 0.585, 0.245, 0.305))
    axes: list[mpl.axes.Axes] = []
    labels = {1: "Lower tier (V1)", 2: "Middle tier (V2)", 3: "Upper tier (V3)"}
    g_cut = float(frame.loc[frame["G_level"].eq(2), "G"].max())
    h_cut = float(frame.loc[frame["H_level"].eq(2), "H"].max())
    mappable: mpl.cm.ScalarMappable | None = None
    for level, position in zip((1, 2, 3), positions):
        ax = fig.add_axes(position)
        axes.append(ax)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            histograms[level].T,
            cmap=cmap,
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        mappable = mesh
        ax.axvline(g_cut, color=ORANGE, lw=0.75, ls=(0, (3, 2)), zorder=4)
        ax.axhline(h_cut, color=PURPLE, lw=0.75, ls=(0, (3, 2)), zorder=4)
        if level == 3:
            ax.add_patch(
                mpl.patches.Rectangle(
                    (0, h_cut),
                    x_cap,
                    y_cap - h_cut,
                    fill=False,
                    edgecolor=PURPLE_DARK,
                    linewidth=1.1,
                    zorder=5,
                )
            )
        ax.set_xlim(0, x_cap)
        ax.set_ylim(0, y_cap)
        ax.set_title(f"{labels[level]}   n={len(frame.loc[frame['V_level'].eq(level)]):,}", pad=3.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        if level == 1:
            ax.set_ylabel("Positive LST anomaly, H (°C)")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Canopy deficit, G (percentage points)")
    if mappable is None:
        raise RuntimeError("No density mappable")
    cax = fig.add_axes((0.885, 0.615, 0.010, 0.235))
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_label("Block groups per bin", labelpad=3)
    colorbar.outline.set_linewidth(0.5)
    fig.text(0.879, 0.868, "Density", fontsize=5.8, ha="center", va="bottom")
    return axes, pd.DataFrame(source_rows), mappable


def benchmark_panel(fig: mpl.figure.Figure, point: pd.DataFrame, ranges: pd.DataFrame) -> tuple[mpl.axes.Axes, pd.DataFrame]:
    ax = fig.add_axes((0.075, 0.105, 0.320, 0.315))
    median_sizes = {
        scale: float(ranges[f"{scale}_block_size_km"].median())
        for scale in ("large", "main", "small")
    }
    order = (
        "city_marginal_independence",
        "large_block_conditioned",
        "main_block_conditioned",
        "small_block_conditioned",
    )
    labels = (
        "City\nmarginals",
        f"Large blocks\n{median_sizes['large']:.1f} km",
        f"Primary blocks\n{median_sizes['main']:.1f} km",
        f"Small blocks\n{median_sizes['small']:.1f} km",
    )
    data = point.loc[
        point["event"].eq("joint_context")
        & point["benchmark"].isin(order)
        & point["grid_origin"].isin(["not_applicable", "origin_mean"])
    ].set_index("benchmark").loc[list(order)].reset_index()
    x = np.arange(len(data))
    y = data["observed_expected_ratio"].to_numpy(float)
    low = data["bootstrap_p025"].to_numpy(float)
    high = data["bootstrap_p975"].to_numpy(float)
    ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.fill_between(x, low, high, color="#DDCFE1", alpha=0.70, zorder=2)
    ax.plot(x, y, color=PURPLE_DARK, lw=1.35, marker="o", ms=4.2, mfc="white", mew=1.0, zorder=3)
    for index, value in enumerate(y):
        ax.text(index, high[index] + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6.1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Observed / expected prevalence")
    ax.set_ylim(0.96, max(high) + 0.13)
    ax.set_xlim(-0.25, 3.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    return ax, data


def map_panel(fig: mpl.figure.Figure, frame: pd.DataFrame, states: gpd.GeoDataFrame) -> tuple[mpl.axes.Axes, pd.DataFrame]:
    ax = fig.add_axes((0.500, 0.080, 0.430, 0.355))
    mainland = states.loc[
        states["adm0_a3"].eq("USA")
        & states["postal"].notna()
        & ~states["postal"].isin(["AK", "HI", "PR"])
    ].to_crs("EPSG:5070")
    mainland.plot(ax=ax, color=STATE_FILL, edgecolor=STATE_EDGE, linewidth=0.40, zorder=1)
    city = (
        frame.assign(joint=frame["context_id"].eq("A"))
        .groupby(["city_key", "city_display"], as_index=False)
        .agg(
            joint_context_share=("joint", "mean"),
            block_groups=("GEOID_str", "size"),
            x_5070_m=("x_5070_m", "median"),
            y_5070_m=("y_5070_m", "median"),
        )
    )
    vmax = max(0.25, float(np.ceil(city["joint_context_share"].max() * 20) / 20))
    norm = Normalize(vmin=0, vmax=vmax)
    scatter = ax.scatter(
        city["x_5070_m"],
        city["y_5070_m"],
        c=city["joint_context_share"],
        cmap="Purples",
        norm=norm,
        s=31,
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )
    selected = pd.concat(
        [city.nsmallest(2, "joint_context_share"), city.nlargest(3, "joint_context_share")]
    ).drop_duplicates("city_key")
    offsets = {
        "left": (-10, 2),
        "right": (8, 2),
    }
    for row in selected.itertuples(index=False):
        side = "left" if row.x_5070_m > city["x_5070_m"].median() else "right"
        dx, dy = offsets[side]
        ax.annotate(
            f"{row.city_display} {100*row.joint_context_share:.1f}%",
            (row.x_5070_m, row.y_5070_m),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if side == "left" else "left",
            va="center",
            fontsize=5.5,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.45, shrinkA=1, shrinkB=2),
            zorder=4,
        )
    ax.set_axis_off()
    ax.set_aspect("equal")
    cax = fig.add_axes((0.595, 0.060, 0.245, 0.014))
    colorbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    colorbar.set_label("Joint-context block groups (%)", labelpad=1.5)
    ticks = np.linspace(0, vmax, 4)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{100*tick:.0f}" for tick in ticks])
    colorbar.outline.set_linewidth(0.5)
    return ax, city


def export_and_qa(fig: mpl.figure.Figure, outputs: dict[str, Path]) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped_text: list[str] = []
    width, height = fig.canvas.get_width_height()
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer=renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            clipped_text.append(text.get_text())
    fig.savefig(outputs["png"], dpi=400, facecolor="white")
    fig.savefig(outputs["pdf"], facecolor="white")
    fig.savefig(outputs["svg"], facecolor="white")
    fig.savefig(outputs["tif"], dpi=400, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    image = Image.open(outputs["png"]).convert("RGB")
    array = np.asarray(image)
    nonwhite = np.any(array < 250, axis=2)
    rows, columns = np.where(nonwhite)
    margins = {
        "left_fraction": float(columns.min() / array.shape[1]),
        "right_fraction": float((array.shape[1] - 1 - columns.max()) / array.shape[1]),
        "top_fraction": float(rows.min() / array.shape[0]),
        "bottom_fraction": float((array.shape[0] - 1 - rows.max()) / array.shape[0]),
    }
    grey = Image.fromarray(array).convert("L")
    grey.save(QA / "Fig4_grayscale.png")
    return {
        "clipped_text": clipped_text,
        "canvas_pixels": [array.shape[1], array.shape[0]],
        "visible_artist_margins": margins,
        "exports_exist": all(path.exists() and path.stat().st_size > 10_000 for path in outputs.values()),
    }


def main() -> int:
    setup_style()
    for directory in (FIGURES, SOURCE, QA, METHODS):
        directory.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, dtype={"GEOID_str": str}, low_memory=False)
    point = pd.read_csv(COOCCURRENCE)
    ranges = pd.read_csv(RANGES)
    states = gpd.read_file(STATES)
    if len(frame) != 34_566 or frame["city_key"].nunique() != 50:
        raise RuntimeError("Fig. 4 input frame mismatch")

    fig = plt.figure(figsize=(7.20, 5.55), facecolor="white")
    panel_heading(fig, 0.055, 0.930, "a", "Continuous canopy-deficit and heat contexts across social-condition tertiles")
    _, density_source, _ = density_panels(fig, frame)
    panel_heading(fig, 0.055, 0.455, "b", "Spatial conditioning attenuates enrichment")
    _, benchmark_source = benchmark_panel(fig, point, ranges)
    panel_heading(fig, 0.480, 0.455, "c", "Joint-context prevalence differs among cities")
    _, city_source = map_panel(fig, frame, states)

    outputs = {
        "png": FIGURES / "Fig4_joint_context_spatial_audit.png",
        "pdf": FIGURES / "Fig4_joint_context_spatial_audit.pdf",
        "svg": FIGURES / "Fig4_joint_context_spatial_audit.svg",
        "tif": FIGURES / "Fig4_joint_context_spatial_audit.tif",
    }
    qa = export_and_qa(fig, outputs)
    plt.close(fig)

    density_source.to_csv(SOURCE / "Fig4a_G_H_density_bins_by_V_tertile.csv", index=False)
    benchmark_source.to_csv(SOURCE / "Fig4b_spatial_conditioning_trajectory.csv", index=False)
    city_source.to_csv(SOURCE / "Fig4c_city_joint_context_prevalence.csv", index=False)
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)

    joint = frame["context_id"].eq("A")
    main_row = benchmark_source.loc[
        benchmark_source["benchmark"].eq("main_block_conditioned")
    ].iloc[0]
    small_row = benchmark_source.loc[
        benchmark_source["benchmark"].eq("small_block_conditioned")
    ].iloc[0]
    qa.update(
        {
            "status": "PASS",
            "input_rows": len(frame),
            "cities": frame["city_key"].nunique(),
            "joint_context_block_groups": int(joint.sum()),
            "joint_context_population": float(frame.loc[joint, "population"].sum()),
            "panel_a_x_axis_capped_at_p99_pp": float(frame["G"].quantile(0.99)),
            "panel_a_y_axis_capped_at_p99_C": float(frame["H"].quantile(0.99)),
            "panel_a_overflow_values_accumulated_in_edge_bins": True,
            "main_block_ratio": float(main_row["observed_expected_ratio"]),
            "small_block_ratio": float(small_row["observed_expected_ratio"]),
            "uncertainty": "two-level city and intact spatial-block bootstrap; sampling stability, not a spatial-null p value",
            "scientific_claim": "Joint context is modestly enriched after spatial conditioning, while city prevalence is heterogeneous.",
            "quality_gate_status": "PASS",
            "remaining_limitations": [
                "Threshold-defined contexts are nominal and locally scale-sensitive.",
                "Block-conditioned ratios are descriptive benchmarks rather than exact spatial-randomization tests.",
                "The highest 1% of G and H values are accumulated in edge bins in panel a and remain available in source data.",
            ],
        }
    )
    if qa["clipped_text"] or not qa["exports_exist"]:
        qa["status"] = "FAIL"
        qa["quality_gate_status"] = "FAIL"
    write_json(QA / "Fig4_figure_qa.json", qa)
    caption = (
        "Fig. 4 | Canopy deficit, surface-heat context and social conditions overlap only partly across urban neighbourhoods. "
        "a, Joint distribution of the density-stratified canopy deficit (G) and positive summer Landsat land-surface-temperature anomaly (H) across tertiles of the prespecified social-condition ordering (V) for 34,566 block groups in 50 cities. Dashed lines mark the positive-value medians used to separate G2/G3 and H2/H3; the outlined region in V3 identifies the joint G-positive, H3, V3 context. Values above the 99th percentile are retained in the outer density bins. "
        "b, Equal-city observed-to-expected prevalence ratio for the joint context under the within-city marginal benchmark and benchmarks conditioned on large, primary and small spatial blocks. Points are central ratios and shading shows 95% two-level bootstrap intervals obtained by resampling cities and intact within-city blocks. The intervals describe sampling stability and are not tests against a spatial-randomization null. "
        "c, Percentage of analysed block groups assigned to the joint context in each city. The 3,602 joint-context block groups represented 4.59 million residents, but local labels were threshold- and scale-dependent."
    )
    (ROOT / "text" / "Fig4_caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
