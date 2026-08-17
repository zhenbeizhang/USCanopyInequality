from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = analysis_store_root()
R4 = PROJECT / "Result" / "Result4"
LEGACY = R4 / "versions" / "joint_context_boundary_spatial_audit"
LOO = R4 / "versions" / "leave_one_unit_out_benchmark"
OUT = R4 / "versions" / "main_figure"
SCRIPT = Path(__file__).with_name("figure4_style_helpers.py")


def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_fig4", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_legacy()


def ensure_dirs() -> None:
    for name in ("figures", "source_data", "qa", "methods", "text"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def benchmark_panel(
    fig: mpl.figure.Figure,
    point: pd.DataFrame,
    ranges: pd.DataFrame,
) -> tuple[mpl.axes.Axes, pd.DataFrame]:
    ax = fig.add_axes((0.075, 0.115, 0.320, 0.305))
    median_sizes = {
        scale: float(ranges[f"{scale}_block_size_km"].median())
        for scale in ("large", "main", "small")
    }
    order = (
        "city_marginal_leave_one_unit_out",
        "large_block_leave_one_unit_out",
        "main_block_leave_one_unit_out",
        "small_block_leave_one_unit_out",
    )
    data = (
        point.loc[
            point["event"].eq("joint_context")
            & point["benchmark"].isin(order)
            & point["grid_origin"].astype(str).isin(["not_applicable", "origin_mean"])
        ]
        .set_index("benchmark")
        .loc[list(order)]
        .reset_index()
    )
    labels = (
        "City marginal\nLOO",
        f"Large blocks\nmedian {median_sizes['large']:.1f} km",
        f"Primary blocks\nmedian {median_sizes['main']:.1f} km",
        f"Small blocks\nmedian {median_sizes['small']:.1f} km",
    )
    x = np.arange(len(data))
    y = data["observed_expected_ratio"].to_numpy(float)
    low = data["bootstrap_p025"].to_numpy(float)
    high = data["bootstrap_p975"].to_numpy(float)
    ax.axhline(1, color=legacy.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.fill_between(x, low, high, color="#DDCFE1", alpha=0.70, zorder=2)
    ax.plot(
        x,
        y,
        color=legacy.PURPLE_DARK,
        lw=1.35,
        marker="o",
        ms=4.2,
        mfc="white",
        mew=1.0,
        zorder=3,
    )
    for index, value in enumerate(y):
        ax.text(index, high[index] + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Observed / expected prevalence")
    ax.set_ylim(0.96, max(high) + 0.13)
    ax.set_xlim(-0.25, 3.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    return ax, data


def main() -> None:
    ensure_dirs()
    legacy.setup_style()
    mpl.rcParams.update(
        {
            "font.size": 6.9,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.15,
            "ytick.labelsize": 6.15,
        }
    )
    legacy.FIGURES = OUT / "figures"
    legacy.SOURCE = OUT / "source_data"
    legacy.QA = OUT / "qa"
    legacy.METHODS = OUT / "methods"

    frame = pd.read_csv(LEGACY / "data" / "fully_contained_primary_GHV27.csv.gz", dtype={"GEOID_str": str}, low_memory=False)
    point = pd.read_csv(LOO / "tables" / "result4_leave_one_unit_out_benchmarks.csv")
    ranges = pd.read_csv(legacy.RANGES)
    states = gpd.read_file(legacy.STATES)
    if len(frame) != 34_566 or frame["city_key"].nunique() != 50:
        raise RuntimeError("Fig. 4 input frame mismatch")

    fig = plt.figure(figsize=(7.20, 5.55), facecolor="white")
    legacy.panel_heading(
        fig,
        0.055,
        0.930,
        "a",
        "Continuous canopy-deficit and heat contexts across social-condition tertiles",
    )
    _, density_source, _ = legacy.density_panels(fig, frame)
    legacy.panel_heading(
        fig,
        0.055,
        0.455,
        "b",
        "Spatial conditioning attenuates joint-context enrichment",
    )
    _, benchmark_source = benchmark_panel(fig, point, ranges)
    legacy.panel_heading(
        fig,
        0.480,
        0.455,
        "c",
        "Joint-context prevalence differs among cities",
    )
    _, city_source = legacy.map_panel(fig, frame, states)

    outputs = {
        "png": OUT / "figures" / "Fig4.png",
        "pdf": OUT / "figures" / "Fig4.pdf",
        "svg": OUT / "figures" / "Fig4.svg",
        "tif": OUT / "figures" / "Fig4.tif",
    }
    qa = legacy.export_and_qa(fig, outputs)
    plt.close(fig)

    density_source.to_csv(OUT / "source_data" / "Fig4a_G_H_density_bins_by_V_tertile.csv", index=False, encoding="utf-8-sig")
    benchmark_source.to_csv(OUT / "source_data" / "Fig4b_leave_one_unit_out_trajectory.csv", index=False, encoding="utf-8-sig")
    city_source.to_csv(OUT / "source_data" / "Fig4c_city_joint_context_prevalence.csv", index=False, encoding="utf-8-sig")

    main_row = benchmark_source.loc[benchmark_source["benchmark"].eq("main_block_leave_one_unit_out")].iloc[0]
    small_row = benchmark_source.loc[benchmark_source["benchmark"].eq("small_block_leave_one_unit_out")].iloc[0]
    qa.update(
        {
            "status": "PASS",
            "input_rows": len(frame),
            "cities": int(frame["city_key"].nunique()),
            "joint_context_block_groups": int(frame["context_id"].eq("A").sum()),
            "main_block_leave_one_unit_out_ratio": float(main_row["observed_expected_ratio"]),
            "main_block_leave_one_unit_out_interval": [float(main_row["bootstrap_p025"]), float(main_row["bootstrap_p975"])],
            "main_block_supported_fraction": float(main_row["equal_city_supported_unit_fraction"]),
            "small_block_leave_one_unit_out_ratio": float(small_row["observed_expected_ratio"]),
            "uncertainty": "2,000 two-level city and intact spatial-block bootstrap draws",
            "scientific_claim": "Joint context remains modestly enriched after leave-one-unit-out spatial conditioning; the benchmark is descriptive rather than a spatial-null test.",
            "quality_gate_status": "PASS",
        }
    )
    if qa["clipped_text"] or not qa["exports_exist"]:
        qa["status"] = "FAIL"
        qa["quality_gate_status"] = "FAIL"
    (OUT / "qa" / "Fig4_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    caption = (
        "Fig. 4 | Canopy deficit, surface-heat context and social conditions overlap only partly across urban neighbourhoods. "
        "a, Joint distribution of the density-stratified canopy deficit (G) and positive summer Landsat land-surface-temperature anomaly (H) across tertiles of the prespecified social-condition score (V) for 34,566 block groups in 50 cities. "
        "Dashed lines mark the positive-value medians used to separate G2/G3 and H2/H3; the outlined region in V3 identifies the joint G-positive, H3, V3 context. Values above the 99th percentile are retained in the outer density bins. "
        "b, Equal-city observed-to-expected prevalence ratios under leave-one-unit-out (LOO) city-marginal and spatial-block-conditioned benchmarks. For each analysed block group, marginal probabilities were estimated from other units in the same city or block. Spatial benchmarks exclude singleton blocks and retain 99.0%, 94.1% and 68.8% of units at the large, primary and small widths, respectively. "
        "Points are deterministic ratios and shading shows 95% intervals from 2,000 city and intact-block bootstrap draws. Block widths are medians of city-specific residual-range widths. These intervals describe sampling stability and are not tests against a generative spatial null. "
        "c, Percentage of analysed block groups assigned to the joint context in each city. The 3,602 joint-context block groups represented 4.59 million residents, but membership remained threshold- and scale-dependent."
    )
    (OUT / "text" / "Fig4_caption.txt").write_text(caption, encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "methods" / Path(__file__).name)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
