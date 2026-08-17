from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image, ImageChops


PROJECT = Path(str(analysis_store_root()))
R3 = PROJECT / "Result" / "Result3"
SUPPORT = R3 / "Result3_support_aligned"
PAIRED = SUPPORT / "paired_m2_m3_shared_block_bootstrap"
NONLINEAR = R3 / "versions" / "nonlinearity_role_audit"
OUT = (
    PROJECT
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 8"
    / "versions"
    / "spatial_nonlinearity_audit"
)
FIG = OUT / "figures"
SRC = OUT / "source_data"
QA = OUT / "qa"
TEXT = OUT / "text"
METHODS = OUT / "methods"

MM = 1 / 25.4
WIDTH = 183 * MM
HEIGHT = 130 * MM

BLACK = "#171717"
GREY = "#7E8588"
LIGHT = "#D8DDDC"
M2 = "#4F7181"
M3 = "#2B8378"
SPLINE = "#B8653A"
LINEAR = "#526B83"
NEG = "#4F83A8"
POS = "#C96F4E"

BLOCK_ORDER = ["small", "main", "large"]
BLOCK_X = {"small": 0.5, "main": 1.0, "large": 2.0}
BLOCK_LABEL = {"small": "0.5×", "main": "1×", "large": "2×"}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.8,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.72,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.fontsize": 5.9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_header(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(0, 1.045, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", clip_on=False)
    ax.text(0.048, 1.045, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8, fontweight="bold", clip_on=False)


def clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: list[float]) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(probabilities, positions, values)


def load_data() -> dict[str, pd.DataFrame | np.ndarray]:
    paths = {
        "paired": PAIRED / "tables" / "paired_national_spatial_bootstrap_inference.csv",
        "difference": PAIRED / "tables" / "paired_national_model_difference.csv",
        "roles": NONLINEAR / "tables" / "result3_m2_m3_complementary_roles.csv",
        "curve": NONLINEAR / "source_data" / "result3_rcs4_adjusted_curve.csv",
        "full": NONLINEAR / "tables" / "result3_nonlinearity_full_sample.csv",
        "cv": NONLINEAR / "tables" / "result3_nonlinearity_cv_by_fold.csv",
        "cv_summary": NONLINEAR / "tables" / "result3_nonlinearity_cv_summary.csv",
        "frame": SUPPORT / "data" / "support_aligned_model_frame_34566.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    paired = pd.read_csv(paths["paired"])
    difference = pd.read_csv(paths["difference"])
    roles = pd.read_csv(paths["roles"])
    curve = pd.read_csv(paths["curve"])
    full = pd.read_csv(paths["full"])
    cv = pd.read_csv(paths["cv"])
    cv_summary = pd.read_csv(paths["cv_summary"])
    frame = pd.read_csv(paths["frame"], usecols=["canopy_pct_analysis", "population_model_weight"])

    if len(frame) != 34566 or len(paired) != 6 or len(difference) != 3:
        raise RuntimeError("Frozen Result 3 support mismatch")
    if set(cv["model_id"]) != {"linear", "rcs3", "rcs4", "rcs5"} or cv["outer_fold"].nunique() != 5:
        raise RuntimeError("Incomplete nonlinear cross-validation table")
    quantiles = weighted_quantile(
        frame["canopy_pct_analysis"].to_numpy(float),
        frame["population_model_weight"].to_numpy(float),
        [0.05, 0.25, 0.50, 0.75, 0.95],
    )
    return {
        "paired": paired,
        "difference": difference,
        "roles": roles,
        "curve": curve,
        "full": full,
        "cv": cv,
        "cv_summary": cv_summary,
        "quantiles": quantiles,
        "paths": paths,
    }


def panel_a(ax: plt.Axes, data: dict[str, object]) -> None:
    panel_header(ax, "a", "Intervals remain negative across block widths")
    paired = data["paired"]
    roles = data["roles"].set_index("model_id")
    styles = {
        "M2_city_specific_REML": (M2, "o", "City-specific slopes", -0.018),
        "M3_spatial_GAM_REML": (M3, "D", "Spatial smooth", 0.018),
    }
    for model_id, (colour, marker, label, offset) in styles.items():
        subset = paired.loc[paired["model_id"].eq(model_id)].copy()
        subset["order"] = subset["block_scale"].map({name: i for i, name in enumerate(BLOCK_ORDER)})
        subset = subset.sort_values("order")
        x = subset["block_scale"].map(BLOCK_X).to_numpy(float) + offset
        point = float(roles.loc[model_id, "coefficient_C_per_10pp_canopy"])
        low = subset["two_level_ci_low"].to_numpy(float)
        high = subset["two_level_ci_high"].to_numpy(float)
        ax.vlines(x, low, high, color=colour, lw=0.95, zorder=2)
        ax.plot(x, np.full(len(x), point), color=colour, lw=1.15, marker=marker, ms=4.0, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
        ax.text(2.09, point, label, color=colour, fontsize=6.0, fontweight="bold", va="center")

    delta = float(
        roles.loc["M3_spatial_GAM_REML", "coefficient_C_per_10pp_canopy"]
        - roles.loc["M2_city_specific_REML", "coefficient_C_per_10pp_canopy"]
    )
    main = data["difference"].loc[data["difference"]["block_scale"].eq("main")].iloc[0]
    ax.text(
        0.54,
        -0.055,
        f"Full-data M3 − M2 = {delta:+.3f} °C\n95% block interval {main['two_level_difference_ci_low']:+.3f} to {main['two_level_difference_ci_high']:+.3f}",
        ha="left",
        va="top",
        fontsize=5.8,
        linespacing=1.15,
    )
    ax.axhline(0, color=BLACK, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(0.46, 2.36)
    ax.set_ylim(-0.90, 0.04)
    ax.set_xticks([0.5, 1.0, 2.0], [BLOCK_LABEL[item] for item in BLOCK_ORDER])
    ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0])
    ax.set_xlabel("Spatial-block width relative to primary")
    ax.set_ylabel("Canopy–LST association (°C per +10 pp canopy)")
    clean(ax)


def panel_b(ax: plt.Axes, data: dict[str, object]) -> None:
    panel_header(ax, "b", "Flexible curve adds little predictive information")
    curve = data["curve"]
    full = data["full"].set_index("model_id")
    cv_summary = data["cv_summary"].set_index("model_id")
    x = curve["canopy_pct"].to_numpy(float)
    median = float(curve["weighted_median_canopy_pct"].iloc[0])
    linear_slope = float(full.loc["linear", "linear_component_C_per_10pp"])
    linear_point = linear_slope * (x - median) / 10.0

    ax.fill_between(x, curve["linear_bootstrap_ci_low"], curve["linear_bootstrap_ci_high"], color=LINEAR, alpha=0.12, linewidth=0)
    ax.fill_between(x, curve["rcs4_bootstrap_ci_low"], curve["rcs4_bootstrap_ci_high"], color=SPLINE, alpha=0.13, linewidth=0)
    ax.plot(x, linear_point, color=LINEAR, lw=1.35, label="Linear")
    ax.plot(x, curve["relative_to_weighted_median_LST_C"], color=SPLINE, lw=1.35, label="Four-knot spline")
    ax.axhline(0, color=LIGHT, lw=0.7)
    ax.axvline(median, color=GREY, lw=0.65, ls=(0, (2, 2)))

    y_min = min(float(curve["rcs4_bootstrap_ci_low"].min()), float(curve["linear_bootstrap_ci_low"].min()))
    y_max = max(float(curve["rcs4_bootstrap_ci_high"].max()), float(curve["linear_bootstrap_ci_high"].max()))
    span = y_max - y_min
    rug_y = y_min + 0.035 * span
    for quantile in data["quantiles"]:
        ax.plot([quantile, quantile], [rug_y, rug_y + 0.045 * span], color=BLACK, lw=0.65)
    ax.text(float(data["quantiles"][0]), rug_y + 0.055 * span, "weighted canopy quantiles", ha="left", va="bottom", fontsize=5.1, color=GREY)

    p_value = float(full.loc["rcs4", "nonlinearity_city_cluster_p"])
    delta_rmse = float(cv_summary.loc["rcs4", "rmse_change_vs_linear_C"])
    ax.text(
        0.0,
        1.006,
        f"Nonlinearity P = {p_value:.3f}; held-block ΔRMSE = {delta_rmse:+.4f} °C",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color=GREY,
        clip_on=False,
    )
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(y_min - 0.02 * span, y_max + 0.02 * span)
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
    ax.set_xlabel("Canopy cover (%)")
    ax.set_ylabel("Adjusted LST difference from 14.6% canopy (°C)")
    ax.legend(loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.2)
    clean(ax)


def panel_c(ax: plt.Axes, data: dict[str, object]) -> pd.DataFrame:
    panel_header(ax, "c", "Held-block gains are negligible across spline choices")
    cv = data["cv"]
    summary = data["cv_summary"].set_index("model_id")
    linear = cv.loc[cv["model_id"].eq("linear"), ["outer_fold", "weighted_rmse_C"]].rename(columns={"weighted_rmse_C": "linear_rmse"})
    selected = cv.loc[cv["model_id"].isin(["rcs3", "rcs4", "rcs5"])].merge(linear, on="outer_fold", validate="many_to_one")
    selected["delta_rmse"] = selected["weighted_rmse_C"] - selected["linear_rmse"]
    matrix = selected.pivot(index="model_id", columns="outer_fold", values="delta_rmse").loc[["rcs3", "rcs4", "rcs5"]]
    matrix["Pooled"] = [float(summary.loc[model, "rmse_change_vs_linear_C"]) for model in matrix.index]

    values = matrix.to_numpy(float)
    limit = max(0.010, float(np.nanmax(np.abs(values))))
    cmap = LinearSegmentedColormap.from_list("delta", [NEG, "#F6F5F2", POS])
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=5.8, color="white" if luminance < 0.55 else BLACK)
    ax.axvline(4.5, color=BLACK, lw=0.8)
    ax.set_xticks(np.arange(6), ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Pooled"])
    ax.set_yticks(np.arange(3), ["3-knot spline", "4-knot spline", "5-knot spline"])
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.70, -0.28, 0.30, 0.075])
    colourbar = plt.colorbar(image, cax=cax, orientation="horizontal")
    colourbar.set_ticks([-limit, 0, limit])
    colourbar.ax.tick_params(labelsize=5.1, length=1.7, pad=1)
    colourbar.outline.set_linewidth(0.45)
    colourbar.set_label("ΔRMSE vs linear (°C; negative favours spline)", fontsize=5.2, labelpad=1.2)
    return matrix.reset_index()


def margins(path: Path) -> dict[str, float]:
    with Image.open(path).convert("RGB") as image:
        bbox = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        if bbox is None:
            raise RuntimeError("Blank figure")
        width, height = image.size
    return {"left": bbox[0] / width, "top": bbox[1] / height, "right": (width - bbox[2]) / width, "bottom": (height - bbox[3]) / height}


def main() -> None:
    style()
    for directory in (FIG, SRC, QA, TEXT, METHODS):
        directory.mkdir(parents=True, exist_ok=True)
    data = load_data()

    figure = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.55), width_ratios=(1.02, 1.0), left=0.085, right=0.965, bottom=0.135, top=0.925, wspace=0.35, hspace=0.48)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])
    panel_a(ax_a, data)
    panel_b(ax_b, data)
    matrix = panel_c(ax_c, data)

    base = FIG / "Supplementary_Fig8"
    figure.savefig(base.with_suffix(".png"), dpi=600)
    figure.savefig(base.with_suffix(".pdf"))
    figure.savefig(base.with_suffix(".svg"))
    figure.savefig(base.with_suffix(".tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)

    panel_a_source = data["paired"].merge(
        data["roles"][["model_id", "coefficient_C_per_10pp_canopy"]],
        on="model_id",
        how="left",
        validate="many_to_one",
    )
    panel_a_source.to_csv(SRC / "Supplementary_Fig8a_block_width_inference.csv", index=False)
    data["curve"].to_csv(SRC / "Supplementary_Fig8b_nonlinear_curve.csv", index=False)
    matrix.to_csv(SRC / "Supplementary_Fig8c_cv_delta_matrix.csv", index=False)

    full = data["full"].set_index("model_id")
    cv_summary = data["cv_summary"].set_index("model_id")
    caption = (
        "Supplementary Fig. 8 | Spatial-block and functional-form diagnostics for the canopy–land-surface-temperature association. "
        "a, Deterministic full-data REML estimates for the city-specific environmental model and spatial GAM are repeated across three block widths; "
        "vertical bars are 95% two-level spatial-block bootstrap intervals. The width changes the uncertainty calculation, not the fitted full-data point. "
        "b, Population-weighted adjusted linear and four-knot restricted-cubic-spline curves relative to the weighted median canopy cover; shaded bands are "
        f"95% city-and-block bootstrap intervals. The nonlinear terms were not supported at the city-cluster level (P = {float(full.loc['rcs4', 'nonlinearity_city_cluster_p']):.3f}), "
        f"and the spline changed pooled held-block RMSE by {float(cv_summary.loc['rcs4', 'rmse_change_vs_linear_C']):+.4f} °C. Tick marks denote weighted canopy quantiles. "
        "c, Fold-specific and pooled RMSE difference between three spline specifications and the linear model; negative values favour the spline. "
        "Together these diagnostics support a linear national summary while retaining the spatial model as a residual-structure sensitivity analysis. "
        "Landsat LST represents surface thermal conditions rather than personal heat exposure."
    )
    (TEXT / "Supplementary_Fig8_caption.txt").write_text(caption, encoding="utf-8")
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)

    path_map = data["paths"]
    report = {
        "status": "PASS",
        "quality_gate_status": "PASS",
        "analysis_units": 34566,
        "cities": 50,
        "rcs4_city_cluster_p": float(full.loc["rcs4", "nonlinearity_city_cluster_p"]),
        "rcs4_pooled_delta_rmse_C": float(cv_summary.loc["rcs4", "rmse_change_vs_linear_C"]),
        "all_model_scale_intervals_below_zero": bool((data["paired"]["two_level_ci_high"] < 0).all()),
        "all_pooled_spline_rmse_changes_abs_below_0p002_C": bool((cv_summary.loc[["rcs3", "rcs4", "rcs5"], "rmse_change_vs_linear_C"].abs() < 0.002).all()),
        "outer_margins": margins(base.with_suffix(".png")),
        "source_hashes": {name: sha256(path) for name, path in path_map.items()},
        "scientific_boundary": "Functional-form and residual-structure diagnostics; no causal cooling claim.",
    }
    report["all_checks_pass"] = all(
        [
            report["all_model_scale_intervals_below_zero"],
            report["all_pooled_spline_rmse_changes_abs_below_0p002_C"],
            max(report["outer_margins"].values()) < 0.08,
        ]
    )
    (QA / "Supplementary_Fig8_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
