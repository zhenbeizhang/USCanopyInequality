from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageOps


PROJECT = analysis_store_root()
R3 = PROJECT / "Result" / "Result3"
SUPPORT = R3 / "Result3_support_aligned"
AUDIT = R3 / "versions" / "nonlinearity_role_audit"
OUT = R3 / "versions" / "main_figure"

COLORS = {
    "black": "#202124",
    "dark": "#596168",
    "grey": "#9DA5AA",
    "light": "#D7DDE0",
    "pale": "#EEF2F2",
    "blue": "#4F7892",
    "steel": "#6F8792",
    "teal": "#247A70",
    "orange": "#B95F45",
}
MODEL_ORDER = [
    "M1_common_FE_density",
    "M2_common_FE_environment",
    "M2_city_specific_REML",
    "M3_spatial_GAM_REML",
]
MODEL_LABELS = {
    "M1_common_FE_density": "Density-adjusted FE",
    "M2_common_FE_environment": "Environment-adjusted",
    "M2_city_specific_REML": "City-specific slopes",
    "M3_spatial_GAM_REML": "Spatial smooth",
}


def ensure_dirs() -> None:
    for name in ("figures", "source_data", "qa", "methods", "text"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.7,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "text.color": COLORS["black"],
            "axes.labelcolor": COLORS["black"],
            "xtick.color": COLORS["black"],
            "ytick.color": COLORS["black"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_header(fig: plt.Figure, letter: str, title: str, x: float, y: float) -> None:
    fig.text(x, y, letter, fontsize=9.3, fontweight="bold", ha="left", va="baseline")
    fig.text(x + 0.024, y, title, fontsize=8.3, fontweight="bold", ha="left", va="baseline")


def load_data() -> dict[str, object]:
    old_source = SUPPORT / "figure" / "source_data"
    city_boot = pd.read_csv(old_source / "Fig3a_city_spatial_bootstrap_intervals.csv")
    analytic = pd.read_csv(SUPPORT / "tables" / "city_canopy_lst_coefficients_hierarchical_models.csv")
    analytic = analytic.loc[analytic["model_id"].eq("M3_spatial_GAM_REML"), ["city_key", "coefficient_C_per_10pp_canopy"]]
    city = city_boot.merge(analytic, on="city_key", validate="one_to_one")
    city["point_estimate"] = city["coefficient_C_per_10pp_canopy"]
    city["city_label"] = city["city_display"].replace(
        {
            "Louisville-Jefferson County": "Louisville-Jefferson",
            "Louisville Jefferson County": "Louisville-Jefferson",
        }
    )
    city["plot_class"] = np.select(
        [
            city["all_scales_ci_below_zero"].astype(bool),
            city["partially_pooled_bootstrap_ci_low"].astype(float).gt(0),
        ],
        ["negative_all_scales", "positive_primary"],
        default="scale_sensitive",
    )
    city = city.sort_values("point_estimate").reset_index(drop=True)

    models = pd.read_csv(SUPPORT / "tables" / "national_canopy_lst_model_comparison.csv")
    models = models.loc[models["model_id"].isin(MODEL_ORDER)].copy()
    boot = pd.read_csv(
        SUPPORT / "paired_m2_m3_shared_block_bootstrap" / "tables" / "paired_national_spatial_bootstrap_inference.csv"
    )
    boot = boot.loc[boot["block_scale"].eq("main")].set_index("model_id")
    models["point_estimate"] = models["coefficient_C_per_10pp_canopy"]
    models["plot_ci_low"] = models["ci_low"]
    models["plot_ci_high"] = models["ci_high"]
    models["interval_type"] = "95% city-clustered CI"
    for model_id in ("M2_city_specific_REML", "M3_spatial_GAM_REML"):
        mask = models["model_id"].eq(model_id)
        models.loc[mask, "plot_ci_low"] = float(boot.loc[model_id, "two_level_ci_low"])
        models.loc[mask, "plot_ci_high"] = float(boot.loc[model_id, "two_level_ci_high"])
        models.loc[mask, "interval_type"] = "95% two-level spatial bootstrap interval"
    models["display_label"] = models["model_id"].map(MODEL_LABELS)
    models["order"] = models["model_id"].map({value: index for index, value in enumerate(MODEL_ORDER)})
    models = models.sort_values("order")

    roles = pd.read_csv(AUDIT / "tables" / "result3_m2_m3_complementary_roles.csv")
    diagnostics = models[["model_id", "display_label", "order"]].merge(
        roles[
            [
                "model_id",
                "weighted_rmse_C",
                "pooled_block_diagonal_moran_i_8nn",
            ]
        ],
        on="model_id",
        how="left",
    )
    # Add M1 and common-M2 diagnostics from the same frozen tables.
    cv = pd.read_csv(SUPPORT / "tables" / "nested_spatial_cv_outer_performance.csv")
    moran = pd.read_csv(SUPPORT / "tables" / "residual_moran_summary.csv")
    all_diagnostics = cv.merge(moran, on="model_id", validate="one_to_one")
    diagnostics = diagnostics.drop(columns=["weighted_rmse_C", "pooled_block_diagonal_moran_i_8nn"]).merge(
        all_diagnostics[["model_id", "weighted_rmse_C", "pooled_block_diagonal_moran_i_8nn"]],
        on="model_id",
        validate="one_to_one",
    )

    contrast = pd.read_csv(old_source / "Fig3d_city_canopy_quintile_contrasts.csv")
    contrast_summary = json.loads(
        (old_source / "Fig3d_city_canopy_quintile_contrast_summary.json").read_text(encoding="utf-8")
    )
    nonlinearity = json.loads(
        (AUDIT / "qa" / "result3_nonlinearity_role_audit.json").read_text(encoding="utf-8")
    )
    return {
        "city": city,
        "models": models,
        "diagnostics": diagnostics,
        "contrast": contrast,
        "contrast_summary": contrast_summary,
        "nonlinearity": nonlinearity,
    }


def draw_city_forest(ax: plt.Axes, data: pd.DataFrame) -> None:
    visual = {
        "negative_all_scales": (COLORS["teal"], "o", 1.0),
        "scale_sensitive": (COLORS["grey"], "o", 0.72),
        "positive_primary": (COLORS["orange"], "^", 1.0),
    }
    for y, row in data.reset_index(drop=True).iterrows():
        color, marker, alpha = visual[row["plot_class"]]
        ax.hlines(
            y,
            row["partially_pooled_bootstrap_ci_low"],
            row["partially_pooled_bootstrap_ci_high"],
            color=color,
            lw=1.05,
            alpha=alpha,
            zorder=1,
        )
        ax.scatter(
            row["point_estimate"],
            y,
            s=17 if marker == "o" else 22,
            marker=marker,
            color=color,
            edgecolor="white",
            lw=0.4,
            alpha=alpha,
            zorder=2,
        )
    ax.axvline(0, color=COLORS["black"], lw=0.75)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["city_label"], fontsize=6.55)
    ax.invert_yaxis()
    ax.set_xlim(-2.75, 0.85)
    ax.set_xticks([-2.5, -1.5, -0.5, 0.5])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, pad=2)
    ax.spines["left"].set_visible(False)


def draw_model_estimates(ax: plt.Axes, data: pd.DataFrame, nonlinear: dict[str, object]) -> None:
    colors = [COLORS["dark"], COLORS["blue"], COLORS["steel"], COLORS["teal"]]
    y = np.arange(len(data))[::-1]
    for yi, (_, row), color in zip(y, data.iterrows(), colors, strict=True):
        ax.hlines(yi, row["plot_ci_low"], row["plot_ci_high"], color=color, lw=2.1, alpha=0.82)
        ax.scatter(row["point_estimate"], yi, s=30, color=color, edgecolor="white", lw=0.45, zorder=3)
        ax.text(row["plot_ci_high"] + 0.055, yi, f"{row['point_estimate']:.2f}", va="center", fontsize=6.6)
    ax.axvline(0, color=COLORS["black"], lw=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(data["display_label"], fontsize=6.55)
    ax.set_xlim(-1.95, 0.18)
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0])
    ax.set_xlabel("Adjusted LST difference (°C per +10 pp canopy)", labelpad=3)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)


def draw_diagnostic_tradeoff(ax: plt.Axes, data: pd.DataFrame) -> None:
    data = data.sort_values("order")
    colors = [COLORS["dark"], COLORS["blue"], COLORS["steel"], COLORS["teal"]]
    ax.plot(
        data["weighted_rmse_C"],
        data["pooled_block_diagonal_moran_i_8nn"],
        color=COLORS["light"],
        lw=1.0,
        zorder=1,
    )
    for (_, row), color in zip(data.iterrows(), colors, strict=True):
        ax.scatter(
            row["weighted_rmse_C"],
            row["pooled_block_diagonal_moran_i_8nn"],
            s=34,
            color=color,
            edgecolor="white",
            lw=0.5,
            zorder=3,
        )
    offsets = {
        "M1_common_FE_density": (-5, 8, "right"),
        "M2_common_FE_environment": (5, 5, "left"),
        "M2_city_specific_REML": (6, 7, "left"),
        "M3_spatial_GAM_REML": (6, -9, "left"),
    }
    short = {
        "M1_common_FE_density": "Density FE",
        "M2_common_FE_environment": "Environment",
        "M2_city_specific_REML": "City slopes",
        "M3_spatial_GAM_REML": "Spatial smooth",
    }
    for _, row in data.iterrows():
        dx, dy, align = offsets[row["model_id"]]
        ax.annotate(
            short[row["model_id"]],
            (row["weighted_rmse_C"], row["pooled_block_diagonal_moran_i_8nn"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=6.1,
        )
    ax.set_xlim(1.40, 2.23)
    ax.set_ylim(0.30, 0.66)
    ax.set_xticks([1.5, 1.8, 2.1])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6])
    ax.set_xlabel("Held-block RMSE (°C)", labelpad=3)
    ax.set_ylabel("Residual Moran's I", labelpad=3)
    ax.annotate(
        "lower on both axes",
        xy=(1.43, 0.315),
        xytext=(1.63, 0.365),
        arrowprops={"arrowstyle": "->", "lw": 0.65, "color": COLORS["grey"]},
        fontsize=5.8,
        color=COLORS["dark"],
        ha="left",
    )


def draw_quintiles(ax: plt.Axes, data: pd.DataFrame, summary: dict[str, float]) -> None:
    low = data["lowest_canopy_quintile_heat_anomaly_C"].to_numpy(float)
    high = data["highest_canopy_quintile_heat_anomaly_C"].to_numpy(float)
    differences = low - high
    for low_value, high_value, difference in zip(low, high, differences, strict=True):
        color = COLORS["orange"] if difference <= 0 else COLORS["grey"]
        alpha = 0.70 if difference <= 0 else 0.34
        ax.plot([0, 1], [low_value, high_value], color=color, lw=0.65, alpha=alpha, zorder=1)
        ax.scatter([0, 1], [low_value, high_value], s=6, color=color, alpha=alpha, lw=0, zorder=2)
    mean_low = float(np.mean(low))
    mean_high = float(np.mean(high))
    ax.plot([0, 1], [mean_low, mean_high], color=COLORS["teal"], lw=2.35, zorder=4)
    ax.scatter([0, 1], [mean_low, mean_high], s=34, color=COLORS["teal"], edgecolor="white", lw=0.5, zorder=5)
    ax.axhline(0, color=COLORS["black"], lw=0.65)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(min(-3.5, float(np.nanmin(np.r_[low, high])) - 0.4), max(6.0, float(np.nanmax(np.r_[low, high])) + 0.4))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Lowest\ncanopy", "Highest\ncanopy"], fontsize=7.0)
    ax.set_ylabel("Population-weighted LST anomaly (°C)", labelpad=3)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.text(
        0.03,
        0.97,
        f"{summary['cities_with_positive_contrast']} of {summary['cities']} cities\n"
        f"Mean difference {summary['mean_contrast_C']:.2f} °C\n"
        f"95% city bootstrap {summary['city_bootstrap_ci_low_C']:.2f} to {summary['city_bootstrap_ci_high_C']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.05,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
    )


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tif"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})


def qa(fig: plt.Figure, png: Path) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer=renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            outside.append(item.get_text())
    with Image.open(png) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb)
        nonwhite = np.any(array < 248, axis=2)
        rows, columns = np.where(nonwhite)
        margins = {
            "left_fraction": float(columns.min() / array.shape[1]),
            "right_fraction": float((array.shape[1] - 1 - columns.max()) / array.shape[1]),
            "top_fraction": float(rows.min() / array.shape[0]),
            "bottom_fraction": float((array.shape[0] - 1 - rows.max()) / array.shape[0]),
        }
        ImageOps.grayscale(rgb).save(OUT / "qa" / "Fig3_grayscale.png")
        size = rgb.size
    return {
        "status": "PASS" if not outside and max(margins.values()) < 0.065 else "REVIEW",
        "outside_text": outside,
        "margins": margins,
        "pixel_size": size,
        "estimated_ppi_at_183mm": size[0] / 7.2047,
    }


def main() -> None:
    ensure_dirs()
    style()
    source = load_data()
    city = source["city"]

    fig = plt.figure(figsize=(7.20, 6.86))
    ax_a1 = fig.add_axes([0.145, 0.500, 0.335, 0.395])
    ax_a2 = fig.add_axes([0.650, 0.500, 0.335, 0.395])
    draw_city_forest(ax_a1, city.iloc[:25])
    draw_city_forest(ax_a2, city.iloc[25:])
    fig.text(
        0.565,
        0.466,
        "Partially pooled canopy–LST association (°C per +10 pp canopy)",
        ha="center",
        va="center",
        fontsize=8.0,
    )
    negative = int(city["all_scales_ci_below_zero"].sum())
    panel_header(fig, "a", f"City associations vary across spatial-block widths", 0.040, 0.962)
    handles = [
        Line2D([0], [0], marker="o", color=COLORS["teal"], lw=1.0, markersize=4.3, label="95% interval < 0 at all scales"),
        Line2D([0], [0], marker="o", color=COLORS["grey"], lw=1.0, markersize=4.3, label="scale-sensitive"),
        Line2D([0], [0], marker="^", color=COLORS["orange"], lw=1.0, markersize=4.6, label="95% interval > 0 at primary scale"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.989, 0.938),
        ncol=3,
        fontsize=6.05,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    ax_b = fig.add_axes([0.145, 0.085, 0.275, 0.292])
    draw_model_estimates(ax_b, source["models"], source["nonlinearity"])
    panel_header(fig, "b", "National estimates remain negative", 0.040, 0.416)
    fig.text(
        0.145,
        0.392,
        f"Spline audit: P = {source['nonlinearity']['nonlinearity_city_cluster_p']:.3f}; "
        f"held-block ΔRMSE = {source['nonlinearity']['rcs4_minus_linear_rmse_C']:+.3f} °C",
        ha="left",
        va="center",
        fontsize=6.0,
        color=COLORS["dark"],
    )

    ax_c = fig.add_axes([0.525, 0.085, 0.190, 0.292])
    draw_diagnostic_tradeoff(ax_c, source["diagnostics"])
    panel_header(fig, "c", "Model diagnostics diverge", 0.455, 0.416)

    ax_d = fig.add_axes([0.790, 0.085, 0.195, 0.292])
    draw_quintiles(ax_d, source["contrast"], source["contrast_summary"])
    panel_header(fig, "d", "Low-canopy areas are warmer", 0.720, 0.416)

    stem = OUT / "figures" / "Fig3"
    save(fig, stem)
    audit = qa(fig, stem.with_suffix(".png"))
    plt.close(fig)

    source["city"].to_csv(OUT / "source_data" / "Fig3a_city_intervals.csv", index=False, encoding="utf-8-sig")
    source["models"].to_csv(OUT / "source_data" / "Fig3b_model_estimates.csv", index=False, encoding="utf-8-sig")
    source["diagnostics"].to_csv(OUT / "source_data" / "Fig3c_model_diagnostics.csv", index=False, encoding="utf-8-sig")
    source["contrast"].to_csv(OUT / "source_data" / "Fig3d_quintile_contrasts.csv", index=False, encoding="utf-8-sig")
    (OUT / "qa" / "Fig3_qa.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    caption = (
        "Fig. 3 | Lower neighbourhood canopy is associated with higher summer land-surface temperature, with substantial city heterogeneity. "
        "a, Deterministic partially pooled M3 city estimates (points) and 95% paired spatial-block bootstrap intervals (lines) across 34,566 fully contained block groups. "
        "Colours indicate whether intervals remained below zero at all three block widths, changed classification across widths or lay above zero at the primary width. "
        "b, Deterministic national estimates from density-adjusted city fixed effects, additional environmental adjustment, partially pooled city-specific slopes and a city-specific spatial smooth. "
        "Intervals are 95% city-clustered confidence intervals for the common-slope models and 95% two-level spatial-block bootstrap intervals for the partially pooled models. "
        "A four-knot restricted cubic spline did not materially improve held-block prediction relative to the linear M2 summary (city-clustered nonlinearity P=0.092; ΔRMSE=-0.001 °C; Supplementary Fig. 8). "
        "c, Held-block RMSE and in-sample residual Moran's I show complementary model roles: city-specific M2 slopes predicted held blocks more accurately, whereas M3 left less in-sample residual spatial structure. "
        "d, Descriptive population-weighted LST anomalies in the lowest and highest within-city canopy quintiles; thin lines denote cities and the thick line is the equal-city mean. "
        "LST denotes Landsat surface temperature rather than air temperature or personal heat exposure; all associations are cross-sectional."
    )
    (OUT / "text" / "Fig3_caption.txt").write_text(caption, encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "methods" / Path(__file__).name)
    print(json.dumps({"negative_all_scales": negative, "qa": audit}, indent=2))


if __name__ == "__main__":
    main()
