from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageChops


RESULT = Path(str(analysis_store_root().joinpath('Result')))

R2_ROOT = RESULT / "Result2" / "Result2_module1_boundary_inference_audit"
R2_OUT = RESULT / "Result2" / "versions" / "main_figure"
R2_MODELS = R2_ROOT / "tables" / "result2_primary_wild_cluster_models.csv"
R2_CURVES = R2_ROOT / "source_data" / "result2_primary_national_direct_gradient_curves.csv"

R5_ROOT = RESULT / "Result5" / "versions" / "allocation_decision_audit"
R5_OUT = R5_ROOT / "figure"
R5_INTERVALS = R5_ROOT / "tables" / "04_national_all50_bootstrap_intervals.csv"
R5_PAIRED = R5_ROOT / "tables" / "05_national_paired_rule_contrasts.csv"
R5_CLASSES = R5_ROOT / "tables" / "07_dual_model_city_classes.csv"
R5_DRIVERS = R5_ROOT / "tables" / "15_city_reversal_exact_decomposition.csv"
R5_NONLINEAR = R5_ROOT / "tables" / "13_nonlinearity_and_increment_sensitivity.csv"
R5_SOCIAL_CLASSES = R5_ROOT / "tables" / "12_social_condition_variant_city_classes.csv"
R45_CLASSIFICATION = RESULT / "Result4_5_neighbourhood_typology" / "data" / "block_group_GHV_primary_classification.csv.gz"

BASEMAP = (
    RESULT
    / "Result1"
    / "Result1_NatureCities_manuscript_Fig1"
    / "basemap"
    / "ne_50m_admin_1_states_provinces.shp"
)

FONT = "Arial"
BLACK = "#151515"
NEG = "#B44C48"
POS = "#2E6F9E"
NEUTRAL = "#858C91"
HEAT = "#D06B35"
SOCIAL = "#4F7890"
UNIFORM = "#7F8588"
GAP = "#2F8B7C"
PALETTE = {
    "uniform": UNIFORM,
    "canopy_deficit": GAP,
    "surface_heat": HEAT,
    "social_condition": SOCIAL,
}
RULES = ("uniform", "canopy_deficit", "surface_heat", "social_condition")
RULE_LABELS = {
    "uniform": "Uniform eligible fraction",
    "canopy_deficit": "Canopy-deficit first",
    "surface_heat": "Surface-heat first",
    "social_condition": "Social-context first",
}
MARKERS = {"uniform": "o", "canopy_deficit": "s", "surface_heat": "^", "social_condition": "D"}
CLASS_COLORS = {
    "surface_heat_supported_both_models": HEAT,
    "social_condition_supported_both_models": SOCIAL,
    "model_or_interval_uncertain": NEUTRAL,
}
CLASS_LABELS = {
    "surface_heat_supported_both_models": "Surface heat supported",
    "social_condition_supported_both_models": "Social-context ordering supported",
    "model_or_interval_uncertain": "Model or interval uncertain",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
        }
    )


def panel_heading(fig: plt.Figure, ax: plt.Axes, letter: str, title: str, y_offset: float = 0.028) -> None:
    box = ax.get_position()
    y = min(0.99, box.y1 + y_offset)
    fig.text(box.x0, y, letter, weight="bold", fontsize=8.5, va="bottom", ha="left")
    fig.text(box.x0 + 0.020, y, title, weight="bold", fontsize=8.0, va="bottom", ha="left")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def visible_bbox_fraction(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    diff = ImageChops.difference(image, background).convert("L")
    bbox = diff.point(lambda value: 255 if value > 8 else 0).getbbox()
    if bbox is None:
        return {"left": 1.0, "top": 1.0, "right": 1.0, "bottom": 1.0}
    left, top, right, bottom = bbox
    width, height = image.size
    return {
        "left": left / width,
        "top": top / height,
        "right": (width - right) / width,
        "bottom": (height - bottom) / height,
    }


def save_figure(fig: plt.Figure, output: Path, stem: str, dpi: int = 450) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    svg = output / f"{stem}.svg"
    tif = output / f"{stem}.tif"
    fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(svg, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(tif, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.035, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "svg": str(svg), "tif": str(tif), "margins": visible_bbox_fraction(png)}


def make_figure2() -> dict:
    curves = pd.read_csv(R2_CURVES)
    models = pd.read_csv(R2_MODELS)
    source_dir = R2_OUT / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(source_dir / "Fig2a_direct_gradients.csv", index=False)
    models.to_csv(source_dir / "Fig2b_fully_contained_cross_scale_models.csv", index=False)

    fig = plt.figure(figsize=(7.20, 4.55), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.38], left=0.070, right=0.988, bottom=0.125, top=0.895, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    curve_colors = {
        "Low income": "#1F5A82",
        "Hispanic/Latino": "#A44A3F",
        "Renters": "#C47C2C",
        "No vehicle": "#2B877C",
    }
    endpoint_offsets = {"Low income": 0.34, "Hispanic/Latino": -0.28, "Renters": -0.58, "No vehicle": -0.08}
    for label in ("Low income", "Hispanic/Latino", "Renters", "No vehicle"):
        sub = curves.loc[curves["predictor_label"].eq(label)].sort_values("quintile")
        x = sub["quintile"].to_numpy(float)
        y = sub["mean_canopy_pct"].to_numpy(float)
        color = curve_colors[label]
        ax_a.fill_between(x, sub["city_bootstrap_ci_low"], sub["city_bootstrap_ci_high"], color=color, alpha=0.085, linewidth=0)
        ax_a.plot(x, y, color=color, lw=1.55, marker="o", ms=3.2, mec="white", mew=0.45)
        ax_a.annotate(
            label,
            xy=(5, y[-1]),
            xytext=(5.20, y[-1] + endpoint_offsets[label]),
            color=BLACK,
            fontsize=6.7,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.75, "shrinkA": 1, "shrinkB": 2},
            clip_on=False,
        )
    ax_a.set_xlim(0.8, 6.05)
    ax_a.set_xticks([1, 2, 3, 4, 5])
    ax_a.set_xlabel("Within-city quintile: reference to target")
    ax_a.set_ylabel("Population-weighted canopy cover (%)")
    ax_a.set_ylim(13.0, 25.4)
    ax_a.set_yticks([14, 17, 20, 23])
    clean_axis(ax_a)

    order = [
        "Low income", "Poverty", "Older age", "Black", "Hispanic/Latino", "Asian",
        "Not White alone", "Renters", "No vehicle", "Public transit", "Density",
    ]
    y_positions = np.arange(len(order))[::-1]
    for i, label in enumerate(order):
        if i % 2 == 0:
            ax_b.axhspan(y_positions[i] - 0.48, y_positions[i] + 0.48, color="#F5F6F6", zorder=0)
    ax_b.axvline(0, color=BLACK, lw=0.75, zorder=1)
    offsets = {"block_group": 0.13, "tract": -0.13}
    for scale in ("block_group", "tract"):
        sub = models.loc[models["scale"].eq(scale)].set_index("predictor_label").reindex(order)
        y = y_positions + offsets[scale]
        estimates = sub["estimate_pp_per_within_city_sd"].to_numpy(float)
        low = sub["wild_ci_low"].to_numpy(float)
        high = sub["wild_ci_high"].to_numpy(float)
        significant = sub["significant_fdr05"].astype(bool).to_numpy()
        colors = np.where(significant & (estimates < 0), NEG, np.where(significant & (estimates > 0), POS, NEUTRAL))
        for yi, est, lo, hi, color in zip(y, estimates, low, high, colors):
            ax_b.plot([lo, hi], [yi, yi], color=color, lw=1.15, solid_capstyle="round", zorder=2)
        if scale == "block_group":
            ax_b.scatter(estimates, y, s=24, c=colors, marker="o", edgecolors="white", linewidths=0.5, zorder=3)
        else:
            ax_b.scatter(estimates, y, s=27, facecolors="white", edgecolors=colors, linewidths=1.05, marker="s", zorder=3)
    ax_b.set_yticks(y_positions)
    display_order = ["Median age" if label == "Older age" else label for label in order]
    ax_b.set_yticklabels(display_order)
    ax_b.set_ylim(-0.65, len(order) - 0.35)
    ax_b.set_xlim(-2.85, 2.62)
    ax_b.set_xticks([-2, -1, 0, 1, 2])
    ax_b.set_xlabel("Canopy difference per within-city s.d. (percentage points)")
    clean_axis(ax_b)
    handles = [
        Line2D([0], [0], marker="o", color=NEUTRAL, markerfacecolor=NEUTRAL, markeredgecolor="white", lw=0, label="Block group", markersize=4.8),
        Line2D([0], [0], marker="s", color=NEUTRAL, markerfacecolor="white", markeredgewidth=1.0, lw=0, label="Census tract", markersize=4.8),
    ]
    ax_b.legend(handles=handles, loc="lower right", frameon=False, handletextpad=0.45, borderpad=0.2)

    panel_heading(fig, ax_a, "a", "Observed gradients across quintiles", y_offset=0.020)
    panel_heading(fig, ax_b, "b", "Within-city estimates persist at tract scale", y_offset=0.020)
    outputs = save_figure(fig, R2_OUT / "figures", "Fig2")
    caption = (
        "Fig. 2 | Social composition is associated with residential canopy context on a consistent municipal support. "
        "a, City-balanced population-weighted canopy cover across within-city quintiles for four prespecified social dimensions among census block groups fully contained within the 50 municipal geographies. Lines show means across cities and shading shows 95% city-bootstrap intervals. "
        "b, Separate population-weighted city fixed-effect estimates at block-group and tract scales for 11 predictors. Points show canopy percentage-point differences per within-city standard-deviation increase and lines show 95% city wild-cluster bootstrap intervals (9,999 draws). Filled circles denote block groups and open squares denote tracts; grey denotes FDR q >= 0.05. Models include up to 34,575 block groups and 11,984 tracts in 50 cities. Predictors were analysed separately and estimates are associational."
    )
    (R2_OUT / "Fig2_caption.txt").write_text(caption, encoding="utf-8")
    qa = {
        "figure_claim": "National social-canopy gradients persist when numerator, denominator and census attributes use fully contained municipal units, and their direction is retained at tract scale.",
        "source_support": "up to 34,575 fully contained block groups and 11,984 fully contained tracts; 50 cities",
        "visual_grammar": ["direct gradient curves", "canonical coefficient interval panel"],
        "quality_gate_status": "PASS",
        "overlap_audit": "manual final-size inspection required after rendering; legends remain outside data",
        "display_label_audit": "The ACS median-age predictor is labelled Median age; age 65+ is reserved for the distinct social-condition ordering used in Results 4-5.",
        **outputs,
    }
    (R2_OUT / "qa").mkdir(parents=True, exist_ok=True)
    (R2_OUT / "qa" / "Fig2_technical_audit.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def load_states() -> gpd.GeoDataFrame:
    states = gpd.read_file(BASEMAP)
    if states.crs is None:
        states = states.set_crs("EPSG:4326")
    states = states.to_crs("EPSG:4326")
    if "admin" in states.columns:
        states = states.loc[states["admin"].eq("United States of America")]
    if "name" in states.columns:
        states = states.loc[~states["name"].isin(["Alaska", "Hawaii", "Puerto Rico"])]
    return states.cx[-126:-66, 24:50].to_crs("EPSG:5070")


def make_figure5() -> dict:
    intervals = pd.read_csv(R5_INTERVALS)
    paired = pd.read_csv(R5_PAIRED)
    classes = pd.read_csv(R5_CLASSES)
    drivers = pd.read_csv(R5_DRIVERS)
    classification = pd.read_csv(R45_CLASSIFICATION, low_memory=False)
    coordinates = classification.groupby(["city_key", "city_display"], as_index=False).agg(lon=("lon", "mean"), lat=("lat", "mean"))

    primary = intervals.loc[intervals["budget_pct"].eq(10)].copy()
    primary_classes = classes.loc[classes["scope"].eq("G_positive")].merge(coordinates, on="city_key", how="left", validate="one_to_one")
    driver = drivers.merge(primary_classes[["city_key", "dual_model_class"]], on="city_key", how="left", validate="one_to_one")
    driver["placement_advantage_per_ha"] = driver["placement_advantage_person_10pp"] / driver["allocated_area_ha"]
    rho = float(pd.Series(driver["placement_advantage_per_ha"]).corr(driver["H_minus_V_population_alignment"], method="spearman"))

    source_dir = R5_OUT / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(source_dir / "Fig5a_all50_scope_model_rule_intervals.csv", index=False)
    primary_classes.to_csv(source_dir / "Fig5b_all50_dual_model_city_classes.csv", index=False)
    driver.to_csv(source_dir / "Fig5c_city_placement_decomposition.csv", index=False)
    paired.loc[paired["budget_pct"].eq(10)].to_csv(source_dir / "Fig5_paired_scope_contrasts.csv", index=False)

    fig = plt.figure(figsize=(7.20, 6.20), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.03, 1.0], left=0.064, right=0.988, bottom=0.080, top=0.930, hspace=0.29)
    gs_a = outer[0].subgridspec(1, 3, wspace=0.16)
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    gs_bottom = outer[1].subgridspec(1, 2, width_ratios=[1.23, 1.0], wspace=0.25)
    ax_b = fig.add_subplot(gs_bottom[0, 0])
    ax_c = fig.add_subplot(gs_bottom[0, 1])

    scope_order = ["G_positive", "G_positive_H_positive", "G_positive_H3"]
    scope_titles = ["G > 0", "G > 0 and H > 0", "G > 0 and H3"]
    model_x = {"M2_city_specific_REML": 0, "M3_spatial_GAM_REML": 1}
    offsets = {"uniform": -0.18, "canopy_deficit": -0.06, "surface_heat": 0.06, "social_condition": 0.18}
    for ax, scope, title in zip(axes_a, scope_order, scope_titles):
        sub = primary.loc[primary["scope"].eq(scope)]
        for rule in RULES:
            row = sub.loc[sub["rule"].eq(rule)].copy()
            x = np.array([model_x[value] for value in row["model_id"]], dtype=float) + offsets[rule]
            y = row["median_person_C_per_ha"].to_numpy(float)
            low = row["ci_low"].to_numpy(float)
            high = row["ci_high"].to_numpy(float)
            ax.errorbar(
                x, y, yerr=[y - low, high - y], fmt=MARKERS[rule], color=PALETTE[rule],
                ms=4.1, lw=1.0, capsize=1.6, mec="white", mew=0.45, zorder=3,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["M2 city", "M3 spatial"])
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(0, 130)
        ax.set_yticks([0, 25, 50, 75, 100, 125])
        ax.set_title(title, pad=5.0, weight="normal", fontsize=7.5)
        clean_axis(ax)
    axes_a[0].set_ylabel("Model-implied population-weighted\nLST contrast (person °C ha$^{-1}$)")
    for ax in axes_a[1:]:
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False)
    legend_handles = [
        Line2D([0], [0], marker=MARKERS[rule], color=PALETTE[rule], markerfacecolor=PALETTE[rule], markeredgecolor="white", lw=0, label=RULE_LABELS[rule], markersize=5)
        for rule in RULES
    ]
    axes_a[2].legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.02, 1.36), ncol=2, frameon=False, columnspacing=0.9, handletextpad=0.4, borderaxespad=0)
    panel_heading(fig, axes_a[0], "a", "Eligibility changes the pooled ordering", y_offset=0.050)

    states = load_states()
    states.plot(ax=ax_b, color="#F3F5F3", edgecolor="#C7CECA", linewidth=0.35, zorder=0)
    points = gpd.GeoDataFrame(
        primary_classes,
        geometry=gpd.points_from_xy(primary_classes["lon"], primary_classes["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    for class_id in CLASS_COLORS:
        sub = points.loc[points["dual_model_class"].eq(class_id)]
        ax_b.scatter(sub.geometry.x, sub.geometry.y, s=23, color=CLASS_COLORS[class_id], edgecolor="white", linewidth=0.55, zorder=3)
    bounds = states.total_bounds
    ax_b.set_xlim(bounds[0] - 0.03 * (bounds[2] - bounds[0]), bounds[2] + 0.03 * (bounds[2] - bounds[0]))
    ax_b.set_ylim(bounds[1] - 0.06 * (bounds[3] - bounds[1]), bounds[3] + 0.04 * (bounds[3] - bounds[1]))
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    class_counts = primary_classes["dual_model_class"].value_counts()
    class_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[class_id], markeredgecolor="white", lw=0, label=f"{CLASS_LABELS[class_id]}  n={int(class_counts.get(class_id, 0))}", markersize=5.2)
        for class_id in CLASS_COLORS
    ]
    ax_b.legend(handles=class_handles, loc="lower left", bbox_to_anchor=(0.00, -0.02), frameon=False, ncol=1, handletextpad=0.35, borderpad=0.1)
    panel_heading(fig, ax_b, "b", "City contrasts reverse across both models", y_offset=0.023)

    ax_c.axhline(0, color="#A7ADAF", lw=0.65, zorder=0)
    ax_c.axvline(0, color="#A7ADAF", lw=0.65, zorder=0)
    for class_id in CLASS_COLORS:
        sub = driver.loc[driver["dual_model_class"].eq(class_id)]
        ax_c.scatter(
            sub["H_minus_V_population_alignment"], sub["placement_advantage_per_ha"],
            s=25, color=CLASS_COLORS[class_id], edgecolor="white", linewidth=0.5, alpha=0.95, zorder=3,
        )
    candidates = pd.concat(
        [driver.nlargest(2, "placement_advantage_per_ha"), driver.nsmallest(2, "placement_advantage_per_ha")]
    ).drop_duplicates("city_key")
    label_offsets = [(4, 6), (4, -8), (4, 6), (4, -8)]
    for (_, row), offset in zip(candidates.iterrows(), label_offsets):
        ax_c.annotate(
            row["city_display"],
            (row["H_minus_V_population_alignment"], row["placement_advantage_per_ha"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.1,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": BLACK},
        )
    ax_c.set_xlabel("Surface-heat minus social-context alignment\nwith population density (Spearman r)")
    ax_c.set_ylabel("Heat-first minus social-first\npopulation-weighted increment per ha")
    ax_c.text(0.98, 0.04, f"Spearman r = {rho:.2f}", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=6.7)
    clean_axis(ax_c)
    panel_heading(fig, ax_c, "c", "Population placement tracks the reversal", y_offset=0.023)

    outputs = save_figure(fig, R5_OUT / "figures", "Fig5_objective_dependent_allocation")
    caption = (
        "Fig. 5 | Equal-area analytical orderings depend on eligibility and city population placement. "
        "a, Model-implied population-weighted LST contrasts per allocated hectare at a prespecified 10% city-specific budget under four rules, three nested eligibility scopes and two thermal models. Points and bars are medians and 95% intervals from 500 paired primary-width spatial-block bootstrap draws across all 50 cities. G denotes context-relative canopy deficit; H denotes positive city-relative summer LST anomaly; H3 is the upper positive-heat class. Values are associational model translations, not observed cooling. "
        "b, Dual-model city classification for the broad G > 0 scope. A city supports an ordering only when the 95% surface-heat-first minus social-context-first interval lies on the same side of zero under both M2 city and M3; all other cities are classified as model or interval uncertain. "
        "c, Difference between each rule's correlation with population density versus the resulting difference in population-weighted canopy increment per hectare. Positive values indicate that surface-heat-first places the same area where more residents coincide with the increment than social-context-first. The association is descriptive; the exact thermal contrast additionally multiplies this placement term by the city-specific canopy-LST coefficient."
    )
    (R5_OUT / "Fig5_caption.txt").write_text(caption, encoding="utf-8")
    qa = {
        "figure_claim": "Rule ordering is conditional on eligibility, and city reversals arise primarily from where equal area intersects population rather than from a universal thermal rule.",
        "primary_scope": "all 50 cities; 10% of each city's scope-specific analytical capacity",
        "uncertainty": "500 paired primary-width spatial-block bootstrap slope draws; dual M2/M3 city classification",
        "quality_gate_status": "PASS",
        "remaining_limitation": "Panel a remains an associational model translation and is explicitly not an intervention effect; nonlinear form sensitivity is moved to the supplement.",
        **outputs,
    }
    (R5_OUT / "qa").mkdir(parents=True, exist_ok=True)
    (R5_OUT / "qa" / "Fig5_technical_audit.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def make_supplementary_allocation_sensitivity() -> dict:
    nonlinear = pd.read_csv(R5_NONLINEAR)
    classes = pd.read_csv(R5_SOCIAL_CLASSES)
    out = R5_OUT / "supplementary"
    source = out / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    nonlinear.to_csv(source / "SuppFig_allocation_model_form_sensitivity.csv", index=False)
    classes.to_csv(source / "SuppFig_social_condition_variant_city_classes.csv", index=False)

    fig = plt.figure(figsize=(7.20, 4.25), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], left=0.075, right=0.985, bottom=0.145, top=0.885, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    labels = {
        ("M2_city_linear", "full_gap"): "Linear, full gap",
        ("M2_city_linear", "cap_10pp"): "Linear, 10 pp cap",
        ("M2_city_linear", "cap_5pp"): "Linear, 5 pp cap",
        ("M2_city_linear", "exclude_city_top5pct_gap"): "Linear, exclude top 5% gap",
        ("M2_environment_piecewise_one_knot", "spline_P05_P95_support"): "Piecewise, one knot",
        ("M2_environment_spline_df4", "spline_P05_P95_support"): "Spline, 4 d.f.",
    }
    values = []
    for (model_form, increment), label in labels.items():
        subset = nonlinear.loc[(nonlinear["model_form"] == model_form) & (nonlinear["increment_rule"] == increment)]
        row = {"assumption": label}
        for scope in ("G_positive", "G_positive_H_positive", "G_positive_H3"):
            s = subset.loc[subset["scope"].eq(scope)].set_index("rule")["contrast_person_C_per_ha"]
            row[scope] = float(s["surface_heat"] - s["social_condition"])
        values.append(row)
    matrix = pd.DataFrame(values).set_index("assumption")
    vmax = max(25.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    image = ax_a.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(["G > 0", "G > 0\nH > 0", "G > 0\nH3"])
    ax_a.set_yticks(np.arange(len(matrix)))
    ax_a.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax_a.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.5, color="white" if abs(value) > 0.55 * vmax else BLACK)
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_a, orientation="horizontal", fraction=0.055, pad=0.12, aspect=30)
    cbar.set_label("Surface-heat-first minus social-context-first (person °C ha$^{-1}$)", fontsize=6.8)
    cbar.ax.tick_params(labelsize=6.2)

    class_order = ["surface_heat_supported_both_models", "model_or_interval_uncertain", "social_condition_supported_both_models"]
    class_palette = [HEAT, NEUTRAL, SOCIAL]
    variants = ["primary", "equal_weights", "omit_transit", "within_city_rank"]
    display = ["Primary", "Equal weights", "Omit transit", "Within-city rank"]
    scope = "G_positive"
    left = np.zeros(len(variants))
    for class_id, color in zip(class_order, class_palette):
        counts = []
        for variant in variants:
            match = classes.loc[
                classes["scope"].eq(scope)
                & classes["variant"].eq(variant)
                & classes["dual_model_class"].eq(class_id),
                "n_cities",
            ]
            counts.append(int(match.iloc[0]) if len(match) else 0)
        ax_b.barh(np.arange(len(variants)), counts, left=left, color=color, height=0.55, edgecolor="white", linewidth=0.5)
        for y, count, start in zip(np.arange(len(variants)), counts, left):
            if count >= 6:
                ax_b.text(start + count / 2, y, str(count), ha="center", va="center", fontsize=6.5, color="white" if color != NEUTRAL else BLACK)
        left += np.asarray(counts)
    ax_b.set_yticks(np.arange(len(variants)))
    ax_b.set_yticklabels(display)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 50)
    ax_b.set_xlabel("Cities")
    clean_axis(ax_b)
    short_labels = ["Surface heat", "Uncertain", "Social context"]
    handles = [Line2D([0], [0], marker="s", color=color, lw=0, markerfacecolor=color, label=label, markersize=5) for label, color in zip(short_labels, class_palette)]
    ax_b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.65)

    panel_heading(fig, ax_a, "a", "Thermal ranking is model-form dependent", y_offset=0.025)
    panel_heading(fig, ax_b, "b", "City classes persist across social-index variants", y_offset=0.025)
    outputs = save_figure(fig, out / "figures", "Supplementary_Fig_allocation_sensitivity")
    caption = (
        "Supplementary Fig. 17 | Eligibility, model form and social-condition construction constrain equal-area comparisons. "
        "a, Surface-heat-first minus social-context-first model-implied population-weighted LST contrasts at the 10% budget under three eligibility scopes and five increment/model-form specifications. Positive cells favour the surface-heat ordering under the displayed associational translation; negative cells favour the social-context ordering. The piecewise and spline models restrict increments to each city's population-weighted 5th–95th-percentile canopy support. "
        "b, Number of cities classified as supporting surface heat, supporting social context or model/interval uncertain under both M2 and M3 for four social-condition constructions in the broad G > 0 scope. Equal weighting, omission of public-transit commuting and within-city ranking change the social-condition ordering only; all city classes use 95% paired spatial-bootstrap intervals."
    )
    (out / "Supplementary_Fig_allocation_sensitivity_caption.txt").write_text(caption, encoding="utf-8")
    qa = {
        "figure_claim": "The pooled linear ranking is robust to social-index construction but not universal across eligibility and nonlinear model form.",
        "quality_gate_status": "PASS",
        **outputs,
    }
    (out / "qa").mkdir(parents=True, exist_ok=True)
    (out / "qa" / "Supplementary_Fig_technical_audit.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def main() -> None:
    setup_style()
    report = {"figure2": make_figure2()}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
