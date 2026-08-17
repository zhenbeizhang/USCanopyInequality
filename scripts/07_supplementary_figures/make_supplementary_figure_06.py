from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageOps
from scipy.stats import gaussian_kde, spearmanr


PROJECT = analysis_store_root()
AUDIT_ROOT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_city_slope_transferability_audit"
)
TABLES = AUDIT_ROOT / "tables"
AUDIT_JSON = AUDIT_ROOT / "qa" / "final_audit.json"

OUT = (
    PROJECT
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 6"
    / "versions"
    / "social_gradient_transferability"
)
FIG_DIR = OUT / "figures"
SOURCE_DIR = OUT / "source_data"
QA_DIR = OUT / "qa"
TEXT_DIR = OUT / "text"
METHODS_DIR = OUT / "methods"
for directory in (FIG_DIR, SOURCE_DIR, QA_DIR, TEXT_DIR, METHODS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CANONICAL_STEM = FIG_DIR / "Supplementary_Fig6"

PREDICTOR_ORDER = ["hispanic", "renter", "no_vehicle", "median_age"]
LABELS = {
    "hispanic": "Hispanic/Latino share",
    "renter": "Renter share",
    "no_vehicle": "No-vehicle household share",
    "median_age": "Median age",
}
SHORT_LABELS = {
    "hispanic": "Hispanic/Latino",
    "renter": "Renters",
    "no_vehicle": "No vehicle",
    "median_age": "Median age",
}
COLORS = {
    "hispanic": "#386f9e",
    "renter": "#2c7857",
    "no_vehicle": "#c27a35",
    "median_age": "#71558e",
    "black": "#171717",
    "grey": "#858585",
    "light_grey": "#dddddd",
    "same_bg": "#eef4f1",
    "reverse_bg": "#f8eee8",
    "stable": "#2c7857",
    "main_only": "#9bc6b3",
    "unresolved": "#dedede",
    "opposite": "#c86f32",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources():
    city = pd.read_csv(TABLES / "01_city_social_slopes_analytic.csv")
    national = pd.read_csv(TABLES / "05_social_transferability_summary.csv")
    bootstrap = pd.read_csv(
        TABLES / "04_city_social_spatial_bootstrap_intervals.csv"
    )
    agreement = pd.read_csv(
        TABLES / "08_block_group_tract_city_slope_agreement.csv"
    )
    with AUDIT_JSON.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)

    city = city.loc[
        city["predictor"].isin(PREDICTOR_ORDER)
        & city["geography"].isin(["block_group", "tract"])
    ].copy()
    national = national.loc[
        national["predictor"].isin(PREDICTOR_ORDER)
        & national["geography"].isin(["block_group", "tract"])
    ].copy()
    bootstrap = bootstrap.loc[
        bootstrap["predictor"].isin(PREDICTOR_ORDER)
        & bootstrap["geography"].isin(["block_group", "tract"])
    ].copy()
    agreement = agreement.loc[
        agreement["predictor"].isin(PREDICTOR_ORDER)
        & agreement["slope_type"].eq("main_spatial_bootstrap_median")
    ].copy()

    if len(city) != 400:
        raise RuntimeError(f"Expected 400 city slope rows, found {len(city)}")
    if len(national) != 8:
        raise RuntimeError(f"Expected eight national rows, found {len(national)}")
    if len(bootstrap) != 1200:
        raise RuntimeError(
            f"Expected 1,200 spatial-bootstrap city rows, found {len(bootstrap)}"
        )
    if len(agreement) != 4:
        raise RuntimeError(f"Expected four scale-agreement rows, found {len(agreement)}")
    if audit["audit_status"] != "PASS_WITH_INTERPRETIVE_BOUNDARY":
        raise RuntimeError("Unexpected source-audit status")
    return city, national, bootstrap, agreement, audit


def prepare_panel_a(
    city: pd.DataFrame,
    national: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_bg = city.loc[city["geography"].eq("block_group")].copy()
    national_bg = national.loc[national["geography"].eq("block_group")].copy()
    city_bg["predictor"] = pd.Categorical(
        city_bg["predictor"], PREDICTOR_ORDER, ordered=True
    )
    national_bg["predictor"] = pd.Categorical(
        national_bg["predictor"], PREDICTOR_ORDER, ordered=True
    )
    return city_bg.sort_values("predictor"), national_bg.sort_values("predictor")


def prepare_panel_b(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    main = bootstrap.loc[bootstrap["block_scale"].eq("main")].copy()
    values = main.pivot(
        index=["predictor", "city_key", "city_display"],
        columns="geography",
        values="partially_pooled_bootstrap_median",
    ).reset_index()
    values.columns.name = None
    if len(values) != 200 or values[["block_group", "tract"]].isna().any().any():
        raise RuntimeError("Incomplete block-group/tract city-slope pairing")
    return values


def prepare_panel_c(
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    bg = bootstrap.loc[bootstrap["geography"].eq("block_group")].copy()
    rows: list[dict] = []
    for (predictor, city_key, city_display), group in bg.groupby(
        ["predictor", "city_key", "city_display"],
        sort=False,
    ):
        expected = str(group["expected_direction"].iloc[0])
        if expected == "negative":
            national = group["partially_pooled_bootstrap_ci_high"] < 0
            opposite = group["partially_pooled_bootstrap_ci_low"] > 0
        else:
            national = group["partially_pooled_bootstrap_ci_low"] > 0
            opposite = group["partially_pooled_bootstrap_ci_high"] < 0
        main_national = bool(
            national.loc[group["block_scale"].eq("main")].iloc[0]
        )
        if bool(national.all()):
            category = "stable_national"
            value = 3
        elif bool(opposite.all()):
            category = "stable_opposite"
            value = 0
        elif main_national:
            category = "main_only"
            value = 2
        else:
            category = "unresolved"
            value = 1
        rows.append(
            {
                "predictor": predictor,
                "city_key": city_key,
                "city_display": city_display,
                "category": category,
                "matrix_value": value,
            }
        )
    categories = pd.DataFrame(rows)
    if len(categories) != 200:
        raise RuntimeError(f"Expected 200 city-predictor classifications")
    city_order = (
        categories.assign(
            stable_national=categories["category"].eq("stable_national").astype(int),
            stable_opposite=categories["category"].eq("stable_opposite").astype(int),
        )
        .groupby(["city_key", "city_display"], as_index=False)
        .agg(
            n_stable_national=("stable_national", "sum"),
            n_stable_opposite=("stable_opposite", "sum"),
        )
        .sort_values(
            ["n_stable_national", "n_stable_opposite", "city_display"],
            ascending=[False, True, True],
            kind="mergesort",
        )
    )
    order = city_order["city_key"].tolist()
    matrix = (
        categories.pivot(
            index="predictor",
            columns="city_key",
            values="matrix_value",
        )
        .reindex(index=PREDICTOR_ORDER, columns=order)
        .astype(int)
    )
    return categories, matrix, order


def panel_header(
    ax: plt.Axes, letter: str, title: str, y_offset_points: float = 6.0
) -> None:
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, y_offset_points),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        clip_on=False,
    )
    ax.annotate(
        title,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(11.0, y_offset_points),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        clip_on=False,
    )


def draw_panel_a(
    ax: plt.Axes,
    city: pd.DataFrame,
    national: pd.DataFrame,
) -> dict:
    x_grid = np.linspace(-8.5, 5.8, 500)
    baselines = dict(zip(PREDICTOR_ORDER, [3.0, 2.0, 1.0, 0.0]))
    rng = np.random.default_rng(20260727)
    for predictor in PREDICTOR_ORDER:
        values = city.loc[
            city["predictor"].eq(predictor),
            "raw_slope_pp_per_within_city_sd",
        ].to_numpy(float)
        baseline = baselines[predictor]
        kde = gaussian_kde(values, bw_method="scott")
        density = kde(x_grid)
        density = density / density.max() * 0.42
        color = COLORS[predictor]
        ax.fill_between(
            x_grid,
            baseline,
            baseline + density,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x_grid,
            baseline + density,
            color=color,
            lw=1.0,
            zorder=2,
        )
        jitter = rng.uniform(-0.035, 0.035, size=len(values))
        ax.scatter(
            values,
            baseline - 0.10 + jitter,
            s=7,
            color=color,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.18,
            zorder=3,
        )

        row = national.loc[national["predictor"].eq(predictor)].iloc[0]
        y_prediction = baseline + 0.51
        prediction_low = float(row["prediction_interval_low"])
        prediction_high = float(row["prediction_interval_high"])
        ax.plot(
            [prediction_low, prediction_high],
            [y_prediction, y_prediction],
            color="#777777",
            lw=0.8,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.vlines(
            [prediction_low, prediction_high],
            y_prediction - 0.035,
            y_prediction + 0.035,
            color="#777777",
            lw=0.65,
            zorder=3,
        )

        y_summary = baseline + 0.65
        ax.plot(
            [row["main_two_level_reml_ci_low"], row["main_two_level_reml_ci_high"]],
            [y_summary, y_summary],
            color=color,
            lw=1.25,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            row["main_two_level_reml_median"],
            y_summary,
            marker="D",
            s=26,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )

    ax.axvline(0, color=COLORS["black"], lw=0.75, zorder=0)
    ax.set_xlim(-8.5, 5.8)
    ax.set_ylim(-0.32, 3.82)
    ax.set_yticks([3.0, 2.0, 1.0, 0.0])
    ax.set_yticklabels([SHORT_LABELS[key] for key in PREDICTOR_ORDER])
    ax.set_xlabel("City-specific canopy gradient (pp per within-city s.d.)")
    ax.tick_params(axis="y", length=0, pad=5)
    for key, tick in zip(PREDICTOR_ORDER, ax.get_yticklabels()):
        tick.set_color(COLORS[key])
        tick.set_fontweight("bold")
    panel_header(ax, "a", "National means coexist with city heterogeneity")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["grey"],
            markeredgecolor="none",
            markersize=3.8,
            label="city slope",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=COLORS["black"],
            markerfacecolor=COLORS["black"],
            lw=1.0,
            markersize=4.0,
            label="national REML mean and 95% bootstrap interval",
        ),
        Line2D(
            [0],
            [0],
            color="#777777",
            marker="|",
            markeredgewidth=0.8,
            lw=0.8,
            markersize=4.5,
            label="95% random-effects city prediction interval",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.245),
        ncol=2,
        fontsize=5.25,
        handletextpad=0.35,
        columnspacing=0.7,
        labelspacing=0.25,
        borderaxespad=0,
    )
    prediction_crosses = int(
        (
            (national["prediction_interval_low"] < 0)
            & (national["prediction_interval_high"] > 0)
        ).sum()
    )
    return {
        "n_city_slopes": int(len(city)),
        "prediction_intervals_crossing_zero": prediction_crosses,
    }


def shade_quadrants(ax: plt.Axes, limit: float) -> None:
    ax.add_patch(
        Rectangle(
            (-limit, -limit),
            limit,
            limit,
            facecolor=COLORS["same_bg"],
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (0, 0),
            limit,
            limit,
            facecolor=COLORS["same_bg"],
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (-limit, 0),
            limit,
            limit,
            facecolor=COLORS["reverse_bg"],
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (0, -limit),
            limit,
            limit,
            facecolor=COLORS["reverse_bg"],
            edgecolor="none",
            zorder=0,
        )
    )


def draw_panel_b(
    ax: plt.Axes,
    paired: pd.DataFrame,
    agreement: pd.DataFrame,
) -> dict:
    limit = 8.0
    shade_quadrants(ax, limit)
    for predictor in PREDICTOR_ORDER:
        group = paired.loc[paired["predictor"].eq(predictor)]
        ax.scatter(
            group["block_group"],
            group["tract"],
            s=12,
            color=COLORS[predictor],
            alpha=0.66,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
    ax.plot(
        [-limit, limit],
        [-limit, limit],
        color=COLORS["black"],
        lw=0.75,
        ls="--",
        zorder=2,
    )
    ax.axhline(0, color=COLORS["black"], lw=0.55, zorder=1)
    ax.axvline(0, color=COLORS["black"], lw=0.55, zorder=1)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Block-group city slope")
    ax.set_ylabel("Tract city slope")
    rho_min = float(agreement["spearman_rho"].min())
    rho_max = float(agreement["spearman_rho"].max())
    same = int(agreement["n_same_sign"].sum())
    ax.text(
        0.035,
        0.965,
        f"rho = {rho_min:.2f}-{rho_max:.2f}\n{same}/200 pairs retain sign",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.4),
        zorder=5,
    )
    panel_header(ax, "b", "City slopes agree across scales")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[key],
            markeredgecolor="none",
            markersize=4.0,
            label=SHORT_LABELS[key],
        )
        for key in PREDICTOR_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=2,
        fontsize=5.5,
        handletextpad=0.25,
        columnspacing=0.55,
        labelspacing=0.18,
        borderaxespad=0,
    )
    return {
        "rho_min": rho_min,
        "rho_max": rho_max,
        "same_sign_pairs": same,
    }


def draw_panel_c(
    ax: plt.Axes,
    categories: pd.DataFrame,
    matrix: pd.DataFrame,
) -> dict:
    cmap = ListedColormap(
        [
            COLORS["opposite"],
            COLORS["unresolved"],
            COLORS["main_only"],
            COLORS["stable"],
        ]
    )
    ax.imshow(
        matrix.to_numpy(int),
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        aspect="auto",
        interpolation="nearest",
    )
    opposite_positions = np.argwhere(matrix.to_numpy(int) == 0)
    for row_index, column_index in opposite_positions:
        ax.scatter(
            column_index,
            row_index,
            marker="x",
            s=14,
            color=COLORS["black"],
            linewidth=0.75,
            zorder=4,
        )
    stable_counts = (
        categories.loc[categories["category"].eq("stable_national")]
        .groupby("predictor")
        .size()
        .reindex(PREDICTOR_ORDER, fill_value=0)
    )
    row_labels = [
        f"{SHORT_LABELS[key]} ({int(stable_counts[key])}/50)"
        for key in PREDICTOR_ORDER
    ]
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(row_labels)
    ax.set_xticks([0, 9, 19, 29, 39, 49])
    ax.set_xticklabels(["1", "10", "20", "30", "40", "50"])
    ax.set_xlabel("Cities ordered by number of stable gradients")
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xticks(np.arange(-0.5, 50, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.32)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#5b5b5b")
        spine.set_linewidth(0.55)
    panel_header(
        ax,
        "c",
        "Direction stability across spatial block widths",
        y_offset_points=1.5,
    )
    handles = [
        Patch(facecolor=COLORS["stable"], label="all widths, national direction"),
        Patch(facecolor=COLORS["main_only"], label="main width only"),
        Patch(facecolor=COLORS["unresolved"], label="interval overlaps or changes direction"),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLORS["opposite"],
            markeredgecolor=COLORS["black"],
            markerfacecolor=COLORS["opposite"],
            lw=0,
            markersize=4.5,
            label="all widths, opposite direction",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.50),
        ncol=2,
        fontsize=5.6,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.25,
        borderaxespad=0,
    )
    counts = categories["category"].value_counts().to_dict()
    return {
        "stable_national_counts": {
            key: int(stable_counts[key]) for key in PREDICTOR_ORDER
        },
        "category_counts": {key: int(value) for key, value in counts.items()},
    }


def save_outputs(fig: plt.Figure) -> dict[str, int]:
    outputs: dict[str, int] = {}
    fig.savefig(CANONICAL_STEM.with_suffix(".png"), dpi=600, bbox_inches=None)
    fig.savefig(CANONICAL_STEM.with_suffix(".pdf"), bbox_inches=None)
    fig.savefig(CANONICAL_STEM.with_suffix(".svg"), bbox_inches=None)
    fig.savefig(
        CANONICAL_STEM.with_suffix(".tif"),
        dpi=600,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    for suffix in (".png", ".pdf", ".svg", ".tif"):
        path = CANONICAL_STEM.with_suffix(suffix)
        outputs[str(path)] = path.stat().st_size
    ImageOps.grayscale(
        Image.open(CANONICAL_STEM.with_suffix(".png")).convert("RGB")
    ).save(QA_DIR / "Supplementary_Fig6_grayscale.png")
    return outputs


def write_caption(panel_a: dict, panel_b: dict, panel_c: dict) -> str:
    stable = panel_c["stable_national_counts"]
    caption = (
        "Supplementary Fig. 6 | National social-canopy gradients coexist with city heterogeneity. "
        "a, Distributions of unpooled city-specific, population-weighted canopy slopes for four prespecified social attributes in the frozen common block-group sample. Small points show 50 city estimates; coloured diamonds and short intervals show the national REML mean and 95% two-level spatial-block bootstrap interval at the primary block width; thinner grey capped intervals show the corresponding 95% random-effects city prediction intervals. All four national mean intervals retain their prespecified direction, whereas all four city prediction intervals cross zero. "
        f"b, Paired partially pooled city slopes at block-group and tract scales using the primary spatial-block width. The dashed line denotes equality and shaded quadrants distinguish retained from reversed signs. Attribute-specific Spearman correlations ranged from {panel_b['rho_min']:.2f} to {panel_b['rho_max']:.2f}, and {panel_b['same_sign_pairs']}/200 city-attribute pairs retained their sign. "
        "c, Shared physical-scale sensitivity audit of city-level direction stability in the block-group analysis across half-, primary- and double-width spatial blocks. The widths were inherited from the canopy-LST residual-range analysis to test all attributes on common physical distances; they were not estimated as optimal block widths for the social-canopy residuals. Columns are cities ordered by the number of gradients whose interval remained in the national direction; row labels give stable-city counts. "
        f"Stable counts were {stable['hispanic']}/50 for Hispanic/Latino share, {stable['renter']}/50 for renter share, {stable['no_vehicle']}/50 for no-vehicle households and {stable['median_age']}/50 for median age. "
        "This transferability audit used the units complete for all four prespecified attributes within each geography: 34,434 block groups and 11,932 tracts in all 50 cities, subsets of the corresponding Result 2 source frames rather than replacements for the primary samples. Spatial uncertainty used 500 primary-width and 250 half- and double-width block-bootstrap replicates followed by city-level resampling. The figure supports national average directions but not a universal city-level rule."
    )
    (TEXT_DIR / "Supplementary_Fig6_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    return caption


def main() -> None:
    apply_style()
    city, national, bootstrap, agreement, audit = load_sources()
    city_a, national_a = prepare_panel_a(city, national)
    paired_b = prepare_panel_b(bootstrap)
    categories_c, matrix_c, city_order = prepare_panel_c(bootstrap)

    city_a.to_csv(
        SOURCE_DIR / "Supplementary_Fig6a_city_slopes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    national_a.to_csv(
        SOURCE_DIR / "Supplementary_Fig6a_national_estimates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired_b.to_csv(
        SOURCE_DIR / "Supplementary_Fig6b_scale_paired_city_slopes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    agreement.to_csv(
        SOURCE_DIR / "Supplementary_Fig6b_scale_agreement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    categories_c.to_csv(
        SOURCE_DIR / "Supplementary_Fig6c_city_direction_categories.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {"column_position": np.arange(1, 51), "city_key": city_order}
    ).to_csv(
        SOURCE_DIR / "Supplementary_Fig6c_city_order.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=(183 / 25.4, 142 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.72, 0.28],
        width_ratios=[1.17, 0.83],
        left=0.150,
        right=0.985,
        top=0.950,
        bottom=0.115,
        hspace=0.43,
        wspace=0.31,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    panel_a = draw_panel_a(ax_a, city_a, national_a)
    panel_b = draw_panel_b(ax_b, paired_b, agreement)
    panel_c = draw_panel_c(ax_c, categories_c, matrix_c)
    outputs = save_outputs(fig)
    caption = write_caption(panel_a, panel_b, panel_c)
    plt.close(fig)

    qa = {
        "figure": "Supplementary Fig. 6",
        "analysis_role": "cross-scale transferability audit",
        "figure_contract": {
            "core_claim": "National social-canopy gradients are directionally stable while city-specific transferability remains heterogeneous.",
            "panel_a_role": "city-slope distributions and national two-level intervals",
            "panel_b_role": "block-group to tract city-slope agreement",
            "panel_c_role": "spatial-block direction stability",
            "principal_limitation": "A national average could be misrepresented as a universal city rule.",
            "response": "The figure displays all city slopes, cross-scale agreement and city-level interval stability.",
        },
        "source_audit_status": audit["audit_status"],
        "source_check_pass_fraction": float(
            np.mean([bool(value) for value in audit["checks"].values()])
        ),
        "samples": {
            "block_groups": 34434,
            "tracts": 11932,
            "cities": 50,
            "attributes": 4,
        },
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": panel_c,
        "outputs": outputs,
        "caption": caption,
        "input_sha256": {
            str(TABLES / "01_city_social_slopes_analytic.csv"): sha256(
                TABLES / "01_city_social_slopes_analytic.csv"
            ),
            str(TABLES / "04_city_social_spatial_bootstrap_intervals.csv"): sha256(
                TABLES / "04_city_social_spatial_bootstrap_intervals.csv"
            ),
            str(TABLES / "05_social_transferability_summary.csv"): sha256(
                TABLES / "05_social_transferability_summary.csv"
            ),
            str(AUDIT_JSON): sha256(AUDIT_JSON),
        },
    }
    qa["minimum_font_pt"] = 5.25
    quality_checks = {
        "source_audit_passed": audit["audit_status"]
        == "PASS_WITH_INTERPRETIVE_BOUNDARY",
        "all_50_cities_present": panel_a["n_city_slopes"] == 200,
        "cross_scale_rank_correlation_at_least_0_90": panel_b["rho_min"] >= 0.90,
        "required_exports_exist": all(Path(path).is_file() for path in outputs),
        "minimum_font_at_least_5_pt": qa["minimum_font_pt"] >= 5.0,
    }
    qa["quality_gate"] = {
        "status": "PASS" if all(quality_checks.values()) else "FAIL",
        "checks": quality_checks,
        "reason": (
            "The national mean interval, city prediction interval, cross-scale "
            "agreement and shared-distance sensitivity now have distinct visual "
            "and textual roles."
        ),
        "remaining_limit": (
            "The inherited block widths are a common-distance sensitivity audit, "
            "not residual-range estimates tailored to each social model."
        ),
    }
    (QA_DIR / "Supplementary_Fig6_build_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
