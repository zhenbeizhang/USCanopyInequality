from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


PROJECT_ROOT = analysis_store_root()
SOURCE = (
    PROJECT_ROOT
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 4"
    / "versions"
    / "public_product_comparison"
)
SUPP_ROOT = SOURCE.parent / "public_product_comparison"
FIG_DIR = SUPP_ROOT / "figures"
SRC_DIR = SUPP_ROOT / "source_data"
SCRIPT_DIR = SUPP_ROOT / "methods"
QA_DIR = SUPP_ROOT / "qa"
TEXT_DIR = SUPP_ROOT / "text"
for directory in (SUPP_ROOT, FIG_DIR, SRC_DIR, SCRIPT_DIR, QA_DIR, TEXT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

R1_ROOT = PROJECT_ROOT / "Result" / "Result1" / "Result1_low_resolution_product_experiment"
R2_ROOT = (
    PROJECT_ROOT
    / "Result"
    / "Result2"
    / "Result2_low_resolution_inequality_experiment_city_interior"
)
SAME_SAMPLE_MODELS = (
    SOURCE
    / "source_data"
    / "Supplementary_Fig4_same_sample_standardized_models.csv"
)
WINDOW_SELECTION = (
    PROJECT_ROOT
    / "Result"
    / "Result1"
    / "Result1_NatureCities_manuscript_Fig1"
    / "source_data"
    / "Fig1a_sparse_scene_window_selection.csv"
)
PRODUCT_MANIFEST = PROJECT_ROOT / "50\u57ce\u5e02\u4f4e\u5206\u8fa8\u7387\u4ea7\u54c1" / "metadata" / "download_manifest.csv"
MASK_ROOT = PROJECT_ROOT / "predict_mosaic_clip2"
RGB_ROOT = PROJECT_ROOT / "\u539f\u59cbtif"

STEM = FIG_DIR / "Supplementary_Fig4"

PRODUCTS = {
    "worldcover": {
        "short": "WorldCover 10 m",
        "header": "WorldCover\n10 m",
        "band": 1,
        "tree_values": (10,),
        "nodata": 255,
    },
    "esri": {
        "short": "Esri LULC 10 m",
        "header": "Esri LULC\n10 m",
        "band": 1,
        "tree_values": (2,),
        "nodata": 255,
    },
    "glad": {
        "short": "GLAD 30 m",
        "header": "GLAD extent\n30 m",
        "band": 2,
        "tree_values": (1,),
        "nodata": 255,
    },
    "glcfcs": {
        "short": "GLC_FCS30D 30 m",
        "header": "GLC_FCS30D\n30 m",
        "band": 2,
        "tree_values": (1,),
        "nodata": 250,
    },
}
PRODUCT_ORDER = list(PRODUCTS)
EXAMPLE_CITIES = ["New York", "Las Vegas", "Seattle"]

PREDICTOR_ORDER = [
    "Low income",
    "Poverty",
    "Older age",
    "Black",
    "Hispanic/Latino",
    "Asian",
    "People of colour",
    "Renters",
    "No vehicle",
    "Public transit",
    "Density",
]
MODEL_PRODUCT_ORDER = ["reference_0p6m", "worldcover", "esri", "glad", "glcfcs"]
MODEL_PRODUCT_LABELS = {
    "reference_0p6m": "0.6 m",
    "worldcover": "World\nCover",
    "esri": "Esri",
    "glad": "GLAD",
    "glcfcs": "FCS\n30D",
}

METRICS = [
    ("between_city_spearman_rho", "Between-city rank correlation", "o", "#30343b", 0.18),
    ("median_within_city_spearman_rho", "Within-city rank correlation", "s", "#287d78", 0.00),
    ("median_bottom_quintile_recall", "Lowest-quintile overlap", "D", "#cc6f3c", -0.18),
]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.moveaxis(rgb, 0, -1).astype(np.float32)
    out = np.zeros_like(arr)
    for band in range(3):
        valid = arr[..., band][np.isfinite(arr[..., band])]
        lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
        if hi <= lo:
            hi = lo + 1
        out[..., band] = np.clip((arr[..., band] - lo) / (hi - lo), 0, 1)
    return out


def mask_rgb(mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    background = np.array([0.955, 0.955, 0.935])
    canopy = np.array([0.035, 0.43, 0.25])
    nodata = np.array([0.80, 0.80, 0.80])
    out = np.full((*mask.shape, 3), background, dtype=float)
    if valid is not None:
        out[~valid] = nodata
    out[mask] = canopy
    return out


def read_public_mask(
    path: Path,
    key: str,
    reference_crs,
    reference_transform,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = PRODUCTS[key]
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=reference_crs,
            transform=reference_transform,
            width=width,
            height=height,
            resampling=Resampling.nearest,
            src_nodata=cfg["nodata"],
            nodata=cfg["nodata"],
        ) as vrt:
            values = vrt.read(cfg["band"])
    valid = values != cfg["nodata"]
    return valid & np.isin(values, cfg["tree_values"]), valid


def panel_header(fig: plt.Figure, ax: plt.Axes, letter: str, title: str, dy: float = 0.026) -> None:
    pos = ax.get_position()
    y = pos.y1 + dy
    fig.text(pos.x0, y, letter, fontsize=9.0, fontweight="bold", ha="left", va="bottom")
    fig.text(pos.x0 + 0.021, y, title, fontsize=8.7, fontweight="bold", ha="left", va="bottom")


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection = pd.read_csv(WINDOW_SELECTION)
    selection = selection.loc[selection["city_display"].isin(EXAMPLE_CITIES)].copy()
    selection["city_display"] = pd.Categorical(
        selection["city_display"], EXAMPLE_CITIES, ordered=True
    )
    selection = selection.sort_values("city_display").reset_index(drop=True)
    if len(selection) != 3:
        raise RuntimeError("Expected three pre-specified diagnostic windows.")

    manifest = pd.read_csv(PRODUCT_MANIFEST)
    manifest = manifest.loc[manifest["product_key"].isin(PRODUCT_ORDER)].copy()
    if manifest.groupby("product_key")["city"].nunique().ne(50).any():
        raise RuntimeError("The public-product manifest is not a complete 50-city set.")

    recovery = pd.read_csv(R1_ROOT / "tables" / "product_summary_city_bootstrap_ci.csv")
    recovery = recovery.loc[recovery["metric"].isin([m[0] for m in METRICS])].copy()
    if len(recovery) != 12:
        raise RuntimeError("Expected four products by three recovery metrics.")

    models = pd.read_csv(SAME_SAMPLE_MODELS)
    models = models.loc[
        models["product_key"].isin(MODEL_PRODUCT_ORDER)
        & models["scale"].isin(["block_group", "tract"])
    ].copy()
    if len(models) != 110:
        raise RuntimeError("Expected 11 predictors by five products by two spatial scales.")
    expected_n = {"block_group": 31819, "tract": 12438}
    for scale, expected in expected_n.items():
        scale_n = models.loc[models["scale"].eq(scale), "n_spatial_units"]
        if not scale_n.eq(expected).all():
            raise RuntimeError(f"Models do not use one frozen sample for {scale}.")
    return selection, manifest, recovery, models


def draw_panel_a(
    fig: plt.Figure,
    slot,
    selection: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[list[plt.Axes], pd.DataFrame]:
    grid = slot.subgridspec(3, 6, wspace=0.025, hspace=0.035)
    axes: list[plt.Axes] = []
    image_records: list[dict] = []
    path_lookup = {
        (r.city, r.product_key): Path(r.output_file) for r in manifest.itertuples(index=False)
    }
    headers = ["Aerial image", "0.6 m atlas"] + [PRODUCTS[k]["header"] for k in PRODUCT_ORDER]

    for row_i, sample in enumerate(selection.itertuples(index=False)):
        window = Window(
            col_off=int(sample.col),
            row_off=int(sample.row),
            width=int(sample.window_size_px),
            height=int(sample.window_size_px),
        )
        mask_path = MASK_ROOT / f"{sample.city_key}_mosaic_clip2.tif"
        rgb_path = RGB_ROOT / f"{sample.city_key}.tif"
        with rasterio.open(mask_path) as src:
            reference_values = src.read(1, window=window)
            transform = src.window_transform(window)
            crs = src.crs
        with rasterio.open(rgb_path) as src:
            rgb = src.read([1, 2, 3], window=window)
        reference_valid = reference_values != 254
        reference_tree = (reference_values == 255) & reference_valid
        images = [stretch_rgb(rgb), mask_rgb(reference_tree, reference_valid)]

        image_records.append(
            {
                "city": sample.city_display,
                "city_key": sample.city_key,
                "surface": "0.6 m canopy atlas",
                "resolution_m": 0.6,
                "row_offset": int(sample.row),
                "column_offset": int(sample.col),
                "window_pixels": int(sample.window_size_px),
                "window_width_m": float(sample.window_size_px) * 0.6,
                "tree_fraction": float(reference_tree[reference_valid].mean()),
                "raw_file": str(mask_path),
            }
        )

        for key in PRODUCT_ORDER:
            product_path = path_lookup[(sample.city_display, key)]
            tree, valid = read_public_mask(
                product_path,
                key,
                crs,
                transform,
                reference_values.shape[0],
                reference_values.shape[1],
            )
            images.append(mask_rgb(tree, valid))
            image_records.append(
                {
                    "city": sample.city_display,
                    "city_key": sample.city_key,
                    "surface": PRODUCTS[key]["short"],
                    "resolution_m": 10 if key in ("worldcover", "esri") else 30,
                    "row_offset": int(sample.row),
                    "column_offset": int(sample.col),
                    "window_pixels": int(sample.window_size_px),
                    "window_width_m": float(sample.window_size_px) * 0.6,
                    "tree_fraction": float(tree[valid].mean()) if valid.any() else np.nan,
                    "raw_file": str(product_path),
                }
            )

        for col_i, image in enumerate(images):
            ax = fig.add_subplot(grid[row_i, col_i])
            axes.append(ax)
            ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#3f3f3f")
                spine.set_linewidth(0.55)
            if row_i == 0:
                ax.set_title(headers[col_i], fontsize=7.4, color="black", pad=3.5, linespacing=0.9)
            if col_i == 0:
                ax.set_ylabel(sample.city_display, fontsize=7.5, rotation=90, labelpad=4.2)
            if row_i == 0 and col_i == 0:
                pixels_200m = 200 / 0.6
                x1 = reference_values.shape[1] - 36
                x0 = x1 - pixels_200m
                y = reference_values.shape[0] - 35
                ax.plot([x0, x1], [y, y], color="white", lw=3.2, solid_capstyle="butt")
                ax.plot([x0, x1], [y, y], color="black", lw=1.6, solid_capstyle="butt")
                ax.text((x0 + x1) / 2, y - 18, "200 m", ha="center", va="bottom", fontsize=6.3, color="black")
    return axes, pd.DataFrame(image_records)


def draw_panel_b(ax: plt.Axes, recovery: pd.DataFrame) -> mpl.cm.ScalarMappable:
    metric_order = [item[0] for item in METRICS]
    metric_labels = ["City rank", "Within-city\nrank", "Lowest-quintile\noverlap"]
    product_labels = ["WorldCover", "Esri LULC", "GLAD", "FCS30D"]

    estimate = np.full((len(PRODUCT_ORDER), len(metric_order)), np.nan)
    ci_low = np.full_like(estimate, np.nan)
    ci_high = np.full_like(estimate, np.nan)
    for i, product in enumerate(PRODUCT_ORDER):
        for j, metric in enumerate(metric_order):
            row = recovery.loc[
                recovery["product_key"].eq(product) & recovery["metric"].eq(metric)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Missing or duplicated concordance cell: {product}, {metric}")
            estimate[i, j] = float(row.iloc[0]["estimate"])
            ci_low[i, j] = float(row.iloc[0]["bootstrap_ci_low_95"])
            ci_high[i, j] = float(row.iloc[0]["bootstrap_ci_high_95"])

    cmap = LinearSegmentedColormap.from_list(
        "atlas_concordance", ["#f5f3ec", "#d7e8d9", "#83bda4", "#1f705f"]
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    image = ax.imshow(estimate, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels(metric_labels, linespacing=0.9)
    ax.set_yticks(np.arange(len(PRODUCT_ORDER)))
    ax.set_yticklabels(product_labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=3)

    ax.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PRODUCT_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(estimate.shape[0]):
        for j in range(estimate.shape[1]):
            value = estimate[i, j]
            ink = "white" if value >= 0.66 else "#111111"
            ax.text(
                j,
                i - 0.13,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.35,
                fontweight="bold",
                color=ink,
                zorder=4,
            )
            x0 = j - 0.34 + 0.68 * np.clip(ci_low[i, j], 0, 1)
            x1 = j - 0.34 + 0.68 * np.clip(ci_high[i, j], 0, 1)
            xc = j - 0.34 + 0.68 * np.clip(value, 0, 1)
            y = i + 0.20
            ax.plot([x0, x1], [y, y], color=ink, lw=0.75, solid_capstyle="butt", zorder=4)
            ax.plot([x0, x0], [y - 0.035, y + 0.035], color=ink, lw=0.65, zorder=4)
            ax.plot([x1, x1], [y - 0.035, y + 0.035], color=ink, lw=0.65, zorder=4)
            ax.scatter(xc, y, s=3.8, color=ink, linewidth=0, zorder=5)

    ax.set_xlim(-0.5, len(metric_order) - 0.5)
    ax.set_ylim(len(PRODUCT_ORDER) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.6)
    return image


def draw_panel_c(ax: plt.Axes, models: pd.DataFrame) -> mpl.cm.ScalarMappable:
    columns = [(scale, product) for scale in ["block_group", "tract"] for product in MODEL_PRODUCT_ORDER]
    matrix = np.full((len(PREDICTOR_ORDER), len(columns)), np.nan)
    sig = np.zeros_like(matrix, dtype=bool)
    for i, predictor in enumerate(PREDICTOR_ORDER):
        for j, (scale, product) in enumerate(columns):
            row = models.loc[
                models["scale"].eq(scale)
                & models["product_key"].eq(product)
                & models["predictor_label"].eq(predictor)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Missing or duplicated model cell: {scale}, {product}, {predictor}")
            matrix[i, j] = float(row.iloc[0]["estimate"])
            sig[i, j] = bool(row.iloc[0]["significant_fdr05"])

    cmap = LinearSegmentedColormap.from_list(
        "canopy_inequality",
        ["#b55545", "#ead0c9", "#f7f7f5", "#cfdfeb", "#42759d"],
    )
    norm = TwoSlopeNorm(vmin=-0.48, vcenter=0, vmax=0.32)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(PREDICTOR_ORDER)))
    ax.set_yticklabels(["Median age" if label == "Older age" else label for label in PREDICTOR_ORDER])
    labels = [MODEL_PRODUCT_LABELS[p] for _, p in columns]
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(labels, rotation=0, linespacing=0.88)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.45, color="black")
            if sig[i, j]:
                ax.scatter(j + 0.36, i - 0.31, s=5.4, c="black", marker="o", linewidth=0, zorder=3)
    ax.axvline(4.5, color="white", lw=2.1)
    ax.axvline(4.5, color="#383838", lw=0.75)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(10.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.6)
    ax.text(0.25, 1.064, "Block group", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax.text(0.75, 1.064, "Tract", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax.text(
        0.995,
        -0.035,
        "dot: FDR q < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
    )
    return image


def save_outputs(fig: plt.Figure) -> dict[str, int]:
    outputs = {}
    for output_stem in (STEM,):
        fig.savefig(output_stem.with_suffix(".svg"), bbox_inches=None)
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches=None)
        fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches=None)
        fig.savefig(
            output_stem.with_suffix(".tif"),
            dpi=600,
            bbox_inches=None,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        for suffix in (".svg", ".pdf", ".png", ".tif"):
            path = output_stem.with_suffix(suffix)
            outputs[str(path)] = path.stat().st_size
    with Image.open(STEM.with_suffix(".png")) as image:
        Image.fromarray(np.asarray(image.convert("L"))).save(QA_DIR / f"{STEM.name}_grayscale.png")
    return outputs


def write_caption() -> None:
    caption = (
        "Supplementary Fig. 4 | Public canopy products differ in neighbourhood patterns and social-gradient estimates. "
        "a, Three 538-m-wide urban windows comparing aerial imagery and the 0.6 m canopy atlas with four public 2020 land-cover or forest products. "
        "All public-product masks were reprojected to the 0.6 m display grid using nearest-neighbour resampling; green denotes pixels classified as tree or forest under each product's native definition. Blank product windows contain no pixels assigned to those classes and are not missing data. "
        "b, Concordance matrix for four public products and three atlas-recovery metrics. Cell colour and printed values show the centre estimate; the thin line, end caps and point within each cell show the 95% bootstrap confidence interval and centre estimate across 50 cities (5,000 resamples). "
        "c, Fully standardised city fixed-effect coefficients for the same 11 prespecified social predictors and the same spatial units within each scale. Variables were standardised within city after freezing the complete-case samples, and models were weighted by spatial-unit population. Values are residential-canopy standard deviations per one within-city predictor standard deviation; dots denote Benjamini-Hochberg FDR q < 0.05 from 9,999 city-level Rademacher wild-cluster bootstrap replicates. "
        "Models used the city-interior 80% rule and one common sample of 31,819 block groups or 12,438 tracts across all five canopy products and 50 cities. Public products differ from the 0.6 m atlas in spatial resolution, canopy ontology, classification algorithm and observation timing; the contrasts therefore quantify combined product disagreement rather than a pure resolution effect. They are not an independent accuracy validation and do not, by themselves, establish atlas superiority."
    )
    (TEXT_DIR / "Supplementary_Fig4_caption.txt").write_text(caption.replace("Older age", "Median age") + "\n", encoding="utf-8")


def main() -> None:
    selection, manifest, recovery, models = load_sources()
    width_in = 183 / 25.4
    height_in = 222 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.12, 1.0],
        left=0.085,
        right=0.985,
        top=0.955,
        bottom=0.092,
        hspace=0.17,
    )
    axes_a, image_source = draw_panel_a(fig, outer[0], selection, manifest)
    lower = outer[1].subgridspec(1, 2, width_ratios=[0.34, 0.66], wspace=0.27)
    ax_b = fig.add_subplot(lower[0, 0])
    ax_c = fig.add_subplot(lower[0, 1])
    concordance_map = draw_panel_b(ax_b, recovery)
    heatmap = draw_panel_c(ax_c, models)

    fig.canvas.draw()
    panel_header(fig, axes_a[0], "a", "Contrasting urban canopy representations", dy=0.026)
    panel_header(fig, ax_b, "b", "Concordance with the 0.6 m atlas", dy=0.045)
    panel_header(fig, ax_c, "c", "Social gradients under matched samples", dy=0.045)

    bpos = ax_b.get_position()
    bbar_ax = fig.add_axes([bpos.x0 + 0.05 * bpos.width, bpos.y0 - 0.043, 0.90 * bpos.width, 0.012])
    bbar = fig.colorbar(concordance_map, cax=bbar_ax, orientation="horizontal")
    bbar.set_ticks([0, 0.5, 1.0])
    bbar.ax.tick_params(labelsize=6.2, length=2, pad=1.5)
    bbar.set_label("Concordance with 0.6 m atlas", fontsize=6.5, labelpad=1.0)
    bbar.outline.set_linewidth(0.55)

    cpos = ax_c.get_position()
    cbar_ax = fig.add_axes([cpos.x0 + 0.05 * cpos.width, cpos.y0 - 0.043, 0.62 * cpos.width, 0.012])
    cbar = fig.colorbar(heatmap, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([-0.4, -0.2, 0, 0.2])
    cbar.ax.tick_params(labelsize=6.2, length=2, pad=1.5)
    cbar.set_label("Standardised residential-canopy coefficient", fontsize=6.5, labelpad=1.0)
    cbar.outline.set_linewidth(0.55)

    image_source.to_csv(SRC_DIR / "Supplementary_Fig4a_window_inventory.csv", index=False, encoding="utf-8-sig")
    recovery.to_csv(SRC_DIR / "Supplementary_Fig4b_recovery_metrics.csv", index=False, encoding="utf-8-sig")
    models.to_csv(SRC_DIR / "Supplementary_Fig4c_standardized_models.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(SRC_DIR / "Supplementary_Fig4a_window_selection.csv", index=False, encoding="utf-8-sig")
    outputs = save_outputs(fig)
    write_caption()
    plt.close(fig)

    qa = {
        "figure_contract": {
            "core_conclusion": "Public products partly retain city-level canopy gradients but show weaker neighbourhood concordance and yield different inequality estimates.",
            "archetype": "image plate plus quantitative comparison",
            "backend": "Python matplotlib",
        },
        "examples": EXAMPLE_CITIES,
        "n_cities": 50,
        "n_block_groups": int(models.loc[models.scale.eq("block_group"), "n_spatial_units"].iloc[0]),
        "n_tracts": int(models.loc[models.scale.eq("tract"), "n_spatial_units"].iloc[0]),
        "agreement_rows": int(len(recovery)),
        "model_rows": int(len(models)),
        "fdr_significant_cells": int(models["significant_fdr05"].sum()),
        "export_bytes": outputs,
        "notes": [
            "Public-product masks use each product's documented tree or forest class.",
            "Panel a uses nearest-neighbour resampling only for common display geometry.",
            "Panel c freezes one complete-case sample per spatial scale before within-city standardisation.",
            "Panel c uncertainty uses 9,999 city-level Rademacher wild-cluster bootstrap replicates.",
            "The contrasts combine spatial resolution, ontology, algorithm and timing differences.",
        ],
    }
    qa["status"] = "CANDIDATE"
    qa["minimum_font_pt"] = 5.45
    qa["blank_product_windows_are_zero_tree_class_not_missing"] = True
    (QA_DIR / "Supplementary_Fig4_build_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
