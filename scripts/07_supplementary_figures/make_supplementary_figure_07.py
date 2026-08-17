from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


PROJECT = analysis_store_root()
RESULT = PROJECT / "Result"
R3_FRAME = (
    RESULT
    / "Result3"
    / "Result3_support_aligned"
    / "data"
    / "support_aligned_model_frame_34566.csv"
)
R2_INVENTORY = (
    RESULT
    / "Result2"
    / "Result2_module1_boundary_inference_audit"
    / "tables"
    / "result2_boundary_sample_inventory_by_city.csv"
)
SCENE_INVENTORY = (
    PROJECT
    / "Landsat_Temperature"
    / "planetary_computer_landsat_scene_inventory.csv"
)

OUT = (
    RESULT
    / "Supplementary figure and table"
    / "Supplementary Fig. 7"
    / "versions"
    / "landsat_support"
)
FIG_DIR = OUT / "figures"
SOURCE_DIR = OUT / "source_data"
QA_DIR = OUT / "qa"
TEXT_DIR = OUT / "text"
METHODS_DIR = OUT / "methods"
for directory in (
    FIG_DIR,
    SOURCE_DIR,
    QA_DIR,
    TEXT_DIR,
    METHODS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

CANONICAL_STEM = FIG_DIR / "Supplementary_Fig7"

BLACK = "#1b1b1b"
BLUE = "#3b7396"
PURPLE = "#684879"
ORANGE = "#c96f35"
PALE_PURPLE = "#e9e3ec"
RANK_CMAP = LinearSegmentedColormap.from_list(
    "support_rank",
    ["#f4f4f1", "#bfd1d1", "#77a5a4", "#28616a"],
    N=256,
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.9,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
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


def normalize_city(value: str) -> str:
    cleaned = value.lower().replace("&", "and")
    cleaned = re.sub(r"[_\-()]+", " ", cleaned)
    for suffix in (
        "metropolitan government balance",
        "metro government balance",
        "city balance",
        "city",
    ):
        cleaned = cleaned.replace(suffix, " ")
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def short_city(name: str) -> str:
    replacements = {
        "Louisville Jefferson County": "Louisville",
        "Louisville-Jefferson County": "Louisville",
    }
    return replacements.get(name, name)


def ecdf(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def source_paths() -> dict[str, Path]:
    return {
        "result3_frame": R3_FRAME,
        "result2_inventory": R2_INVENTORY,
        "landsat_scene_inventory": SCENE_INVENTORY,
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [str(path) for path in source_paths().values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Landsat support inputs: {missing}")

    block_groups = pd.read_csv(R3_FRAME)
    if len(block_groups) != 34_566 or block_groups["city_key"].nunique() != 50:
        raise RuntimeError("Unexpected Result 3 support-aligned frame")

    result2 = pd.read_csv(R2_INVENTORY)
    result2 = result2.loc[
        result2["scale"].eq("block_group")
        & result2["sample"].eq("fully_contained"),
        ["result_city_key", "retained_units"],
    ].rename(
        columns={
            "result_city_key": "city_key",
            "retained_units": "n_result2_block_groups",
        }
    )
    if result2["city_key"].nunique() != 50:
        raise RuntimeError("Result 2 city inventory is incomplete")

    city = (
        block_groups.groupby(["city_key", "city_display"], as_index=False)
        .agg(
            n_result3_block_groups=("GEOID_str", "size"),
            valid_pixel_pct=(
                "valid_pixel_fraction",
                lambda values: float(np.mean(values) * 100),
            ),
        )
        .merge(result2, on="city_key", how="left", validate="one_to_one")
    )
    city["result3_retained_pct"] = (
        city["n_result3_block_groups"]
        / city["n_result2_block_groups"]
        * 100
    )
    city["city_norm"] = city["city_display"].map(normalize_city)

    scenes = (
        pd.read_csv(SCENE_INVENTORY)
        .drop_duplicates(["city", "item_id"])
        .copy()
    )
    scenes["city_norm"] = scenes["city"].map(normalize_city)
    scenes["sensor"] = scenes["platform"].str.lower().map(
        {
            "landsat-8": "landsat8_city_scenes",
            "landsat-9": "landsat9_city_scenes",
        }
    )
    if scenes["sensor"].isna().any():
        raise RuntimeError("Unexpected Landsat platform in scene inventory")
    scene_counts = (
        scenes.groupby(["city_norm", "sensor"], as_index=False)
        .size()
        .pivot(index="city_norm", columns="sensor", values="size")
        .fillna(0)
        .reset_index()
    )
    city = city.merge(
        scene_counts,
        on="city_norm",
        how="left",
        validate="one_to_one",
    )
    count_columns = ["landsat8_city_scenes", "landsat9_city_scenes"]
    if city[count_columns].isna().any().any():
        missing_cities = city.loc[
            city[count_columns].isna().any(axis=1), "city_display"
        ].tolist()
        raise RuntimeError(f"Scene inventory failed to align: {missing_cities}")
    city[count_columns] = city[count_columns].astype(int)
    city["total_city_scenes"] = city[count_columns].sum(axis=1)
    city["short_city"] = city["city_display"].map(short_city)
    city = city.sort_values(
        ["valid_pixel_pct", "result3_retained_pct", "city_display"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    block_group_output = block_groups[
        [
            "GEOID_str",
            "city_key",
            "city_display",
            "valid_pixel_fraction",
            "mean_valid_observations",
            "n_valid_Landsat_pixels",
            "n_total_Landsat_pixels",
            "population",
        ]
    ].copy()
    block_group_output["valid_pixel_pct"] = (
        block_group_output["valid_pixel_fraction"] * 100
    )
    return city, block_group_output, scenes


def matrix_values(city: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = [
        "valid_pixel_pct",
        "result3_retained_pct",
        "landsat8_city_scenes",
        "landsat9_city_scenes",
    ]
    raw = city[columns].to_numpy(float)
    ranks = np.column_stack(
        [
            city[column].rank(method="average", pct=True).to_numpy(float)
            for column in columns
        ]
    )
    return raw, ranks


def panel_heading(
    fig: plt.Figure,
    x: float,
    y: float,
    letter: str,
    title: str,
) -> None:
    fig.text(
        x,
        y,
        letter,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=BLACK,
    )
    fig.text(
        x + 0.020,
        y,
        title,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=BLACK,
    )


def draw_matrix_block(
    ax: plt.Axes,
    block: pd.DataFrame,
    raw: np.ndarray,
    ranks: np.ndarray,
) -> mpl.collections.QuadMesh:
    rows = len(block)
    mesh = ax.pcolormesh(
        np.arange(5) - 0.5,
        np.arange(rows + 1) - 0.5,
        ranks,
        cmap=RANK_CMAP,
        norm=Normalize(0, 1),
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
    )
    formats = [
        lambda value: f"{value:.1f}",
        lambda value: f"{value:.1f}",
        lambda value: f"{int(value)}",
        lambda value: f"{int(value)}",
    ]
    for row in range(rows):
        for column in range(4):
            colour = "white" if ranks[row, column] >= 0.67 else BLACK
            ax.text(
                column,
                row,
                formats[column](raw[row, column]),
                ha="center",
                va="center",
                fontsize=5.65,
                color=colour,
            )
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_yticks(np.arange(rows), block["short_city"])
    ax.tick_params(axis="y", length=0, pad=3.0, labelsize=5.95)
    ax.set_xticks(
        np.arange(4),
        [
            "Valid pixels\n(%)",
            "Units retained\n(%)",
            "Landsat 8\nrecords",
            "Landsat 9\nrecords",
        ],
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3.0, labelsize=5.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    return mesh


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(length=3.0, width=0.7, pad=2.0)
    ax.grid(False)


def build_figure():
    apply_style()
    city, block_groups, scenes = load_data()
    raw, ranks = matrix_values(city)
    if raw.shape != (50, 4):
        raise RuntimeError("Expected a 50-city by 4-diagnostic matrix")

    fig = plt.figure(figsize=(183 / 25.4, 156 / 25.4), facecolor="white")
    ax_a1 = fig.add_axes([0.155, 0.492, 0.305, 0.387])
    ax_a2 = fig.add_axes([0.655, 0.492, 0.305, 0.387])
    ax_b1 = fig.add_axes([0.075, 0.075, 0.395, 0.255])
    ax_b2 = fig.add_axes([0.575, 0.075, 0.395, 0.255])
    panel_heading(
        fig,
        0.045,
        0.957,
        "a",
        "City-level LST support and scene inventory",
    )
    panel_heading(
        fig,
        0.045,
        0.378,
        "b",
        "Block-group observation support",
    )

    mesh = draw_matrix_block(ax_a1, city.iloc[:25], raw[:25], ranks[:25])
    draw_matrix_block(ax_a2, city.iloc[25:], raw[25:], ranks[25:])
    colorbar_axis = fig.add_axes([0.395, 0.430, 0.210, 0.016])
    colorbar = fig.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["low", "median", "high"])
    colorbar.ax.tick_params(labelsize=5.8, length=1.8, width=0.55, pad=1.2)
    colorbar.outline.set_linewidth(0.55)
    colorbar.set_label(
        "Within-column percentile",
        fontsize=6.25,
        labelpad=1.0,
    )

    pixel_x, pixel_y = ecdf(block_groups["valid_pixel_pct"])
    pixel_p05 = float(block_groups["valid_pixel_pct"].quantile(0.05))
    share_exact_100 = float(
        np.isclose(block_groups["valid_pixel_pct"], 100.0).mean()
    )
    share_below_90 = float(
        (block_groups["valid_pixel_pct"] < 90.0).mean()
    )
    ax_b1.step(pixel_x, pixel_y, where="post", color=BLUE, lw=1.75)
    ax_b1.axvline(pixel_p05, color=ORANGE, lw=0.9, ls=(0, (3, 2)))
    ax_b1.annotate(
        f"5th percentile = {pixel_p05:.1f}%",
        xy=(pixel_p05, 0.05),
        xytext=(94.7, 0.17),
        ha="left",
        va="center",
        fontsize=6.1,
        color=BLACK,
        arrowprops={"arrowstyle": "-", "color": BLACK, "lw": 0.6},
    )
    ax_b1.annotate(
        f"{share_exact_100 * 100:.1f}% at 100%",
        xy=(100.0, 0.55),
        xytext=(92.0, 0.76),
        ha="left",
        va="center",
        fontsize=6.2,
        color=BLACK,
        arrowprops={"arrowstyle": "-", "color": BLACK, "lw": 0.65},
    )
    ax_b1.text(
        90.15,
        0.965,
        f"{share_below_90 * 100:.1f}% below 90%",
        ha="left",
        va="top",
        fontsize=5.9,
        color=BLACK,
    )
    ax_b1.set_xlim(90, 100.15)
    ax_b1.set_ylim(0, 1.02)
    ax_b1.set_xticks([90, 95, 97.5, 100], ["90", "95", "97.5", "100"])
    ax_b1.set_yticks([0, 0.5, 1.0])
    ax_b1.set_xlabel("Valid LST pixels within block group (%)")
    ax_b1.set_ylabel("Cumulative share")
    clean_axis(ax_b1)

    observation_x, observation_y = ecdf(
        block_groups["mean_valid_observations"]
    )
    observation_p05 = float(
        block_groups["mean_valid_observations"].quantile(0.05)
    )
    observation_median = float(
        block_groups["mean_valid_observations"].median()
    )
    observation_p95 = float(
        block_groups["mean_valid_observations"].quantile(0.95)
    )
    ax_b2.axvspan(
        observation_p05,
        observation_p95,
        color=PALE_PURPLE,
        alpha=0.72,
        lw=0,
    )
    ax_b2.step(
        observation_x,
        observation_y,
        where="post",
        color=PURPLE,
        lw=1.75,
    )
    ax_b2.axvline(
        observation_median,
        color=ORANGE,
        lw=0.9,
        ls=(0, (3, 2)),
    )
    ax_b2.annotate(
        f"median = {observation_median:.1f}",
        xy=(observation_median, 0.5),
        xytext=(82, 0.58),
        ha="center",
        va="center",
        fontsize=6.2,
        color=BLACK,
        arrowprops={"arrowstyle": "-", "color": BLACK, "lw": 0.6},
    )
    ax_b2.text(
        (observation_p05 + observation_p95) / 2,
        0.075,
        (
            f"5th-95th: {observation_p05:.1f}-"
            f"{observation_p95:.1f}"
        ),
        ha="center",
        va="bottom",
        fontsize=5.9,
        color=BLACK,
    )
    ax_b2.set_xlim(0, 160)
    ax_b2.set_ylim(0, 1.02)
    ax_b2.set_xticks([0, 40, 80, 120, 160])
    ax_b2.set_yticks([0, 0.5, 1.0])
    ax_b2.set_xlabel("Mean valid observations per LST pixel")
    ax_b2.set_ylabel("Cumulative share")
    clean_axis(ax_b2)

    stats = {
        "cities": int(len(city)),
        "block_groups": int(len(block_groups)),
        "unique_city_scene_records": int(len(scenes)),
        "landsat8_city_scene_records": int(
            scenes["platform"].str.lower().eq("landsat-8").sum()
        ),
        "landsat9_city_scene_records": int(
            scenes["platform"].str.lower().eq("landsat-9").sum()
        ),
        "minimum_city_valid_pixel_pct": float(city["valid_pixel_pct"].min()),
        "minimum_city_retained_pct": float(city["result3_retained_pct"].min()),
        "share_exactly_100_pct_valid": share_exact_100,
        "share_below_90_pct_valid": share_below_90,
        "valid_pixel_p05_pct": pixel_p05,
        "median_mean_valid_observations": observation_median,
        "mean_valid_observations_p05": observation_p05,
        "mean_valid_observations_p95": observation_p95,
    }
    return fig, city, block_groups, scenes, stats


def caption_text() -> str:
    return (
        "Supplementary Fig. 7 | Landsat land-surface-temperature input "
        "support across cities and block groups. a, Four city-level "
        "diagnostics for the fixed 50-city analysis frame: mean valid-pixel "
        "coverage within retained block groups, the percentage of the 34,575-unit "
        "fully contained block-group support retained in the 34,566-unit surface-"
        "temperature frame, and deduplicated Landsat 8 and Landsat 9 "
        "city-scene records from June-August 2020-2024. Cities are ordered "
        "from lower to higher valid-pixel coverage and split into two columns. "
        "Printed values retain their native units; shading is the within-"
        "column percentile and is not a common-unit score. b, Empirical "
        "cumulative distributions for the 34,566 fixed analysis units. "
        "The valid-pixel panel enlarges the 90-100% range while reporting the "
        "fraction below that range: 88.9% of units had complete valid-pixel "
        "coverage and the fifth percentile was 97.4%. Repeated valid "
        "observations per LST pixel had a median of 37.1 and a fifth-to-95th "
        "percentile range of 18.6-71.7. The inventory contained 4,434 unique "
        "city-scene records. These diagnostics establish input support but do not "
        "remove retrieval error or temporal mismatch. LST describes remotely sensed "
        "summer surface thermal context, not air temperature, personal exposure or "
        "a health outcome."
    )


def save_outputs(
    fig: plt.Figure,
    city: pd.DataFrame,
    block_groups: pd.DataFrame,
    scenes: pd.DataFrame,
    stats: dict,
) -> None:
    fig.savefig(CANONICAL_STEM.with_suffix(".png"), dpi=600)
    fig.savefig(CANONICAL_STEM.with_suffix(".pdf"))
    fig.savefig(CANONICAL_STEM.with_suffix(".svg"))
    fig.savefig(
        CANONICAL_STEM.with_suffix(".tif"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )

    ImageOps.grayscale(
        Image.open(CANONICAL_STEM.with_suffix(".png")).convert("RGB")
    ).save(QA_DIR / "Supplementary_Fig7_grayscale.png")

    source_exports = {
        "Supplementary_Fig7a_city_landsat_support_matrix.csv": city,
        "Supplementary_Fig7b_block_group_observation_support.csv": block_groups,
        "Supplementary_Fig7_scene_inventory_unique_records.csv": scenes[
            [
                "city",
                "item_id",
                "datetime_utc",
                "platform",
                "cloud_cover",
                "wrs_path",
                "wrs_row",
            ]
        ],
    }
    for name, frame in source_exports.items():
        frame.to_csv(SOURCE_DIR / name, index=False)

    caption = caption_text()
    (TEXT_DIR / "Supplementary_Fig7_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )

    provenance = {
        "figure": "Supplementary Fig. 7",
        "analysis_role": "Landsat support audit",
        "figure_contract": {
            "core_claim": (
                "Landsat summer LST support is broad across all 50 cities, "
                "with the small lower-coverage tail shown explicitly."
            ),
            "panel_a_role": "city-level support and sensor inventory",
            "panel_b_role": "block-group coverage and repeat-observation distributions",
            "principal_limitation": (
                "Cloud and scene availability could drive the apparent "
                "canopy-LST association or selectively remove neighbourhoods."
            ),
            "response": (
                "The figure reports all cities, the retained-unit fraction and "
                "the full lower-tail distributions without converting support "
                "metrics into a composite quality score."
            ),
        },
        "input_sha256": {
            str(path): sha256(path) for path in source_paths().values()
        },
        "stats": stats,
        "caption": caption,
    }
    provenance["minimum_font_pt"] = 5.8
    provenance["status"] = "CANDIDATE"
    (QA_DIR / "Supplementary_Fig7_build_qa.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    fig, city, block_groups, scenes, stats = build_figure()
    save_outputs(fig, city, block_groups, scenes, stats)
    plt.close(fig)
    print(
        json.dumps(
            {
                "figure": "Supplementary Fig. 7",
                "analysis_role": "Landsat support audit",
                "stats": stats,
                "outputs": {
                    str(CANONICAL_STEM.with_suffix(ext)): (
                        CANONICAL_STEM.with_suffix(ext).stat().st_size
                    )
                    for ext in [".png", ".pdf", ".svg", ".tif"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
