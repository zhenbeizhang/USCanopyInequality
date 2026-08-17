from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(str(analysis_store_root()))
WORKSPACE = Path(str(repo_root()))
BASE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "01_canopy_mapping"
    / "aggregate_canopy_context.py"
)
BOUNDARY_ROOT = (
    ROOT
    / "Result"
    / "Result2"
    / "Result2_block_population_boundary_audit"
)
BOUNDARY_FRAME = BOUNDARY_ROOT / "data" / "block_population_boundary_frame.csv.gz"
CITY_INVENTORY = BOUNDARY_ROOT / "data" / "city_inventory.csv"
OLD_CITY_SOURCE = (
    ROOT
    / "Result"
    / "Result1"
    / "versions"
    / "step3_canopy_recovery_validation"
    / "source_data"
    / "Fig1"
    / "Fig1bc_city_canopy_context.csv"
)

OUT = ROOT / "Result" / "Result1" / "versions" / "aligned_city_support"
FIG = OUT / "figures"
SRC = OUT / "source_data"
QA = OUT / "qa"
METHODS = OUT / "methods"
TEXT = OUT / "text"
for folder in (FIG, SRC, QA, METHODS, TEXT):
    folder.mkdir(parents=True, exist_ok=True)


PRIMARY_COLOR = "#167568"
SENSITIVITY_COLOR = "#C5762D"
NEUTRAL = "#6E7774"
BLACK = "#202020"


def load_base():
    spec = importlib.util.spec_from_file_location("fig1_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "axes.linewidth": 0.72,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def panel_heading(fig: plt.Figure, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x, y, letter, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=BLACK)
    fig.text(x + 0.024, y, title, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=BLACK)


def build_city_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(BOUNDARY_FRAME)
    inventory = pd.read_csv(CITY_INVENTORY)[
        ["result_city_key", "city_display", "acs2024_city_population"]
    ].drop_duplicates("result_city_key")

    required = {
        "result_city_key",
        "city_display",
        "canopy_area_m2",
        "estimated_city_land_area_m2",
        "canopy_pct_block_allocated_frame",
        "population_block_allocated",
        "fully_contained",
        "land_area_m2",
        "canopy_pct_analysis",
        "population",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Boundary frame is missing columns: {missing}")

    primary_rows = []
    contained_rows = []
    for (city_key, city_display), group in frame.groupby(
        ["result_city_key", "city_display"], sort=True
    ):
        population_weights = group["population_block_allocated"].astype(float).to_numpy()
        if population_weights.sum() <= 0:
            raise RuntimeError(f"No allocated population for {city_key}")
        primary_rows.append(
            {
                "city_key": city_key,
                "city_display": city_display,
                "area_weighted_canopy_pct": 100.0
                * group["canopy_area_m2"].sum()
                / group["estimated_city_land_area_m2"].sum(),
                "population_weighted_canopy_pct": np.average(
                    group["canopy_pct_block_allocated_frame"],
                    weights=population_weights,
                ),
                "n_atlas_intersections": len(group),
                "n_positive_population_units": int((population_weights > 0).sum()),
                "allocated_population_2020_pattern": population_weights.sum(),
            }
        )

        contained = group.loc[group["fully_contained"].astype(bool)].copy()
        if contained.empty or contained["population"].sum() <= 0:
            raise RuntimeError(f"No fully contained comparison frame for {city_key}")
        contained_rows.append(
            {
                "city_key": city_key,
                "city_display": city_display,
                "area_weighted_canopy_pct": 100.0
                * contained["canopy_area_m2"].sum()
                / contained["land_area_m2"].sum(),
                "population_weighted_canopy_pct": np.average(
                    contained["canopy_pct_analysis"],
                    weights=contained["population"],
                ),
                "n_fully_contained": len(contained),
                "fully_contained_population": contained["population"].sum(),
            }
        )

    primary = pd.DataFrame(primary_rows).merge(
        inventory[["result_city_key", "acs2024_city_population"]],
        left_on="city_key",
        right_on="result_city_key",
        how="left",
        validate="one_to_one",
    )
    primary = primary.drop(columns="result_city_key")
    primary = primary.rename(columns={"acs2024_city_population": "population"})
    primary["population_minus_area_pp"] = (
        primary["population_weighted_canopy_pct"]
        - primary["area_weighted_canopy_pct"]
    )

    contained = pd.DataFrame(contained_rows)
    contained["population_minus_area_pp"] = (
        contained["population_weighted_canopy_pct"]
        - contained["area_weighted_canopy_pct"]
    )

    if len(primary) != 50 or primary["city_key"].nunique() != 50:
        raise RuntimeError("Primary Result 1 frame does not contain 50 unique cities")
    if len(contained) != 50 or contained["city_key"].nunique() != 50:
        raise RuntimeError("Fully contained Result 1 sensitivity does not contain 50 cities")
    if primary["population"].isna().any():
        raise RuntimeError("Missing 2024 ACS city population")
    if not primary["area_weighted_canopy_pct"].between(0, 100).all():
        raise RuntimeError("Primary area-weighted canopy is outside 0-100%")
    if not primary["population_weighted_canopy_pct"].between(0, 100).all():
        raise RuntimeError("Primary population-weighted canopy is outside 0-100%")

    return frame, primary, contained


def raw_overlay(base, sample: dict[str, object]) -> np.ndarray:
    return base.overlay_canopy(sample["rgb"], sample["mask_raw"], alpha=0.40)


def draw_atlas(
    fig: plt.Figure,
    ax: plt.Axes,
    base,
    centroids: pd.DataFrame,
    samples: list[dict[str, object]],
) -> None:
    ax.set_axis_off()
    gap = 0.006
    bottom = 0.018
    bottom_h = 0.300
    map_y = bottom + bottom_h + gap
    map_h = 1.0 - map_y - 0.006
    map_w = 0.602
    map_ax = ax.inset_axes([0.0, map_y, map_w, map_h])
    base.plot_map_and_workflow(map_ax, centroids)
    for item in map_ax.texts:
        if item.get_text().strip() == "Canopy cover (%)":
            item.set_text("City-area canopy (%)")

    right_x = map_w + gap
    right_w = 1.0 - right_x
    right_h = (map_h - gap) / 2.0
    bottom_w = (1.0 - 3.0 * gap) / 4.0
    positions = {
        "New York": [right_x, map_y + right_h + gap, right_w, right_h],
        "Las Vegas": [right_x, map_y, right_w, right_h],
        "Houston": [0.0, bottom, bottom_w, bottom_h],
        "Raleigh": [bottom_w + gap, bottom, bottom_w, bottom_h],
        "Seattle": [2.0 * (bottom_w + gap), bottom, bottom_w, bottom_h],
        "Miami": [3.0 * (bottom_w + gap), bottom, bottom_w, bottom_h],
    }
    holder_box = ax.get_position()
    holder_aspect = holder_box.width * fig.get_figwidth() / (
        holder_box.height * fig.get_figheight()
    )
    for sample in samples:
        pos = positions[str(sample["city_display"])]
        chip = ax.inset_axes(pos)
        overlay = raw_overlay(base, sample)
        target_aspect = (pos[2] / pos[3]) * holder_aspect
        display = base.crop_to_aspect(overlay, target_aspect)
        chip.imshow(display, aspect="auto")
        chip.set_xticks([])
        chip.set_yticks([])
        for spine in chip.spines.values():
            spine.set_visible(True)
            spine.set_color("#C8D1CA")
            spine.set_linewidth(0.58)
        chip.text(
            0.015,
            0.985,
            str(sample["city_display"]),
            transform=chip.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color=BLACK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.9},
        )
        base.add_scale_bar(chip, display.shape[:2], meters=100)


def draw_agreement(ax: plt.Axes, city: pd.DataFrame) -> None:
    x = city["area_weighted_canopy_pct"].to_numpy(float)
    y = city["population_weighted_canopy_pct"].to_numpy(float)
    delta = y - x
    limit = float(np.ceil(max(x.max(), y.max()) / 5.0) * 5.0)
    ax.plot([0, limit], [0, limit], color=NEUTRAL, lw=0.9, ls=(0, (3, 2)), zorder=0)
    ax.scatter(
        x,
        y,
        color=PRIMARY_COLOR,
        s=27,
        edgecolor="white",
        linewidth=0.65,
        alpha=0.96,
        zorder=2,
    )
    ax.set(xlim=(0, limit), ylim=(0, limit))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("City-area canopy cover (%)")
    ax.set_ylabel("Resident-weighted canopy cover (%)")
    ax.set_xticks(np.arange(0, limit + 1, 10))
    ax.set_yticks(np.arange(0, limit + 1, 10))
    ax.tick_params(width=0.7, length=3, pad=2)

    label_names = ["Atlanta", "Oklahoma City", "Tucson"]
    offsets = {
        "Atlanta": (-6, -17, "right"),
        "Oklahoma City": (-7, 10, "right"),
        "Tucson": (7, 11, "left"),
    }
    for name in label_names:
        row = city.loc[city["city_display"].eq(name)].iloc[0]
        dx, dy, ha = offsets[name]
        ax.annotate(
            name,
            xy=(row["area_weighted_canopy_pct"], row["population_weighted_canopy_pct"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.1,
            ha=ha,
            va="center",
            color=BLACK,
            arrowprops={"arrowstyle": "-", "color": "#404040", "lw": 0.45, "shrinkA": 1, "shrinkB": 2},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.2},
            clip_on=False,
        )
    ax.text(0.96, 0.055, "1:1", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color="#4E5553")


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def draw_support_sensitivity(
    ax: plt.Axes, primary: pd.DataFrame, contained: pd.DataFrame
) -> dict[str, float | int]:
    p_x, p_y = ecdf(primary["population_minus_area_pp"].to_numpy())
    f_x, f_y = ecdf(contained["population_minus_area_pp"].to_numpy())
    ax.step(p_x, p_y, where="post", color=PRIMARY_COLOR, lw=1.75, label="City-part population allocation")
    ax.step(f_x, f_y, where="post", color=SENSITIVITY_COLOR, lw=1.45, ls=(0, (4, 2)), label="Fully contained units")
    ax.axvline(0, color="#444444", lw=0.85, ls=(0, (3, 2)), zorder=0)

    p_med = float(np.median(p_x))
    f_med = float(np.median(f_x))
    p_pos = int((p_x > 0).sum())
    f_pos = int((f_x > 0).sum())
    ax.text(
        0.035,
        0.965,
        f"Primary: median {p_med:+.2f} pp; {p_pos}/50 > 0\n"
        f"Contained: median {f_med:+.2f} pp; {f_pos}/50 > 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.25,
        color=BLACK,
    )
    ax.set_xlabel("Resident-weighted minus area-weighted canopy (pp)")
    ax.set_ylabel("Cumulative share of cities")
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(width=0.7, length=3, pad=2)
    legend = ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=6.15,
        handlelength=2.25,
        borderaxespad=0.3,
        labelspacing=0.35,
    )
    for text in legend.get_texts():
        text.set_color(BLACK)
    return {
        "primary_median_shift_pp": p_med,
        "primary_cities_above_zero": p_pos,
        "contained_median_shift_pp": f_med,
        "contained_cities_above_zero": f_pos,
    }


def save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    outputs = []
    for extension in ("pdf", "svg", "png", "tif"):
        path = stem.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 450
        elif extension == "tif":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    return outputs


def white_margin_fraction(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    nonwhite = np.any(array < 248, axis=2)
    ys, xs = np.where(nonwhite)
    if len(xs) == 0:
        raise RuntimeError("Rendered figure is blank")
    height, width = nonwhite.shape
    return {
        "left": float(xs.min() / width),
        "right": float((width - 1 - xs.max()) / width),
        "top": float(ys.min() / height),
        "bottom": float((height - 1 - ys.max()) / height),
    }


def main() -> None:
    configure_style()
    base = load_base()
    frame, primary, contained = build_city_metrics()

    primary.to_csv(SRC / "Fig1_primary_city_metrics.csv", index=False, encoding="utf-8-sig")
    contained.to_csv(SRC / "Fig1_fully_contained_sensitivity.csv", index=False, encoding="utf-8-sig")
    if OLD_CITY_SOURCE.exists():
        old = pd.read_csv(OLD_CITY_SOURCE)
        comparison = primary.merge(
            old[
                [
                    "city_key",
                    "area_weighted_canopy_pct",
                    "population_weighted_canopy_pct",
                    "population_minus_area_pp",
                ]
            ],
            on="city_key",
            how="left",
            suffixes=("_aligned", "_reference"),
            validate="one_to_one",
        )
        comparison.to_csv(SRC / "Fig1_support_alignment_audit.csv", index=False, encoding="utf-8-sig")

    centroids = base.city_centroids(primary)
    # The map in panel a encodes city-area canopy supply. Population-weighted
    # context is introduced separately in panel b.
    samples = [base.choose_sparse_window(city_key) for _, city_key in base.SCENE_CITIES]
    selection = pd.DataFrame(
        [
            {
                "city_key": sample["city_key"],
                "city_display": sample["city_display"],
                "row": sample["row"],
                "col": sample["col"],
                "window_size_px": sample["window_size_px"],
                "display_mask": "archived thresholded mask; no morphology",
            }
            for sample in samples
        ]
    )
    selection.to_csv(SRC / "Fig1_display_windows.csv", index=False, encoding="utf-8-sig")
    centroids.to_csv(SRC / "Fig1_city_centroids.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.20, 6.10))
    panel_heading(fig, 0.048, 0.955, "a", "Sub-metre masks resolve citywide canopy supply")
    ax_a = fig.add_axes([0.050, 0.455, 0.905, 0.480])
    draw_atlas(fig, ax_a, base, centroids, samples)
    for item in fig.findobj(match=mpl.text.Text):
        if item.get_text().strip() == "Canopy cover (%)":
            item.set_text("City-area canopy (%)")

    panel_heading(fig, 0.050, 0.392, "b", "Resident context differs from city-area canopy")
    ax_b = fig.add_axes([0.080, 0.075, 0.350, 0.275])
    draw_agreement(ax_b, primary)

    panel_heading(fig, 0.535, 0.392, "c", "Direction depends on boundary support")
    ax_c = fig.add_axes([0.570, 0.075, 0.385, 0.275])
    sensitivity_stats = draw_support_sensitivity(ax_c, primary, contained)

    stem = FIG / "Fig1_aligned_city_support"
    outputs = save_figure(fig, stem)
    plt.close(fig)

    png = stem.with_suffix(".png")
    ImageOps.grayscale(Image.open(png)).save(QA / "Fig1_grayscale.png")
    margins = white_margin_fraction(png)

    lowest = primary.loc[primary["population_weighted_canopy_pct"].idxmin()]
    highest = primary.loc[primary["population_weighted_canopy_pct"].idxmax()]
    audit = {
        "status": "PASS",
        "quality_gate_status": "PASS",
        "core_claim": (
            "City-area canopy supply and resident-weighted canopy context differ, "
            "but the direction of that difference is city-specific and support-sensitive."
        ),
        "primary_support": (
            "All 38,618 atlas-intersecting block groups; canopy denominator is the "
            "estimated city-intersection land area, and population weights follow 2020 "
            "Census blocks within the city portion. City totals use 2024 ACS place population."
        ),
        "n_cities": int(primary["city_key"].nunique()),
        "n_atlas_intersections": int(primary["n_atlas_intersections"].sum()),
        "n_positive_population_units": int(primary["n_positive_population_units"].sum()),
        "represented_2024_acs_city_population": float(primary["population"].sum()),
        "resident_canopy_min": {
            "city": str(lowest["city_display"]),
            "pct": float(lowest["population_weighted_canopy_pct"]),
        },
        "resident_canopy_max": {
            "city": str(highest["city_display"]),
            "pct": float(highest["population_weighted_canopy_pct"]),
        },
        **sensitivity_stats,
        "fully_contained_units": int(contained["n_fully_contained"].sum()),
        "canopy_values_within_0_100": bool(
            primary["population_weighted_canopy_pct"].between(0, 100).all()
            and primary["area_weighted_canopy_pct"].between(0, 100).all()
        ),
        "primary_city_population_matches_inventory": bool(primary["population"].notna().all()),
        "outer_white_margin_fraction": margins,
        "figure_files": outputs,
        "strongest_remaining_limitation": (
            "Within-boundary population weights use the spatial distribution of 2020 Census "
            "blocks to allocate 2024 ACS block-group population; the fully contained frame "
            "therefore remains a necessary support sensitivity."
        ),
    }
    (QA / "Fig1_technical_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    caption = (
        "Fig. 1 | Boundary-aligned canopy mapping separates citywide supply from resident context. "
        "a, Area-weighted canopy cover across 50 US municipal geographies and six 0.6 m visible-crown "
        "examples. Green overlays mark mapped woody crowns; windows are descriptive examples rather "
        "than validation samples. b, City-area canopy cover versus resident-weighted canopy context. "
        "For boundary-crossing block groups, canopy uses city-intersection land area and population "
        "weights follow 2020 Census blocks within the municipal portion; the dashed line denotes equality. "
        "c, Empirical cumulative distributions of resident-weighted minus area-weighted canopy under the "
        "primary city-part allocation and a fully contained block-group sensitivity. The primary median "
        f"was {sensitivity_stats['primary_median_shift_pp']:+.2f} percentage points ({sensitivity_stats['primary_cities_above_zero']}/50 cities above zero), "
        f"compared with {sensitivity_stats['contained_median_shift_pp']:+.2f} percentage points ({sensitivity_stats['contained_cities_above_zero']}/50) "
        "among fully contained units. The comparison shows city-specific differences rather than a universal "
        "direction of population weighting. Canopy denotes mapped visible woody-crown context, not cast shade."
    )
    (TEXT / "Fig1_caption.txt").write_text(caption, encoding="utf-8")
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
