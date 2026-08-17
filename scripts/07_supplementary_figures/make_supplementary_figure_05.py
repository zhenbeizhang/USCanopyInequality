from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image


PROJECT = analysis_store_root()
RESULT = PROJECT / "Result"
FRAME = (
    RESULT
    / "Result2"
    / "Result2_block_population_boundary_audit"
    / "data"
    / "result2_fully_contained_primary_complete_case_frame.csv.gz"
)
OLD_MATRIX = (
    RESULT
    / "Result2"
    / "Result2_module1_boundary_inference_audit"
    / "figure"
    / "source_data"
    / "Fig2a_city_social_gap_matrix.csv"
)
OUT = (
    RESULT
    / "Supplementary figure and table"
    / "Supplementary Fig. 5"
    / "versions"
    / "complete_case_city_atlas"
)
FIGURES = OUT / "figures"
SOURCE = OUT / "source_data"
QA = OUT / "qa"
METHODS = OUT / "methods"
TEXT = OUT / "text"

PREDICTORS = [
    ("Low income", "median_household_income_usd", -1.0),
    ("Poverty", "poverty_rate_pct", 1.0),
    ("Older age", "median_age", 1.0),
    ("Black", "black_alone_pct", 1.0),
    ("Hispanic/Latino", "hispanic_or_latino_pct", 1.0),
    ("Asian", "asian_alone_pct", 1.0),
    ("Not White alone", "people_of_color_pct", 1.0),
    ("Renters", "renter_occupied_pct", 1.0),
    ("No vehicle", "no_vehicle_households_pct", 1.0),
    ("Public transit", "public_transport_commute_pct", 1.0),
    ("Density", "log_pop_density", 1.0),
]
VARIABLES = [item[0] for item in PREDICTORS]
DISPLAY = [
    "Low income",
    "Poverty",
    "Median age",
    "Black",
    "Hispanic/Latino",
    "Asian",
    "Not White alone",
    "Renters",
    "No vehicle",
    "Public transit",
    "Density",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.6,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    x = values.to_numpy(float)
    w = weights.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[valid], weights=w[valid])) if valid.any() else math.nan


def compute_matrix(frame: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    quintile_rows = []
    label_map = labels.set_index("city_key")["city_display"].to_dict()
    for city_key, city in frame.groupby("result_city_key", sort=True):
        row = {
            "city_key": city_key,
            "city_display": label_map[city_key],
            "n_block_groups": len(city),
            "represented_population": float(city["population"].sum()),
            "population_weighted_canopy_pct": weighted_mean(city["canopy_reference_0p6m"], city["population"]),
        }
        for label, column, direction in PREDICTORS:
            oriented = city[column].astype(float) * direction
            ranks = oriented.rank(method="first")
            quintiles = pd.qcut(ranks, 5, labels=False).astype(int) + 1
            values = {}
            for quintile in range(1, 6):
                subset = city.loc[quintiles.eq(quintile)]
                canopy = weighted_mean(subset["canopy_reference_0p6m"], subset["population"])
                values[quintile] = canopy
                quintile_rows.append(
                    {
                        "city_key": city_key,
                        "city_display": label_map[city_key],
                        "predictor": label,
                        "quintile": quintile,
                        "n_block_groups": len(subset),
                        "represented_population": float(subset["population"].sum()),
                        "population_weighted_canopy_pct": canopy,
                    }
                )
            row[label] = values[5] - values[1]
        row["mean_high_minus_low_contrast_pp"] = float(np.mean([row[label] for label in VARIABLES]))
        rows.append(row)
    matrix = pd.DataFrame(rows).sort_values(
        ["mean_high_minus_low_contrast_pp", "city_display"], ascending=[True, True]
    ).reset_index(drop=True)
    matrix["atlas_rank"] = np.arange(1, len(matrix) + 1)
    return matrix, pd.DataFrame(quintile_rows)


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

    frame = pd.read_csv(FRAME, low_memory=False)
    labels = pd.read_csv(OLD_MATRIX, usecols=["city_key", "city_display"]).drop_duplicates()
    if len(frame) != 30826 or frame["result_city_key"].nunique() != 50:
        raise RuntimeError("Latest fully contained complete-case frame is not 30,826 block groups in 50 cities")
    required = ["population", "canopy_reference_0p6m", *[item[1] for item in PREDICTORS]]
    if frame[required].isna().any().any():
        raise RuntimeError("Complete-case atlas input contains missing values")
    matrix, quintiles = compute_matrix(frame, labels)

    old = pd.read_csv(OLD_MATRIX)[["city_key", *VARIABLES]]
    comparison = matrix[["city_key", *VARIABLES]].merge(old, on="city_key", suffixes=("_new", "_old"), validate="one_to_one")
    sign_pairs = []
    absolute_changes = []
    for variable in VARIABLES:
        new = comparison[f"{variable}_new"].to_numpy(float)
        previous = comparison[f"{variable}_old"].to_numpy(float)
        sign_pairs.extend((np.sign(new) == np.sign(previous)).tolist())
        absolute_changes.extend(np.abs(new - previous).tolist())

    values = matrix[VARIABLES].to_numpy(float)
    max_abs = float(np.nanmax(np.abs(values)))
    colour_limit = max(20.0, 5.0 * math.ceil(max_abs / 5.0))
    cmap = LinearSegmentedColormap.from_list(
        "canopy_gap",
        ["#96382E", "#D57B5D", "#F4F1EC", "#B7D0D9", "#5C9FB6"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-colour_limit, vcenter=0, vmax=colour_limit)

    fig = plt.figure(figsize=(7.20, 6.45), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 0.040],
        left=0.112,
        right=0.992,
        top=0.815,
        bottom=0.080,
        wspace=0.27,
        hspace=0.16,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for panel, (ax, part) in enumerate(zip(axes, [matrix.iloc[:25], matrix.iloc[25:]])):
        ax.imshow(part[VARIABLES].to_numpy(float), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(VARIABLES)), labels=DISPLAY)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=3.0, labelsize=5.8)
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_rotation_mode("anchor")
            label.set_ha("left")
            label.set_va("center")
        city_labels = [
            name.replace("Louisville Jefferson County", "Louisville").replace("Nashville-Davidson", "Nashville")
            for name in part["city_display"]
        ]
        ax.set_yticks(np.arange(len(part)), labels=city_labels)
        ax.tick_params(axis="y", length=0, pad=3.0, labelsize=5.8)
        ax.set_xticks(np.arange(-0.5, len(VARIABLES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(part), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.42)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#4A4A4A")
            spine.set_linewidth(0.55)
        position = ax.get_position()
        fig.text(
            position.x0,
            0.942,
            f"{'a' if panel == 0 else 'b'}  {'Larger average deficits' if panel == 0 else 'Smaller average deficits'}",
            ha="left",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color="black",
        )

    cax = fig.add_subplot(gs[1, :])
    ticks = np.arange(-colour_limit, colour_limit + 0.1, 10.0)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
    colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1.5)
    colorbar.outline.set_linewidth(0.55)
    colorbar.set_label(
        "Highest minus lowest within-city quintile canopy (percentage points)",
        fontsize=6.2,
        labelpad=2.5,
    )

    overlaps = text_overlap_audit(fig)
    outputs = {
        "png": FIGURES / "Supplementary_Fig5.png",
        "pdf": FIGURES / "Supplementary_Fig5.pdf",
        "svg": FIGURES / "Supplementary_Fig5.svg",
        "tif": FIGURES / "Supplementary_Fig5.tif",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["png"], dpi=450, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["tif"], dpi=450, bbox_inches="tight", pad_inches=0.035, pil_kwargs={"compression": "tiff_lzw"})
    image = np.asarray(Image.open(outputs["png"]).convert("RGB"))
    grayscale = Image.fromarray(image).convert("L")
    grayscale.save(QA / "Supplementary_Fig5_grayscale.png")
    Image.fromarray(image).resize(
        (max(1, image.shape[1] * 300 // 450), max(1, image.shape[0] * 300 // 450)),
        Image.Resampling.LANCZOS,
    ).save(QA / "Supplementary_Fig5_300dpi_preview.png")
    minimum_font = min(
        float(text.get_fontsize())
        for text in fig.findobj(mpl.text.Text)
        if text.get_visible() and text.get_text().strip()
    )
    plt.close(fig)

    matrix.to_csv(SOURCE / "Supplementary_Fig5_city_social_gap_atlas.csv", index=False)
    quintiles.to_csv(SOURCE / "Supplementary_Fig5_city_social_quintiles.csv.gz", index=False, compression="gzip")
    comparison.to_csv(SOURCE / "Supplementary_Fig5_old_new_sample_comparison.csv", index=False)
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)

    qa = {
        "status": "PASS",
        "quality_gate_status": "PASS",
        "analysis_units": len(frame),
        "cities": frame["result_city_key"].nunique(),
        "variables": len(VARIABLES),
        "finite_city_variable_cells": int(np.isfinite(values).sum()),
        "colour_limit_pp": colour_limit,
        "old_new_sign_retention": float(np.mean(sign_pairs)),
        "median_absolute_change_from_old_sample_pp": float(np.median(absolute_changes)),
        "minimum_font_pt": minimum_font,
        "text_text_overlap_count": len(overlaps),
        "text_text_overlaps": overlaps,
        "exports_exist": all(path.exists() and path.stat().st_size > 10_000 for path in outputs.values()),
        "scientific_claim": "The national direction of social-canopy disadvantage coexists with substantial city and attribute heterogeneity on a common 30,826-unit frame.",
        "remaining_limitations": [
            "The heatmap shows descriptive highest-versus-lowest quintile contrasts, not independent regression effects.",
            "City ordering is based on the mean of 11 descriptive contrasts and is not a vulnerability or policy ranking.",
            "Density is an urban-form context and should not be interpreted as a social group.",
        ],
    }
    qa["status"] = "PASS" if all(
        [
            qa["analysis_units"] == 30826,
            qa["cities"] == 50,
            qa["variables"] == 11,
            qa["finite_city_variable_cells"] == 550,
            qa["minimum_font_pt"] >= 5.8,
            qa["text_text_overlap_count"] == 0,
            qa["exports_exist"],
        ]
    ) else "FAIL"
    (QA / "Supplementary_Fig5_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    caption = (
        "Supplementary Fig. 5 | City heterogeneity in descriptive social-canopy contrasts on a common complete-case frame. "
        "a,b, Population-weighted residential canopy in the highest oriented within-city quintile minus that in the lowest quintile for 11 prespecified attributes across 50 cities and the same 30,826 fully contained block groups. Household income is reversed so that the highest oriented quintile denotes lower income; all other columns compare higher with lower values or group concentrations. Age denotes median age, Hispanic denotes Hispanic/Latino concentration and Not White alone denotes 100 minus the ACS White-alone racial share. Cities are ordered by their mean contrast across the 11 columns and split into two panels only for legibility. Negative values indicate less canopy in the named higher-disadvantage or higher-concentration context, whereas positive values indicate more. These are descriptive within-city contrasts rather than mutually adjusted demographic effects. Density is included as an urban-form context, and the city ordering is not a vulnerability or allocation ranking."
    )
    (TEXT / "Supplementary_Fig5_caption.txt").write_text(caption.replace("Older age", "Median age"), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
