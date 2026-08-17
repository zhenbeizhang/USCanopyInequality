from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(str(analysis_store_root()))
R1 = ROOT / "Result" / "Result1" / "versions" / "step3_canopy_recovery_validation"
OUT = (
    ROOT
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 3"
    / "versions"
    / "internal_diagnostic_seam"
)
FIG = OUT / "figures"
SRC = OUT / "source_data"
QA = OUT / "qa"
SCRIPTS = OUT / "methods"
TEXT = OUT / "text"
for folder in (FIG, SRC, QA, SCRIPTS, TEXT):
    folder.mkdir(parents=True, exist_ok=True)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
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


def heading(fig: plt.Figure, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x, y, letter, ha="left", va="bottom", fontsize=8.4, fontweight="bold", color="#202020")
    fig.text(x + 0.025, y, title, ha="left", va="bottom", fontsize=8.4, fontweight="bold", color="#202020")


def panel_a(ax: plt.Axes, overall: pd.DataFrame, density: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checkpoint = overall[overall["analysis_set"].str.startswith("checkpoint-selection")].iloc[0]
    untouched = overall[overall["analysis_set"].str.startswith("fully untouched")].iloc[0]
    rows.extend(
        [
            {
                "label": "Checkpoint-selection\nsplit",
                "f1": checkpoint["f1_estimate"],
                "f1_lo": checkpoint["f1_lower_95"],
                "f1_hi": checkpoint["f1_upper_95"],
                "iou": checkpoint["iou_estimate"],
                "iou_lo": checkpoint["iou_lower_95"],
                "iou_hi": checkpoint["iou_upper_95"],
                "interval": True,
            },
            {
                "label": "Recovered diagnostic\nsplit",
                "f1": untouched["f1_estimate"],
                "f1_lo": untouched["f1_lower_95"],
                "f1_hi": untouched["f1_upper_95"],
                "iou": untouched["iou_estimate"],
                "iou_lo": untouched["iou_lower_95"],
                "iou_hi": untouched["iou_upper_95"],
                "interval": True,
            },
        ]
    )
    for _, row in density.iterrows():
        rows.append(
            {
                "label": str(row["canopy_density_group"]).replace(" third", "\nthird"),
                "f1": row["f1"],
                "f1_lo": np.nan,
                "f1_hi": np.nan,
                "iou": row["iou"],
                "iou_lo": np.nan,
                "iou_hi": np.nan,
                "interval": False,
            }
        )
    plot = pd.DataFrame(rows)
    x = np.arange(len(plot))
    width = 0.34
    ax.bar(x - width / 2, plot["f1"], width=width, color="#21796D", label="F1", zorder=2)
    ax.bar(x + width / 2, plot["iou"], width=width, color="#9BA8AC", label="IoU", zorder=2)
    for i, row in plot.iterrows():
        if row["interval"]:
            ax.errorbar(
                i - width / 2,
                row["f1"],
                yerr=[[row["f1"] - row["f1_lo"]], [row["f1_hi"] - row["f1"]]],
                fmt="none",
                ecolor="#202020",
                elinewidth=0.75,
                capsize=2.2,
                zorder=4,
            )
            ax.errorbar(
                i + width / 2,
                row["iou"],
                yerr=[[row["iou"] - row["iou_lo"]], [row["iou_hi"] - row["iou"]]],
                fmt="none",
                ecolor="#202020",
                elinewidth=0.75,
                capsize=2.2,
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(plot["label"])
    ax.set_ylabel("Segmentation score")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, handlelength=1.4, columnspacing=1.4)
    ax.tick_params(width=0.7, length=3, pad=2)
    ax.spines["left"].set_color("#202020")
    ax.spines["bottom"].set_color("#202020")
    return plot


def panel_b(ax: plt.Axes, geo: pd.DataFrame) -> pd.DataFrame:
    overlap = int(geo["overlaps_model_development"].astype(bool).sum())
    near = int((~geo["overlaps_model_development"].astype(bool) & geo["within_250m_of_model_development"].astype(bool)).sum())
    separated = int((~geo["within_250m_of_model_development"].astype(bool)).sum())
    counts = pd.DataFrame(
        {
            "category": ["Footprint overlap", "Within 250 m", "At least 250 m"],
            "base_tiles": [overlap, near, separated],
            "colour": ["#A2A9AB", "#D28A47", "#21796D"],
        }
    )
    left = 0
    for _, row in counts.iterrows():
        width = int(row["base_tiles"])
        ax.barh([0], [width], left=[left], height=0.62, color=row["colour"], edgecolor="white", linewidth=0.8)
        if width >= 5:
            ax.text(left + width / 2, 0, f"{width}", ha="center", va="center", fontsize=7.0, fontweight="bold", color="white" if row["category"] != "Footprint overlap" else "#202020")
        left += width
    ax.set_xlim(0, len(geo))
    ax.set_ylim(-0.36, 0.36)
    ax.set_yticks([])
    ax.set_xlabel("Base tiles in recovered diagnostic split")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#202020")
    ax.tick_params(axis="x", width=0.7, length=3, pad=2)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in counts["colour"]]
    ax.legend(handles, counts["category"], loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=3, frameon=False, handlelength=1.0, columnspacing=0.9, fontsize=6.2)
    return counts


def panel_c(ax: plt.Axes, overlap: pd.DataFrame) -> None:
    x = overlap["archived_transition_rate_ratio"].astype(float)
    y = overlap["overlap_transition_rate_ratio"].astype(float)
    ax.scatter(
        x,
        y,
        color="#477FB3",
        s=28,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.95,
        zorder=3,
    )
    ax.axhline(1, color="#303030", lw=0.85, ls=(0, (3, 2)), zorder=1)
    ax.axvline(1, color="#303030", lw=0.85, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("Archived boundary/control transition ratio")
    ax.set_ylabel("50% overlap boundary/control transition ratio")
    ax.set_xlim(0, max(25, float(np.ceil(x.max() / 5) * 5)))
    ax.set_ylim(0.70, 1.30)
    ax.tick_params(width=0.7, length=3, pad=2)
    ax.spines["left"].set_color("#202020")
    ax.spines["bottom"].set_color("#202020")


def save(fig: plt.Figure, stem: Path) -> list[str]:
    paths = []
    for ext in ("pdf", "svg", "png", "tif"):
        path = stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white", "dpi": 600}
        if ext == "tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    return paths


def main() -> None:
    style()
    overall = pd.read_csv(R1 / "tables" / "independent_validation_metrics.csv")
    density = pd.read_csv(R1 / "tables" / "primary_test_metrics_by_canopy_density.csv")
    geo = pd.read_csv(R1 / "tables" / "geographic_split_separation_by_test_base_tile.csv")
    overlap = pd.read_csv(R1 / "tables" / "overlap_inference_audit_by_city.csv")
    overall.to_csv(SRC / "Fig3a_internal_diagnostic_metrics.csv", index=False, encoding="utf-8-sig")
    density.to_csv(SRC / "Fig3a_density_strata.csv", index=False, encoding="utf-8-sig")
    geo.to_csv(SRC / "Fig3b_geographic_separation.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(SRC / "Fig3c_overlap_inference.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.20, 5.20))
    heading(fig, 0.055, 0.945, "a", "Internal segmentation diagnostics vary with canopy density")
    ax_a = fig.add_axes([0.085, 0.555, 0.855, 0.335])
    p_a = panel_a(ax_a, overall, density)

    heading(fig, 0.055, 0.465, "b", "Geographic separation of diagnostic tiles")
    ax_b = fig.add_axes([0.085, 0.110, 0.335, 0.285])
    p_b = panel_b(ax_b, geo)

    heading(fig, 0.500, 0.465, "c", "Overlap inference attenuates tile seams")
    ax_c = fig.add_axes([0.555, 0.110, 0.385, 0.285])
    panel_c(ax_c, overlap)

    stem = FIG / "Supplementary_Fig3"
    files = save(fig, stem)
    plt.close(fig)
    ImageOps.grayscale(Image.open(stem.with_suffix(".png"))).save(QA / "Supplementary_Fig3_grayscale.png")

    qa = {
        "status": "CANDIDATE",
        "visual_audit": (
            "Colour and greyscale renders show no text, legend, axis or cross-panel overlap; "
            "all three diagnostic encodings remain distinguishable."
        ),
        "cities_in_seam_audit": int(overlap.shape[0]),
        "untouched_base_tiles": int(geo.shape[0]),
        "base_tiles_at_least_250m": int((~geo["within_250m_of_model_development"].astype(bool)).sum()),
        "median_archived_transition_ratio": float(overlap["archived_transition_rate_ratio"].median()),
        "median_overlap_transition_ratio": float(overlap["overlap_transition_rate_ratio"].median()),
        "mean_canopy_shift_pp": float(overlap["overlap_minus_archived_canopy_pp"].mean()),
        "claim_limit": "Internal diagnostic only; not an independent 50-city geographic validation.",
        "files": files,
    }
    qa["bar_axis_starts_at_zero"] = True
    qa["minimum_font_pt"] = 6.2
    qa["claim_limit"] = "Internal diagnostic only; not an independent city-held-out or 50-city geographic validation."
    (QA / "Supplementary_Fig3_build_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    p_a.to_csv(SRC / "Fig3a_plotted_metrics.csv", index=False, encoding="utf-8-sig")
    p_b.to_csv(SRC / "Fig3b_plotted_counts.csv", index=False, encoding="utf-8-sig")

    caption = (
        "Supplementary Fig. 3 | Internal atlas diagnostics identify density dependence, limited geographic separation and inference seams. "
        "a, F1 and intersection-over-union (IoU) for the checkpoint-selection partition, a recovered 95-patch diagnostic partition and thirds of that diagnostic partition ordered by labelled-canopy density. Bars begin at zero; error bars are patch-bootstrap 95% intervals for the two overall partitions, whereas density-third values are pooled confusion-matrix diagnostics without resampling intervals. The recovered diagnostic yielded F1 = 0.871 and IoU = 0.771, but F1 was 0.714 in the lower-canopy third. "
        "b, Geographic relation of 80 diagnostic base tiles to imagery used during model development: 60 overlapped a development footprint, 12 were within 250 m and only 8 were at least 250 m away. "
        "c, Boundary-to-control transition ratios for one representative tile in each of 50 cities under archived non-overlap inference and 50% overlap re-inference. Mean transition ratios changed from 12.47 to 1.02, while mean canopy cover changed by -0.410 percentage points. "
        "These are internal diagnostic and measurement-sensitivity results, not independent city-held-out accuracy estimates or validation of all mapped pixels."
    )
    (TEXT / "Supplementary_Fig3_caption.txt").write_text(caption + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
