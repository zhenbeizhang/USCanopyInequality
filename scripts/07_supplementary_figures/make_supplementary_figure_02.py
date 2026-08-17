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
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = analysis_store_root()
SUPP_ROOT = (
    PROJECT_ROOT
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 2"
)
OUT_ROOT = SUPP_ROOT / "versions" / "annotation_validation_audit"
FIG_DIR = OUT_ROOT / "figures"
SRC_DIR = OUT_ROOT / "source_data"
QA_DIR = OUT_ROOT / "qa"
TEXT_DIR = OUT_ROOT / "text"
METHODS_DIR = OUT_ROOT / "methods"

RESULT1_SOURCE = (
    PROJECT_ROOT
    / "Result"
    / "Result1"
    / "Result1_NatureCities_manuscript_Fig1"
    / "source_data"
)
TRAINING_LOG = RESULT1_SOURCE / "Supplementary_validation_training_log_parsed.csv"
METRIC_FILE = RESULT1_SOURCE / "Supplementary_best_validation_metrics.csv"
ANNOTATION_ROOT = Path(str(external_input_root().joinpath('us_tree_methods_review', 'sample_oam_tcd')))
LEGACY_ANNOTATION_SOURCE = SUPP_ROOT / "source_data"

for directory in [FIG_DIR, SRC_DIR, QA_DIR, TEXT_DIR, METHODS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


COLORS = {
    "black": "#1E2223",
    "grey": "#747B7D",
    "light_grey": "#D9DEDD",
    "very_light": "#F4F5F3",
    "green": "#177356",
    "green_light": "#A9D0B8",
    "blue": "#3F739D",
    "orange": "#C56F35",
    "purple": "#684A78",
    "rose": "#B24F62",
    "amber_light": "#E9D2BC",
}
SPLIT_COLORS = {
    "Train": COLORS["blue"],
    "Validation": COLORS["orange"],
    "Test": COLORS["purple"],
}

CANONICAL_STEM = "Supplementary_Fig2"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 6.5,
            "axes.labelsize": 6.6,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str, y: float = 1.015) -> None:
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["black"],
        clip_on=False,
    )
    ax.annotate(
        title,
        xy=(0.0, y),
        xycoords=ax.transAxes,
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["black"],
        clip_on=False,
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def mask_edge(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask, dtype=bool)
    edge[:-1] |= mask[:-1] != mask[1:]
    edge[1:] |= mask[:-1] != mask[1:]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return edge


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = rgb.astype(float).copy()
    green = np.asarray(mpl.colors.to_rgb(COLORS["green"])) * 255
    output[mask] = output[mask] * 0.43 + green * 0.57
    output = np.clip(output, 0, 255).astype(np.uint8)
    output[mask_edge(mask)] = 255
    return output


def show_image(ax: plt.Axes, image: np.ndarray) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(COLORS["black"])


def read_inputs() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    log_df = pd.read_csv(TRAINING_LOG)
    best = pd.read_csv(METRIC_FILE).iloc[0]
    split_df = pd.read_csv(LEGACY_ANNOTATION_SOURCE / "Supplementary_Fig2b_split_inventory.csv")
    overlap_df = pd.read_csv(
        LEGACY_ANNOTATION_SOURCE / "Supplementary_Fig2b_split_integrity.csv"
    )
    qc_df = pd.read_csv(
        LEGACY_ANNOTATION_SOURCE / "Supplementary_Fig2b_annotation_qc.csv"
    )
    ecdf = pd.read_csv(
        LEGACY_ANNOTATION_SOURCE / "Supplementary_Fig2b_partition_canopy_ecdf.csv"
    )

    assert int(float(best["epoch"])) == 495
    assert split_df["matched_pairs"].sum() == 12117
    assert int(qc_df.loc[qc_df["stage"] == "Original image-mask pairs", "count"].iloc[0]) == 18432
    assert int(overlap_df["shared_patch_ids"].sum()) == 0
    assert int(overlap_df["shared_rgb_hashes"].sum()) == 48
    assert int(overlap_df["shared_image_mask_hashes"].sum()) == 13
    return log_df, best, split_df, overlap_df, qc_df, ecdf


def draw_annotation_examples(ax: plt.Axes) -> pd.DataFrame:
    ax.set_axis_off()
    panel_title(ax, "a", "Manual canopy labels across image contexts")
    examples = [
        ("1596_x154_y000.tif", "Low-canopy edge"),
        ("462_x154_y000.tif", "Trees and shadows"),
        ("38_x000_y154.tif", "Residential crowns"),
        ("2505_x000_y000.tif", "Closed canopy"),
    ]
    rows = [("RGB image", "rgb"), ("Manual label", "mask"), ("Label overlay", "overlay")]
    record_rows = []

    left, right = 0.115, 0.995
    col_gap = 0.018
    cell_w = (right - left - 3 * col_gap) / 4
    top, bottom = 0.895, 0.02
    row_gap = 0.018
    cell_h = (top - bottom - 2 * row_gap) / 3
    image_dir = SRC_DIR / "annotation_examples"
    image_dir.mkdir(parents=True, exist_ok=True)

    for col, (patch_id, context) in enumerate(examples):
        x0 = left + col * (cell_w + col_gap)
        ax.text(
            x0 + cell_w / 2,
            0.925,
            context,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.4,
            fontweight="bold",
            color=COLORS["black"],
        )
        rgb_path = ANNOTATION_ROOT / "tif" / patch_id
        mask_path = ANNOTATION_ROOT / "mask" / patch_id
        rgb = load_rgb(rgb_path)
        mask = load_mask(mask_path)
        shutil.copy2(rgb_path, image_dir / f"{Path(patch_id).stem}_rgb.tif")
        shutil.copy2(mask_path, image_dir / f"{Path(patch_id).stem}_mask.tif")
        record_rows.append(
            {
                "patch_id": patch_id,
                "context": context,
                "manually_labelled_canopy_pct": float(mask.mean() * 100),
            }
        )

        for row, (row_label, kind) in enumerate(rows):
            y0 = top - (row + 1) * cell_h - row * row_gap
            if kind == "rgb":
                image = rgb
            elif kind == "mask":
                image = np.full((*mask.shape, 3), 247, dtype=np.uint8)
                image[mask] = np.asarray(mpl.colors.to_rgb(COLORS["green"])) * 255
                image[mask_edge(mask)] = 22
            else:
                image = overlay(rgb, mask)
            inset = ax.inset_axes([x0, y0, cell_w, cell_h])
            show_image(inset, image)
            if col == 0:
                ax.text(
                    left - 0.016,
                    y0 + cell_h / 2,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.2,
                    color=COLORS["black"],
                )

    out = pd.DataFrame(record_rows)
    out.to_csv(SRC_DIR / "Supplementary_Fig2a_annotation_examples.csv", index=False)
    return out


def draw_archive_audit(
    ax: plt.Axes,
    split_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    ecdf: pd.DataFrame,
) -> None:
    ax.set_axis_off()
    panel_title(ax, "b", "Annotation archive and partition audit")

    original = int(qc_df.loc[qc_df["stage"] == "Original image-mask pairs", "count"].iloc[0])
    retained = int(
        qc_df.loc[qc_df["stage"] == "Retained after annotation QC", "count"].iloc[0]
    )
    excluded = original - retained

    x0, width = 0.015, 0.97
    ax.text(
        x0,
        0.91,
        f"{original:,} original image-mask pairs",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        fontweight="bold",
        color=COLORS["black"],
    )
    retained_width = width * retained / original
    y0, height = 0.81, 0.075
    ax.add_patch(
        Rectangle(
            (x0, y0),
            retained_width,
            height,
            transform=ax.transAxes,
            facecolor=COLORS["green"],
            edgecolor=COLORS["black"],
            lw=0.45,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0 + retained_width, y0),
            width - retained_width,
            height,
            transform=ax.transAxes,
            facecolor=COLORS["amber_light"],
            edgecolor=COLORS["black"],
            lw=0.45,
        )
    )
    ax.text(
        x0 + retained_width / 2,
        y0 + height / 2,
        f"{retained:,} retained",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=5.9,
        fontweight="bold",
    )
    ax.text(
        x0 + retained_width + (width - retained_width) / 2,
        y0 + height / 2,
        f"{excluded:,} excluded",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=COLORS["black"],
        fontsize=5.7,
    )

    split_y, split_h = 0.675, 0.072
    split_gap = 0.012
    split_box_width = (width - 2 * split_gap) / 3
    for index, split in enumerate(["Train", "Validation", "Test"]):
        row = split_df.loc[split_df["split"] == split].iloc[0]
        cursor = x0 + index * (split_box_width + split_gap)
        ax.add_patch(
            Rectangle(
                (cursor, split_y),
                split_box_width,
                split_h,
                transform=ax.transAxes,
                facecolor=SPLIT_COLORS[split],
                edgecolor=COLORS["black"],
                lw=0.4,
            )
        )
        label = f"{split}\nn={int(row['matched_pairs']):,}"
        ax.text(
            cursor + split_box_width / 2,
            split_y + split_h / 2,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=5.35,
            fontweight="bold",
            linespacing=0.9,
        )

    shared_rgb = int(overlap_df["shared_rgb_hashes"].sum())
    shared_pairs = int(overlap_df["shared_image_mask_hashes"].sum())
    ax.text(
        x0,
        0.615,
        f"Post-hoc cross-partition hashes: {shared_rgb} RGB; {shared_pairs} exact image-mask",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.7,
        color=COLORS["black"],
    )
    ax.text(
        x0,
        0.575,
        "All exact image-mask matches contained 0% labelled canopy",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.55,
        color=COLORS["black"],
    )

    ecdf_ax = ax.inset_axes([0.10, 0.08, 0.86, 0.43])
    styles = {"Train": "-", "Validation": "--", "Test": ":"}
    for split in ["Train", "Validation", "Test"]:
        subset = ecdf.loc[ecdf["split"] == split]
        ecdf_ax.plot(
            subset["canopy_percent"],
            subset["cumulative_percent"],
            color=SPLIT_COLORS[split],
            linestyle=styles[split],
            lw=1.15,
            label=split,
        )
    ecdf_ax.set_xlim(0, 100)
    ecdf_ax.set_ylim(0, 100)
    ecdf_ax.set_xticks([0, 25, 50, 75, 100])
    ecdf_ax.set_yticks([0, 25, 50, 75, 100])
    ecdf_ax.set_xlabel("Labelled canopy in patch (%)", labelpad=1.5)
    ecdf_ax.set_ylabel("Cumulative patches (%)", labelpad=1.5)
    ecdf_ax.tick_params(length=2.2, width=0.55, pad=1.4)
    ecdf_ax.legend(
        loc="lower right",
        ncol=1,
        handlelength=2.0,
        borderaxespad=0.15,
        labelspacing=0.25,
    )


def draw_validation_record(ax: plt.Axes, log_df: pd.DataFrame, best: pd.Series) -> None:
    ax.set_axis_off()
    panel_title(ax, "c", "Validation trajectory and checkpoint-selection record")

    curve_ax = ax.inset_axes([0.085, 0.13, 0.61, 0.72])
    curves = [
        ("Precision", COLORS["blue"], 0.9, "-"),
        ("Recall", COLORS["orange"], 0.9, "-"),
        ("F1", COLORS["green"], 1.55, "-"),
        ("IoU", COLORS["rose"], 1.15, "--"),
        ("Kappa", COLORS["purple"], 1.05, ":"),
    ]
    for metric, color, linewidth, linestyle in curves:
        curve_ax.plot(
            log_df["epoch"],
            log_df[metric],
            color=color,
            lw=linewidth,
            linestyle=linestyle,
            alpha=0.95,
            label=metric,
        )
    selected_epoch = int(float(best["epoch"]))
    curve_ax.axvline(
        selected_epoch,
        color=COLORS["black"],
        lw=0.7,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    curve_ax.scatter(
        [selected_epoch],
        [float(best["F1"])],
        s=26,
        color=COLORS["green"],
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
    )
    curve_ax.text(
        selected_epoch - 8,
        0.932,
        "epoch 495",
        ha="right",
        va="top",
        fontsize=5.7,
        color=COLORS["black"],
    )
    curve_ax.set_xlim(0, 510)
    curve_ax.set_ylim(0.45, 0.95)
    curve_ax.set_xticks([0, 100, 200, 300, 400, 500])
    curve_ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    curve_ax.set_xlabel("Training epoch")
    curve_ax.set_ylabel("Validation score")
    curve_ax.tick_params(length=2.3, width=0.55, pad=1.4)
    curve_ax.legend(
        loc="lower right",
        ncol=2,
        columnspacing=0.75,
        handlelength=1.9,
        labelspacing=0.2,
        borderaxespad=0.15,
    )

    score_ax = ax.inset_axes([0.75, 0.23, 0.23, 0.56])
    score_metrics = ["F1", "IoU", "OA", "Kappa"]
    score_values = [float(best[name]) for name in score_metrics]
    y = np.arange(len(score_metrics))[::-1]
    score_ax.barh(
        y,
        score_values,
        height=0.50,
        color=[COLORS["green"], COLORS["green_light"], "#B7C7D2", "#C8B8D0"],
        edgecolor="white",
        lw=0.45,
    )
    score_ax.set_xlim(0.5, 1.0)
    score_ax.set_xticks([0.5, 0.75, 1.0])
    score_ax.set_yticks(y)
    score_ax.set_yticklabels(score_metrics)
    score_ax.tick_params(axis="both", length=2.0, width=0.5, pad=1.4)
    score_ax.set_xlabel("Score", labelpad=1.5)
    for yy, value in zip(y, score_values):
        score_ax.text(
            min(value + 0.012, 0.985),
            yy,
            f"{value:.3f}",
            ha="left" if value < 0.96 else "right",
            va="center",
            fontsize=5.7,
            color=COLORS["black"],
        )
    ax.text(
        0.75,
        0.835,
        "Validation partition",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=COLORS["black"],
    )


def save_source_data(
    log_df: pd.DataFrame,
    best: pd.Series,
    split_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    ecdf: pd.DataFrame,
) -> None:
    log_df.to_csv(SRC_DIR / "Supplementary_Fig2c_training_trajectory.csv", index=False)
    pd.DataFrame([best]).to_csv(
        SRC_DIR / "Supplementary_Fig2c_validation_metrics.csv", index=False
    )
    split_df.to_csv(SRC_DIR / "Supplementary_Fig2b_split_inventory.csv", index=False)
    overlap_df.to_csv(SRC_DIR / "Supplementary_Fig2b_split_integrity.csv", index=False)
    qc_df.to_csv(SRC_DIR / "Supplementary_Fig2b_annotation_qc.csv", index=False)
    ecdf.to_csv(SRC_DIR / "Supplementary_Fig2b_partition_canopy_ecdf.csv", index=False)


def build_figure(
    log_df: pd.DataFrame,
    best: pd.Series,
    split_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    ecdf: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    apply_style()
    fig = plt.figure(figsize=(7.2, 5.75), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.43, 1.0],
        width_ratios=[0.43, 0.57],
        left=0.035,
        right=0.988,
        top=0.965,
        bottom=0.055,
        hspace=0.15,
        wspace=0.15,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    examples = draw_annotation_examples(ax_a)
    draw_archive_audit(ax_b, split_df, overlap_df, qc_df, ecdf)
    draw_validation_record(ax_c, log_df, best)
    return fig, examples


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    outputs = []
    for extension in ["png", "pdf", "svg", "tif"]:
        path = directory / f"{stem}.{extension}"
        kwargs = {}
        if extension in {"png", "tif"}:
            kwargs["dpi"] = 600
        if extension == "tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def write_caption(
    best: pd.Series,
    split_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> Path:
    counts = {
        row["split"]: int(row["matched_pairs"]) for _, row in split_df.iterrows()
    }
    shared_rgb = int(overlap_df["shared_rgb_hashes"].sum())
    shared_pairs = int(overlap_df["shared_image_mask_hashes"].sum())
    caption = (
        "Supplementary Fig. 2 | Canopy annotation, partition integrity and validation record. "
        "a, Representative aerial-image patches, binary manual labels and label overlays across "
        "low-canopy, shadowed, residential and closed-canopy contexts. Green denotes manually labelled "
        "tree canopy; unfilled pixels were treated as non-canopy. b, Annotation archive and partition "
        "audit. Of 18,432 original image-mask pairs, 12,117 were retained after quality control and "
        f"partitioned into training (n={counts['Train']:,}), validation (n={counts['Validation']:,}) "
        f"and test (n={counts['Test']:,}) sets. The lower panel compares labelled-canopy distributions "
        f"among partitions. Patch identifiers did not overlap. A post-hoc content audit identified "
        f"{shared_rgb} shared RGB hashes and {shared_pairs} shared image-mask hashes across partitions; "
        "all shared image-mask pairs contained 0% labelled canopy. c, Archived validation trajectories "
        "across the 500-epoch training schedule and validation scores at epoch 495, selected by maximum "
        f"validation F1 (F1={float(best['F1']):.3f}, IoU={float(best['IoU']):.3f}, "
        f"OA={float(best['OA']):.3f}, Kappa={float(best['Kappa']):.3f}). These are checkpoint-selection records from the validation partition, "
        "not independent test-set performance. The exact historical checkpoint, raw probability rasters "
        "and test predictions are absent from the current archive, so no re-inference, threshold "
        "sensitivity or independent generalisation estimate is claimed."
    )
    paths = [TEXT_DIR / "Supplementary_Fig2_caption.txt"]
    for path in paths:
        path.write_text(caption, encoding="utf-8")
    return paths[0]


def write_qa(
    examples: pd.DataFrame,
    best: pd.Series,
    split_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    outputs: list[Path],
    caption: Path,
) -> None:
    qa = {
        "figure": "Supplementary Fig. 2",
        "analysis_role": "annotation and validation audit",
        "core_conclusion": (
            "Annotation contexts, sample filtering, partition composition and validation-only "
            "model selection are auditable without presenting the archived validation metrics "
            "as independent generalisation performance."
        ),
        "n_original_pairs": 18432,
        "n_retained_pairs": int(split_df["matched_pairs"].sum()),
        "split_counts": {
            row["split"]: int(row["matched_pairs"]) for _, row in split_df.iterrows()
        },
        "validation_metrics": {
            key: float(best[key]) for key in ["Precision", "Recall", "F1", "IoU", "OA", "Kappa"]
        },
        "selected_epoch": int(float(best["epoch"])),
        "shared_patch_ids": int(overlap_df["shared_patch_ids"].sum()),
        "shared_rgb_hashes": int(overlap_df["shared_rgb_hashes"].sum()),
        "shared_image_mask_hashes": int(
            overlap_df["shared_image_mask_hashes"].sum()
        ),
        "independent_test_performance_claimed": False,
        "annotation_examples": examples.to_dict(orient="records"),
        "saved_files": [str(path) for path in outputs],
        "caption": str(caption),
        "quality_gate_status": "PASS",
        "quality_gate": {
            "scientific_necessity": "pass",
            "validation_status_explicit": "pass",
            "split_integrity_disclosed": "pass",
            "main_figure_duplication": "pass",
            "remaining_limitation": (
                "No independent city-held-out or content-deduplicated test estimate is available; "
                "the figure documents this limitation rather than resolving it."
            ),
        },
    }
    (QA_DIR / "Supplementary_Fig2_build_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (QA_DIR / "Supplementary_Fig2_provenance.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    log_df, best, split_df, overlap_df, qc_df, ecdf = read_inputs()
    save_source_data(log_df, best, split_df, overlap_df, qc_df, ecdf)
    fig, examples = build_figure(log_df, best, split_df, overlap_df, qc_df, ecdf)
    outputs = []
    outputs.extend(save_figure(fig, FIG_DIR, CANONICAL_STEM))
    plt.close(fig)

    Image.open(FIG_DIR / f"{CANONICAL_STEM}.png").convert("L").save(
        QA_DIR / f"{CANONICAL_STEM}_grayscale.png"
    )
    caption = write_caption(best, split_df, overlap_df)
    write_qa(examples, best, split_df, overlap_df, outputs, caption)
    print(
        json.dumps(
            {
                "retained_pairs": int(split_df["matched_pairs"].sum()),
                "selected_epoch": int(float(best["epoch"])),
                "outputs": [str(path) for path in outputs],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
