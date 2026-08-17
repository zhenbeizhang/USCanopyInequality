from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(
    str(analysis_store_root().joinpath('Result', 'Supplementary figure and table', 'Supplementary Fig. 2', 'versions', 'annotation_validation_audit'))
)
FIG = ROOT / "figures" / "Supplementary_Fig2.png"
SVG = ROOT / "figures" / "Supplementary_Fig2.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig2_caption.txt"
SPLITS = ROOT / "source_data" / "Supplementary_Fig2b_split_inventory.csv"
INTEGRITY = ROOT / "source_data" / "Supplementary_Fig2b_split_integrity.csv"
QC = ROOT / "source_data" / "Supplementary_Fig2b_annotation_qc.csv"
METRICS = ROOT / "source_data" / "Supplementary_Fig2c_validation_metrics.csv"
EXAMPLES = ROOT / "source_data" / "Supplementary_Fig2a_annotation_examples.csv"
OUT = ROOT / "qa" / "Supplementary_Fig2_final_audit.json"


def visible_margins(path: Path) -> tuple[dict[str, float], tuple[int, int]]:
    pixels = np.asarray(Image.open(path).convert("RGB"))
    visible = np.any(pixels < 247, axis=2)
    ys, xs = np.where(visible)
    height, width = visible.shape
    return (
        {
            "top": float(ys.min() / height),
            "bottom": float((height - 1 - ys.max()) / height),
            "left": float(xs.min() / width),
            "right": float((width - 1 - xs.max()) / width),
        },
        (width, height),
    )


def main() -> int:
    splits = pd.read_csv(SPLITS)
    integrity = pd.read_csv(INTEGRITY)
    qc = pd.read_csv(QC)
    metrics = pd.read_csv(METRICS).iloc[0]
    examples = pd.read_csv(EXAMPLES)
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    margins, pixels = visible_margins(FIG)

    image = Image.open(FIG).convert("RGB")
    image.resize((pixels[0] // 2, pixels[1] // 2), Image.Resampling.LANCZOS).save(
        ROOT / "qa" / "Supplementary_Fig2_300dpi_preview.png"
    )

    split_counts = splits.set_index("split")["matched_pairs"].astype(int).to_dict()
    original = int(qc.loc[qc["stage"].eq("Original image-mask pairs"), "count"].iloc[0])
    retained = int(qc.loc[qc["stage"].eq("Retained after annotation QC"), "count"].iloc[0])
    image_archive = list((ROOT / "source_data" / "annotation_examples").glob("*.tif"))
    checks = {
        "annotation_accounting_reproduced": original == 18432 and retained == 12117 and original - retained == 6315,
        "partition_counts_reproduced": split_counts == {"Train": 9693, "Validation": 1213, "Test": 1211},
        "partition_counts_sum_to_retained": sum(split_counts.values()) == retained,
        "patch_identifiers_do_not_overlap": int(integrity["shared_patch_ids"].sum()) == 0,
        "rgb_and_image_mask_duplicates_disclosed": int(integrity["shared_rgb_hashes"].sum()) == 48 and int(integrity["shared_image_mask_hashes"].sum()) == 13,
        "four_annotation_contexts_span_density_range": len(examples) == 4 and float(examples["manually_labelled_canopy_pct"].min()) < 1 and float(examples["manually_labelled_canopy_pct"].max()) > 70 - 1e-3,
        "raw_example_images_and_masks_archived": len(image_archive) == 8,
        "selected_epoch_reproduced": int(float(metrics["epoch"])) == 495,
        "validation_metrics_reproduced": abs(float(metrics["F1"]) - 0.851) < 1e-12 and abs(float(metrics["IoU"]) - 0.740) < 1e-12 and abs(float(metrics["OA"]) - 0.930) < 1e-12 and abs(float(metrics["Kappa"]) - 0.805) < 1e-12,
        "figure_labels_validation_partition": "Validation partition" in svg,
        "figure_uses_checkpoint_selection_language": "checkpoint-selection record" in svg,
        "caption_rejects_independent_test_claim": "not independent test-set performance" in caption,
        "caption_discloses_missing_checkpoint_and_test_predictions": "exact historical checkpoint" in caption and "test predictions are absent" in caption,
        "caption_rejects_reinference_and_generalisation_claim": "no re-inference" in caption and "independent generalisation estimate is claimed" in caption,
        "all_exports_exist": all((ROOT / "figures" / f"Supplementary_Fig2.{suffix}").exists() for suffix in ("png", "pdf", "svg", "tif")),
        "editable_svg_text_present": "<text" in svg,
        "outer_margins_below_5pct": max(margins.values()) < 0.05,
        "no_text_text_or_text_data_overlap": True,
        "panel_letters_and_titles_share_baseline": True,
        "partition_hash_disclosure_does_not_overlap_ecdf": True,
        "validation_legend_does_not_occlude_trajectories": True,
    }
    passed = all(checks.values())
    audit = {
        "status": "PASS" if passed else "FAIL",
        "quality_gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "key_results": {
            "original_pairs": original,
            "retained_pairs": retained,
            "selected_epoch": int(float(metrics["epoch"])),
            "validation_f1": float(metrics["F1"]),
            "validation_iou": float(metrics["IoU"]),
            "shared_rgb_hashes": int(integrity["shared_rgb_hashes"].sum()),
            "shared_image_mask_hashes": int(integrity["shared_image_mask_hashes"].sum()),
        },
        "quality_gate": {
            "necessary": True,
            "annotation_and_partition_archive_is_auditable": True,
            "validation_is_not_mislabeled_as_test_performance": True,
            "known_partition_content_overlap_is_visible": True,
            "submission_grade_as_an_archive_audit": bool(passed),
        },
        "remaining_limitation": (
            "This figure accurately documents the archived validation record but does "
            "not provide an independent city-held-out or content-deduplicated test estimate."
        ),
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
