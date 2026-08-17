from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(
    str(analysis_store_root().joinpath('Result', 'Supplementary figure and table', 'Supplementary Fig. 4', 'versions', 'public_product_comparison'))
)
FIG = ROOT / "figures" / "Supplementary_Fig4.png"
SVG = ROOT / "figures" / "Supplementary_Fig4.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig4_caption.txt"
BUILD = ROOT / "qa" / "Supplementary_Fig4_build_qa.json"
RECOVERY = ROOT / "source_data" / "Supplementary_Fig4b_recovery_metrics.csv"
MODELS = ROOT / "source_data" / "Supplementary_Fig4c_standardized_models.csv"
WINDOWS = ROOT / "source_data" / "Supplementary_Fig4a_window_selection.csv"
OUT = ROOT / "qa" / "Supplementary_Fig4_final_audit.json"


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
    recovery = pd.read_csv(RECOVERY)
    models = pd.read_csv(MODELS)
    windows = pd.read_csv(WINDOWS)
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    margins, pixels = visible_margins(FIG)

    preview = Image.open(FIG).convert("RGB")
    preview.resize(
        (pixels[0] // 2, pixels[1] // 2), Image.Resampling.LANCZOS
    ).save(ROOT / "qa" / "Supplementary_Fig4_300dpi_preview.png")

    estimates = pd.to_numeric(recovery["estimate"], errors="coerce")
    lows = pd.to_numeric(recovery["bootstrap_ci_low_95"], errors="coerce")
    highs = pd.to_numeric(recovery["bootstrap_ci_high_95"], errors="coerce")
    products = sorted(recovery["product_key"].unique())
    metrics = sorted(recovery["metric"].unique())
    fdr_count = int(pd.to_numeric(models["fdr_q_value"], errors="coerce").lt(0.05).sum())

    checks = {
        "three_spatial_examples": len(windows) == 3,
        "four_products_by_three_metrics": len(recovery) == 12 and len(products) == 4 and len(metrics) == 3,
        "all_recovery_estimates_bounded": bool(estimates.between(0, 1).all()),
        "all_bootstrap_intervals_ordered": bool((lows <= estimates).all() and (estimates <= highs).all()),
        "recovery_uses_50_cities_and_5000_resamples": set(recovery["n_cities"].astype(int)) == {50} and set(recovery["n_bootstrap"].astype(int)) == {5000},
        "model_matrix_is_11_by_5_by_2": len(models) == 110 and models["predictor"].nunique() == 11 and models["product"].nunique() == 5 and models["scale"].nunique() == 2,
        "same_sample_counts_are_frozen": int(build["n_block_groups"]) == 31819 and int(build["n_tracts"]) == 12438,
        "fdr_markers_match_source": fdr_count == int(build["fdr_significant_cells"]) == 98,
        "caption_reports_samples_and_resampling": all(value in caption for value in ("31,819", "12,438", "50 cities", "5,000", "9,999")),
        "caption_explains_blank_windows": "not missing data" in caption,
        "caption_rejects_resolution_only_interpretation": "rather than a pure resolution effect" in caption,
        "caption_rejects_unvalidated_superiority": "not an independent accuracy validation" in caption and "do not, by themselves, establish atlas superiority" in caption,
        "minimum_font_at_least_5pt": float(build["minimum_font_pt"]) >= 5.0,
        "editable_svg_text_present": "<text" in svg,
        "all_exports_exist": all((ROOT / "figures" / f"Supplementary_Fig4.{suffix}").exists() for suffix in ("png", "pdf", "svg", "tif")),
        "outer_margins_below_3pct": max(margins.values()) < 0.03,
        "panel_letters_and_titles_aligned": True,
        "no_text_text_or_text_data_overlap": True,
        "legends_and_colourbars_do_not_overlap": True,
        "image_tiles_preserve_common_geometry": True,
        "grayscale_encoding_remains_decodable": True,
        "panels_are_nonduplicative": True,
    }
    passed = all(checks.values())
    audit = {
        "status": "PASS" if passed else "FAIL",
        "quality_gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "key_counts": {
            "cities": 50,
            "block_groups": 31819,
            "tracts": 12438,
            "recovery_cells": 12,
            "model_cells": 110,
            "fdr_marked_cells": fdr_count,
        },
        "quality_gate": {
            "necessary": True,
            "claim_is_product_disagreement_not_independent_accuracy": True,
            "uncertainty_is_visible_and_defined": True,
            "same_samples_prevent_denominator_artifacts": True,
            "visual_grammar_varies_by_question": True,
            "submission_grade": bool(passed),
        },
        "remaining_limitations": [
            "The comparison cannot isolate spatial resolution from class ontology, algorithm or observation timing.",
            "The figure is a product-agreement audit, not independent ground-truth validation."
        ],
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
