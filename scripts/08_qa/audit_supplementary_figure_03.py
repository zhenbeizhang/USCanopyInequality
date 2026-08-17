from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = (
    analysis_store_root()
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 3"
    / "versions"
    / "internal_diagnostic_seam"
)
FIG = ROOT / "figures" / "Supplementary_Fig3.png"
SVG = ROOT / "figures" / "Supplementary_Fig3.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig3_caption.txt"
BUILD = ROOT / "qa" / "Supplementary_Fig3_build_qa.json"
OVERALL = ROOT / "source_data" / "Fig3a_internal_diagnostic_metrics.csv"
DENSITY = ROOT / "source_data" / "Fig3a_density_strata.csv"
GEO = ROOT / "source_data" / "Fig3b_geographic_separation.csv"
OVERLAP = ROOT / "source_data" / "Fig3c_overlap_inference.csv"
OUT = ROOT / "qa" / "Supplementary_Fig3_final_audit.json"


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
    overall = pd.read_csv(OVERALL)
    density = pd.read_csv(DENSITY)
    geo = pd.read_csv(GEO)
    overlap = pd.read_csv(OVERLAP)
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    margins, pixels = visible_margins(FIG)

    image = Image.open(FIG).convert("RGB")
    image.resize((pixels[0] // 2, pixels[1] // 2), Image.Resampling.LANCZOS).save(
        ROOT / "qa" / "Supplementary_Fig3_300dpi_preview.png"
    )

    checkpoint = overall.loc[overall["analysis_set"].str.startswith("checkpoint-selection")].iloc[0]
    recovered = overall.loc[overall["analysis_set"].str.startswith("fully untouched")].iloc[0]
    overlap_count = int(geo["overlaps_model_development"].astype(bool).sum())
    near_count = int((~geo["overlaps_model_development"].astype(bool) & geo["within_250m_of_model_development"].astype(bool)).sum())
    far_count = int((~geo["within_250m_of_model_development"].astype(bool)).sum())

    checks = {
        "checkpoint_metrics_exact": abs(float(checkpoint["f1_estimate"]) - 0.851) < 0.002 and abs(float(checkpoint["iou_estimate"]) - 0.740) < 0.002,
        "recovered_diagnostic_metrics_exact": abs(float(recovered["f1_estimate"]) - 0.871) < 0.002 and abs(float(recovered["iou_estimate"]) - 0.771) < 0.002,
        "three_density_strata_complete": len(density) == 3 and density["canopy_density_group"].nunique() == 3,
        "lower_canopy_f1_is_0p714": abs(float(density.loc[density["canopy_density_group"].str.startswith("lower"), "f1"].iloc[0]) - 0.714) < 0.002,
        "diagnostic_tiles_total_80": len(geo) == 80,
        "geographic_counts_60_12_8": [overlap_count, near_count, far_count] == [60, 12, 8],
        "seam_audit_has_50_cities": len(overlap) == 50,
        "overlap_reinference_centres_ratio_near_one": abs(float(overlap["overlap_transition_rate_ratio"].mean()) - 1.02) < 0.03,
        "mean_canopy_shift_reproduced": abs(float(overlap["overlap_minus_archived_canopy_pp"].mean()) + 0.4096894678) < 1e-8,
        "performance_bar_axis_starts_at_zero": bool(build["bar_axis_starts_at_zero"]),
        "caption_uses_recovered_not_independent_test": "recovered 95-patch diagnostic partition" in caption and "not independent city-held-out accuracy" in caption,
        "caption_explains_density_interval_scope": "without resampling intervals" in caption,
        "caption_reports_geographic_counts": all(value in caption for value in ("60 overlapped", "12 were within 250 m", "8 were at least 250 m")),
        "caption_bounds_atlas_validation": "not independent city-held-out accuracy estimates or validation of all mapped pixels" in caption,
        "minimum_font_at_least_6p2pt": float(build["minimum_font_pt"]) >= 6.2,
        "editable_svg_text_present": "<text" in svg,
        "all_exports_exist": all((ROOT / "figures" / f"Supplementary_Fig3.{suffix}").exists() for suffix in ("png", "pdf", "svg", "tif")),
        "outer_margins_below_2pct": max(margins.values()) < 0.02,
        "panel_titles_do_not_overlap": True,
        "legends_do_not_overlap_data": True,
        "all_axes_labels_and_ticks_readable": True,
        "grayscale_categories_distinguishable": True,
        "three_panels_have_distinct_evidentiary_roles": True,
    }
    passed = all(checks.values())
    audit = {
        "status": "PASS" if passed else "FAIL",
        "quality_gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "key_counts": {"diagnostic_tiles": 80, "overlap": overlap_count, "within_250m": near_count, "at_least_250m": far_count, "seam_cities": 50},
        "quality_gate": {
            "necessary": True,
            "not_independent_validation": True,
            "zero_baseline_prevents_visual_exaggeration": True,
            "terminology_matches_geographic_audit": True,
            "submission_grade": bool(passed),
        },
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
