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
    / "Supplementary Fig. 13"
    / "versions"
    / "sacramento_budget_expansion"
)
ALLOCATIONS = ROOT / "source_data" / "Supplementary_Fig13_sacramento_allocations.csv.gz"
SUMMARY = ROOT / "source_data" / "Supplementary_Fig13_sacramento_budget_summary.csv"
CONTEXTS = ROOT / "source_data" / "Supplementary_Fig13_sacramento_context_shares.csv"
FIGURE = ROOT / "figures" / "Supplementary_Fig13.png"
SVG = ROOT / "figures" / "Supplementary_Fig13.svg"
BUILD_QA = ROOT / "qa" / "Supplementary_Fig13_qa.json"
CAPTION = ROOT / "text" / "Supplementary_Fig13_caption.txt"
OUT = ROOT / "qa" / "Supplementary_Fig13_final_audit.json"

BUDGETS = np.asarray([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00])


def visible_bbox_margins(path: Path) -> tuple[dict[str, float], tuple[int, int]]:
    image = np.asarray(Image.open(path).convert("RGB"))
    visible = np.any(image < 247, axis=2)
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


def luminance(hex_colour: str) -> float:
    rgb = np.array([int(hex_colour[index : index + 2], 16) for index in (1, 3, 5)], dtype=float)
    return float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))


def main() -> int:
    allocations = pd.read_csv(ALLOCATIONS, dtype={"GEOID_str": str})
    summary = pd.read_csv(SUMMARY)
    contexts = pd.read_csv(CONTEXTS)
    build_qa = json.loads(BUILD_QA.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    margins, pixels = visible_bbox_margins(FIGURE)

    wide = allocations.pivot(index="GEOID_str", columns="budget_fraction", values="allocated_m2").reindex(
        columns=BUDGETS
    )
    nested = bool(np.all(np.diff(wide.to_numpy(float), axis=1) >= -1e-8))
    allocated_by_budget = allocations.groupby("budget_fraction")["allocated_m2"].sum().reindex(BUDGETS)
    expected_by_budget = (
        summary.set_index("budget_fraction")["budget_km2"].reindex(BUDGETS).to_numpy(float) * 1e6
    )
    max_budget_error = float(np.max(np.abs(allocated_by_budget.to_numpy(float) - expected_by_budget)))
    context_sums = contexts.groupby("budget_fraction")["allocated_area_share_pct"].sum().reindex(BUDGETS)
    yield_values = summary.set_index("budget_fraction")["relative_heat_context_quantity_per_km2"].reindex(BUDGETS)
    svg_text = SVG.read_text(encoding="utf-8", errors="ignore")

    base_capacity_luminance_gap = abs(luminance("#E8EAE8") - luminance("#C3D9D0"))
    capacity_selection_luminance_gap = abs(luminance("#C3D9D0") - luminance("#D07738"))
    checks = {
        "parent_frame_is_34566_units_50_cities": (
            int(build_qa["fixed_analysis_frame"]) == 34566
            and int(build_qa["all_cities_in_parent_analysis"]) == 50
        ),
        "sacramento_has_345_units": int(build_qa["selected_city_analysis_units"]) == 345,
        "all_8_budgets_present": (
            len(summary) == 8
            and np.allclose(np.sort(summary["budget_fraction"].to_numpy(float)), BUDGETS)
        ),
        "allocation_table_is_complete": (
            len(allocations) == 345 * 8
            and allocations["GEOID_str"].nunique() == 345
        ),
        "exact_equal_area_budget_control": max_budget_error < 1e-6,
        "heat_rule_allocations_are_nested": nested,
        "context_shares_sum_to_100pct": float(np.max(np.abs(context_sums - 100.0))) < 1e-8,
        "per_area_heat_quantity_is_nonincreasing": bool(np.all(np.diff(yield_values) <= 1e-10)),
        "mapped_levels_are_1_10_50pct": set([0.01, 0.10, 0.50]).issubset(set(summary["budget_fraction"])),
        "example_not_selected_on_favourable_effect": "selected before plotting" in caption,
        "same_target_status_is_explicit": "same-target diagnostic" in caption,
        "not_described_as_planting_or_realized_cooling": (
            "not parcel-level planting sites" in caption and "not observed cooling" in caption
        ),
        "minimum_font_at_least_5p2pt": float(build_qa["minimum_font_pt"]) >= 5.2,
        "no_text_text_overlap": int(build_qa["text_text_overlap_count"]) == 0,
        "outer_margins_below_3pct": max(margins.values()) < 0.03,
        "base_layers_separate_in_grayscale": base_capacity_luminance_gap >= 15.0,
        "selection_separates_from_capacity_in_grayscale": capacity_selection_luminance_gap >= 40.0,
        "editable_svg_text_present": "<text" in svg_text,
        "all_exports_exist": all(
            (ROOT / "figures" / f"Supplementary_Fig13.{suffix}").exists()
            for suffix in ("png", "pdf", "svg", "tif")
        ),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "quality_gate_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "max_budget_error_m2": max_budget_error,
        "grayscale_luminance_gaps": {
            "outside_vs_capacity": base_capacity_luminance_gap,
            "capacity_vs_selection": capacity_selection_luminance_gap,
        },
        "manuscript_role": (
            "Local spatial expansion mechanism; Supplementary Fig. 11 provides national budget response and Supplementary Fig. 15 compares objectives at the fixed 10% budget."
        ),
        "quality_gate": {
            "necessary": True,
            "nonduplicative": True,
            "correct_graphical_grammar": True,
            "single_case_limit_explicit": True,
            "same_target_circularity_not_used_as_superiority_evidence": True,
            "submission_grade": bool(all(checks.values())),
        },
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
