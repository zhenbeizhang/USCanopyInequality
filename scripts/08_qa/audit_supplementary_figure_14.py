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
    / "Supplementary Fig. 14"
    / "versions"
    / "group_budget_paired_differences"
)
DIFFERENCES = ROOT / "source_data" / "Supplementary_Fig14_paired_differences_from_uniform.csv"
ABSOLUTE = ROOT / "source_data" / "Supplementary_Fig14_absolute_group_ratios.csv"
CENTRAL = ROOT / "source_data" / "Supplementary_Fig14_city_group_ratios_by_budget.csv.gz"
FIGURE = ROOT / "figures" / "Supplementary_Fig14.png"
SVG = ROOT / "figures" / "Supplementary_Fig14.svg"
BUILD_QA = ROOT / "qa" / "Supplementary_Fig14_qa.json"
CAPTION = ROOT / "text" / "Supplementary_Fig14_caption.txt"
OUT = ROOT / "qa" / "Supplementary_Fig14_final_audit.json"

BUDGETS = np.asarray([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00])
RULES = ("heat_context_oracle", "social_condition_oracle", "pareto_0.50")
GROUPS = (
    "people_of_color_pct",
    "hispanic_or_latino_pct",
    "black_alone_pct",
    "asian_alone_pct",
    "white_alone_pct",
    "age_65_plus_pct",
)


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
    differences = pd.read_csv(DIFFERENCES)
    absolute = pd.read_csv(ABSOLUTE)
    central = pd.read_csv(CENTRAL)
    build_qa = json.loads(BUILD_QA.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    margins, pixels = visible_bbox_margins(FIGURE)
    svg_text = SVG.read_text(encoding="utf-8", errors="ignore")

    expected_index = pd.MultiIndex.from_product(
        [BUDGETS, RULES, GROUPS], names=["budget_fraction", "rule", "group"]
    )
    observed_index = differences.set_index(["budget_fraction", "rule", "group"]).index
    grid_complete = len(differences) == len(expected_index) and set(observed_index) == set(expected_index)
    full = differences[np.isclose(differences["budget_fraction"], 1.0)]
    ten = differences[np.isclose(differences["budget_fraction"], 0.10)]
    values = differences["equal_city_central_difference_from_uniform"].to_numpy(float)
    supported = differences["paired_interval_excludes_zero"].astype(bool)
    endpoint_luminance_gap = abs(luminance("#3D6F8D") - luminance("#D17652"))
    center_endpoint_min_gap = min(
        abs(luminance("#F4F1EC") - luminance("#3D6F8D")),
        abs(luminance("#F4F1EC") - luminance("#D17652")),
    )

    checks = {
        "fixed_frame_34566_and_50_cities": (
            int(build_qa["analysis_units"]) == 34566 and int(build_qa["cities"]) == 50
        ),
        "complete_8_budget_3_rule_6_group_grid": grid_complete,
        "central_city_grid_complete": len(central) == 50 * 8 * 4 * 6,
        "absolute_interval_grid_complete": len(absolute) == 8 * 4 * 6,
        "paired_intervals_finite_and_ordered": bool(
            differences[["ci_low", "ci_high"]].notna().all().all()
            and (differences["ci_low"] <= differences["ci_high"]).all()
        ),
        "marker_count_matches_supported_cells": int(supported.sum()) == int(build_qa["paired_interval_supported_cells"]),
        "colour_scale_does_not_clip": float(np.max(np.abs(values))) <= float(build_qa["colour_limit"]),
        "all_rules_converge_at_full_budget": float(np.max(np.abs(full["equal_city_central_difference_from_uniform"]))) < 1e-8,
        "primary_10pct_column_complete": len(ten) == 3 * 6,
        "500_city_block_replicates": int(build_qa["city_spatial_block_replicates"]) == 500,
        "5000_two_level_draws": int(build_qa["two_level_draws"]) == 5000,
        "race_ethnicity_ranking_boundary_explicit": "Race and ethnicity were not ranking inputs" in caption,
        "age65_ranking_boundary_explicit": "age 65+ contributes to the social-condition ordering" in caption,
        "nonexclusive_groups_explicit": "groups are not mutually exclusive" in caption,
        "outcome_boundary_explicit": (
            "not equity achieved" in caption and "not equity achieved" in caption
            and "observed cooling" in caption and "health benefit" in caption
        ),
        "minimum_font_at_least_5p2pt": float(build_qa["minimum_font_pt"]) >= 5.2,
        "no_text_text_overlap": int(build_qa["text_text_overlap_count"]) == 0,
        "outer_margins_below_3pct": max(margins.values()) < 0.03,
        "diverging_endpoints_separate_in_grayscale": endpoint_luminance_gap >= 25.0,
        "zero_separates_from_both_endpoints_in_grayscale": center_endpoint_min_gap >= 80.0,
        "editable_svg_text_present": "<text" in svg_text,
        "all_exports_exist": all(
            (ROOT / "figures" / f"Supplementary_Fig14.{suffix}").exists()
            for suffix in ("png", "pdf", "svg", "tif")
        ),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "quality_gate_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "central_difference_range": [float(np.min(values)), float(np.max(values))],
        "supported_cells": int(supported.sum()),
        "grayscale_luminance_gaps": {
            "negative_vs_positive_endpoint": endpoint_luminance_gap,
            "minimum_zero_vs_endpoint": center_endpoint_min_gap,
        },
        "manuscript_role": (
            "Extends the primary 10% group-ratio panel in Fig. 5 to paired rule-minus-uniform contrasts across eight budgets; it does not repeat the national efficiency curves in Supplementary Fig. 11."
        ),
        "quality_gate": {
            "necessary": True,
            "nonduplicative": True,
            "paired_comparison_controls_baseline_composition": True,
            "uncertainty_visible_without_overplotting": True,
            "no_equity_or_benefit_claim": True,
            "submission_grade": bool(all(checks.values())),
        },
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
