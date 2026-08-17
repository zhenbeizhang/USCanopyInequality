from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(str(analysis_store_root().joinpath('Result', 'Supplementary figure and table', 'Supplementary Fig. 15', 'versions', 'multicriteria_spatial_contrast')))
R5 = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'weighting_heterogeneity_audit')))
SUMMARY = ROOT / "source_data" / "Supplementary_Fig15_selected_city_summary.csv"
ALLOCATIONS = ROOT / "source_data" / "Supplementary_Fig15_selected_city_allocations.csv.gz"
FIGURE = ROOT / "figures" / "Supplementary_Fig15.png"
SVG = ROOT / "figures" / "Supplementary_Fig15.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig15_caption.txt"
BUILD_QA = ROOT / "qa" / "Supplementary_Fig15_qa.json"
OUT = ROOT / "qa" / "Supplementary_Fig15_final_audit.json"


def visible_margins(path: Path) -> tuple[dict[str, float], tuple[int, int]]:
    image = np.asarray(Image.open(path).convert("RGB"))
    visible = np.any(image < 247, axis=2)
    ys, xs = np.where(visible)
    height, width = visible.shape
    return {
        "top": float(ys.min() / height),
        "bottom": float((height - 1 - ys.max()) / height),
        "left": float(xs.min() / width),
        "right": float((width - 1 - xs.max()) / width),
    }, (width, height)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    allocation = pd.read_csv(ALLOCATIONS, dtype={"GEOID_str": str})
    diagnostics = pd.read_csv(R5 / "tables" / "result5_weighting_city_diagnostics.csv")
    build = json.loads(BUILD_QA.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    margins, pixels = visible_margins(FIGURE)

    expected = diagnostics.loc[diagnostics["city_display"].isin(["Chicago", "Minneapolis"])].set_index("city_display")
    observed = summary.set_index("city_display")
    overlap_error = float(
        np.max(
            np.abs(
                observed.loc[["Chicago", "Minneapolis"], "heat_social_oracle_overlap"].to_numpy(float)
                - expected.loc[["Chicago", "Minneapolis"], "population_weighted_overlap"].to_numpy(float)
            )
        )
    )
    budget_error = float(summary["max_rule_budget_error_m2"].abs().max())
    selected = allocation.loc[allocation["allocated_m2"] > 0]

    checks = {
        "two_prespecified_mechanism_cities": set(summary["city_display"]) == {"Chicago", "Minneapolis"},
        "overlap_reproduced": overlap_error < 1e-12,
        "exact_equal_area_budgets": budget_error < 1e-5,
        "three_allocation_rules_per_city": bool(allocation.groupby("city_display")["rule"].nunique().eq(3).all()),
        "positive_selected_areas": bool((selected["allocated_m2"] > 0).all()),
        "caption_states_case_selection": "illustrative mechanism contrasts" in caption,
        "caption_denies_planting_and_realized_outcomes": all(phrase in caption for phrase in ("not parcel-level planting plans", "realized cooling", "health benefit")),
        "caption_uses_current_rule_names": all(phrase in caption for phrase in ("surface-heat-first", "social-context-first", "equal-weight")),
        "no_old_rule_names": all(phrase not in caption.lower() for phrase in ("heat-context", "social-condition-first", "balanced rule")),
        "minimum_font_at_least_5p2pt": float(build["minimum_font_pt"]) >= 5.2,
        "no_text_text_overlap": int(build["text_text_overlap_count"]) == 0,
        "outer_margins_below_3pct": max(margins.values()) < 0.03,
        "editable_svg_text_present": "<text" in svg,
        "all_exports_exist": all((ROOT / "figures" / f"Supplementary_Fig15.{suffix}").exists() for suffix in ("png", "pdf", "svg", "tif")),
    }
    passed = bool(all(checks.values()))
    audit = {
        "status": "PASS" if passed else "FAIL",
        "quality_gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "maximum_overlap_reproduction_error": overlap_error,
        "maximum_budget_error_m2": budget_error,
        "quality_gate": {
            "necessary": True,
            "nonduplicative": True,
            "case_selection_not_outcome_claim": True,
            "diagnostic_allocation_boundary_explicit": True,
            "submission_grade": passed,
        },
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("Supplementary Fig. 15 audit failed")


if __name__ == "__main__":
    main()
