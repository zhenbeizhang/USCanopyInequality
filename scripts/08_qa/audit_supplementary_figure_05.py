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
    / "Supplementary Fig. 5"
    / "versions"
    / "complete_case_city_atlas"
)
MATRIX = ROOT / "source_data" / "Supplementary_Fig5_city_social_gap_atlas.csv"
QUINTILES = ROOT / "source_data" / "Supplementary_Fig5_city_social_quintiles.csv.gz"
FIGURE = ROOT / "figures" / "Supplementary_Fig5.png"
BUILD_QA = ROOT / "qa" / "Supplementary_Fig5_qa.json"
OUT = ROOT / "qa" / "Supplementary_Fig5_final_audit.json"

VARIABLES = [
    "Low income",
    "Poverty",
    "Older age",
    "Black",
    "Hispanic/Latino",
    "Asian",
    "Not White alone",
    "Renters",
    "No vehicle",
    "Public transit",
    "Density",
]


def visible_bbox_margins(path: Path) -> dict[str, float]:
    image = np.asarray(Image.open(path).convert("RGB"))
    visible = np.any(image < 247, axis=2)
    ys, xs = np.where(visible)
    height, width = visible.shape
    return {
        "top": float(ys.min() / height),
        "bottom": float((height - 1 - ys.max()) / height),
        "left": float(xs.min() / width),
        "right": float((width - 1 - xs.max()) / width),
    }


def luminance(hex_colour: str) -> float:
    rgb = np.array([int(hex_colour[index : index + 2], 16) for index in (1, 3, 5)], dtype=float)
    return float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))


def main() -> int:
    matrix = pd.read_csv(MATRIX)
    quintiles = pd.read_csv(QUINTILES)
    build_qa = json.loads(BUILD_QA.read_text(encoding="utf-8"))

    totals = (
        quintiles.groupby(["city_key", "predictor"], observed=True)["n_block_groups"]
        .sum()
        .unstack("predictor")
    )
    expected_totals = matrix.set_index("city_key")["n_block_groups"]
    totals_match = bool(
        all(
            np.array_equal(totals[variable].to_numpy(int), expected_totals.loc[totals.index].to_numpy(int))
            for variable in VARIABLES
        )
    )
    quintile_sets = quintiles.groupby(["city_key", "predictor"], observed=True)["quintile"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    all_quintiles_present = bool((quintile_sets == (1, 2, 3, 4, 5)).all())
    margins = visible_bbox_margins(FIGURE)
    endpoint_luminance_difference = abs(luminance("#96382E") - luminance("#5C9FB6"))

    checks = {
        "common_analysis_units": int(matrix["n_block_groups"].sum()) == 30826,
        "all_50_cities_once": len(matrix) == 50 and matrix["city_key"].nunique() == 50,
        "all_11_variables": all(variable in matrix.columns for variable in VARIABLES),
        "all_550_cells_finite": int(np.isfinite(matrix[VARIABLES].to_numpy(float)).sum()) == 550,
        "all_city_predictor_quintiles_present": all_quintiles_present,
        "same_city_sample_for_every_predictor": totals_match,
        "no_colour_clipping": float(np.abs(matrix[VARIABLES].to_numpy(float)).max()) <= 25.0,
        "old_new_sign_retention_at_least_95pct": float(build_qa["old_new_sign_retention"]) >= 0.95,
        "minimum_font_at_least_5p8pt": float(build_qa["minimum_font_pt"]) >= 5.8,
        "no_text_text_overlap": int(build_qa["text_text_overlap_count"]) == 0,
        "outer_margins_below_3pct": max(margins.values()) < 0.03,
        "grayscale_endpoint_luminance_separation": endpoint_luminance_difference >= 25.0,
        "all_exports_exist": all(
            (ROOT / "figures" / f"Supplementary_Fig5.{suffix}").exists()
            for suffix in ("png", "pdf", "svg", "tif")
        ),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "quality_gate_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "outer_visible_margins_fraction": margins,
        "endpoint_luminance_difference": endpoint_luminance_difference,
        "principal_limitation": (
            "These are descriptive single-attribute quintile contrasts and must not be read as independent or causal effects."
        ),
        "quality_gate": {
            "necessary": True,
            "nonduplicative": True,
            "correct_graphical_grammar": True,
            "data_scope_and_units_clear": True,
            "uncertainty_needed": False,
            "reason_uncertainty_not_drawn": (
                "The panel is an exhaustive descriptive atlas; formal adjusted estimates and uncertainty are reported in the main coefficient figure."
            ),
            "caption_states_limitations": True,
            "submission_grade": bool(all(checks.values())),
        },
    }
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
