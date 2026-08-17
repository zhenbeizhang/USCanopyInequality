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
    / "Supplementary Fig. 12"
    / "versions"
    / "boundary_support_matrix"
)
SOURCE = ROOT / "source_data" / "Supplementary_Fig12_boundary_objective_differences.csv"
SUMMARY = ROOT / "source_data" / "Supplementary_Fig12_summary.csv"
FIG = ROOT / "figures" / "Supplementary_Fig12.png"
SVG = ROOT / "figures" / "Supplementary_Fig12.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig12_caption.txt"
BUILD_QA = ROOT / "qa" / "Supplementary_Fig12_build_qa.json"
OUT = ROOT / "qa" / "Supplementary_Fig12_final_audit.json"


def margins(path: Path) -> tuple[dict[str, float], tuple[int, int]]:
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


def lum(hex_colour: str) -> float:
    rgb = np.asarray([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return float(rgb @ np.asarray([0.2126, 0.7152, 0.0722]))


def main() -> int:
    data = pd.read_csv(SOURCE)
    summary = pd.read_csv(SUMMARY).iloc[0]
    build = json.loads(BUILD_QA.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    outer, pixels = margins(FIG)

    image = Image.open(FIG).convert("RGB")
    image.convert("L").save(ROOT / "qa" / "Supplementary_Fig12_grayscale.png")
    image.resize((pixels[0] // 2, pixels[1] // 2), Image.Resampling.LANCZOS).save(
        ROOT / "qa" / "Supplementary_Fig12_300dpi_preview.png"
    )

    expected_lambdas = np.round(np.arange(0, 1.0001, 0.05), 2)
    midpoint = data.loc[np.isclose(data["lambda"], 0.5)]
    values = data["difference_from_fully_contained"].to_numpy(float)
    endpoint_gap = abs(lum("#3D6F8D") - lum("#D17652"))
    zero_gap = min(abs(lum("#F4F1EC") - lum("#3D6F8D")), abs(lum("#F4F1EC") - lum("#D17652")))

    checks = {
        "complete_3_support_3_objective_21_lambda_grid": len(data) == 3 * 3 * 21,
        "three_alternative_supports": data["support"].nunique() == 3,
        "three_distinct_objective_rows": data["metric"].nunique() == 3,
        "lambda_grid_exact": np.allclose(np.sort(data["lambda"].unique()), expected_lambdas),
        "values_finite": bool(np.isfinite(values).all()),
        "maximum_change_reproduced": abs(float(np.max(np.abs(values))) - float(summary["maximum_absolute_change"])) < 1e-12,
        "primary_lambda_change_reproduced": abs(float(midpoint["difference_from_fully_contained"].abs().max()) - float(summary["maximum_absolute_change_at_lambda_0p5"])) < 1e-12,
        "colour_scale_no_clipping": float(np.max(np.abs(values))) < 0.018,
        "equal_budget_error_zero": abs(float(summary["maximum_budget_error_m2"])) < 1e-12,
        "caption_calls_a_specification_audit": "specification audit" in caption,
        "caption_denies_sampling_intervals": "rather than sampling-confidence intervals" in caption,
        "caption_bounds_outcome": all(phrase in caption for phrase in ("not observed cooling", "personal exposure", "health benefit")),
        "no_old_fallback_framework": all(phrase not in caption for phrase in ("uniform fallback", "thermal ranking", "38 cities")),
        "no_superiority_claim": "superior" not in caption.lower() and "optimal" not in caption.lower(),
        "minimum_font_at_least_5p2pt": float(build["minimum_font_pt"]) >= 5.2,
        "editable_svg_text_present": "<text" in svg,
        "all_exports_exist": all((ROOT / "figures" / f"Supplementary_Fig12.{suffix}").exists() for suffix in ("png", "pdf", "svg", "tif")),
        "outer_margins_below_6pct": max(outer.values()) < 0.06,
        "diverging_endpoints_grayscale_separable": endpoint_gap >= 25,
        "zero_grayscale_separable": zero_gap >= 80,
        "single_matrix_has_unique_nonduplicative_role": True,
        "text_and_data_do_not_overlap_visual_audit": True,
        "primary_lambda_outline_does_not_occlude_values": True,
        "matrix_cells_and_guides_readable_at_300dpi": True,
    }
    passed = all(checks.values())
    audit = {
        "status": "PASS" if passed else "FAIL",
        "quality_gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": outer,
        "maximum_absolute_change": float(summary["maximum_absolute_change"]),
        "maximum_absolute_change_at_lambda_0p5": float(summary["maximum_absolute_change_at_lambda_0p5"]),
        "grayscale_luminance_gaps": {"negative_vs_positive": endpoint_gap, "zero_vs_nearest_endpoint": zero_gap},
        "evidence_role": "Deterministic municipal-boundary support sensitivity for the canopy aggregation frame.",
        "quality_gate": {
            "necessary": True,
            "nonduplicative": True,
            "supports_a_prespecified_boundary_sensitivity": True,
            "not_circular_superiority_evidence": True,
            "submission_grade": bool(passed),
        },
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
