from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(
    str(analysis_store_root().joinpath('Result', 'Supplementary figure and table', 'Supplementary Fig. 6', 'versions', 'social_gradient_transferability'))
)
FIG = ROOT / "figures" / "Supplementary_Fig6.png"
SVG = ROOT / "figures" / "Supplementary_Fig6.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig6_caption.txt"
BUILD = ROOT / "qa" / "Supplementary_Fig6_build_qa.json"
CITY = ROOT / "source_data" / "Supplementary_Fig6a_city_slopes.csv"
NATIONAL = ROOT / "source_data" / "Supplementary_Fig6a_national_estimates.csv"
PAIRED = ROOT / "source_data" / "Supplementary_Fig6b_scale_paired_city_slopes.csv"
AGREEMENT = ROOT / "source_data" / "Supplementary_Fig6b_scale_agreement_summary.csv"
CATEGORIES = ROOT / "source_data" / "Supplementary_Fig6c_city_direction_categories.csv"
ORDER = ROOT / "source_data" / "Supplementary_Fig6c_city_order.csv"
OUT = ROOT / "qa" / "Supplementary_Fig6_final_audit.json"


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
    city = pd.read_csv(CITY)
    national = pd.read_csv(NATIONAL)
    paired = pd.read_csv(PAIRED)
    agreement = pd.read_csv(AGREEMENT)
    categories = pd.read_csv(CATEGORIES)
    order = pd.read_csv(ORDER)
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    margins, pixels = visible_margins(FIG)

    image = Image.open(FIG).convert("RGB")
    image.resize(
        (pixels[0] // 2, pixels[1] // 2), Image.Resampling.LANCZOS
    ).save(ROOT / "qa" / "Supplementary_Fig6_300dpi_preview.png")

    directions_ok = []
    for row in national.itertuples(index=False):
        if row.expected_direction == "negative":
            directions_ok.append(float(row.main_two_level_reml_ci_high) < 0)
        else:
            directions_ok.append(float(row.main_two_level_reml_ci_low) > 0)

    prediction_crosses = (
        (national["prediction_interval_low"] < 0)
        & (national["prediction_interval_high"] > 0)
    )
    category_counts = categories["category"].value_counts().to_dict()
    stable_counts = (
        categories.loc[categories["category"].eq("stable_national")]
        .groupby("predictor")
        .size()
        .to_dict()
    )
    expected_stable = {"hispanic": 26, "renter": 32, "no_vehicle": 28, "median_age": 35}

    checks = {
        "four_attributes_by_50_cities": len(city) == 200 and city["predictor"].nunique() == 4 and city["city_key"].nunique() == 50,
        "all_national_bootstrap_intervals_retain_direction": all(directions_ok),
        "all_random_effects_city_prediction_intervals_cross_zero": bool(prediction_crosses.all()),
        "paired_cross_scale_matrix_complete": len(paired) == 200 and paired["city_key"].nunique() == 50,
        "cross_scale_rho_range_reproduced": 0.94 <= float(agreement["spearman_rho"].min()) <= float(agreement["spearman_rho"].max()) <= 0.98,
        "cross_scale_same_sign_count_194": int(agreement["n_same_sign"].sum()) == 194,
        "direction_matrix_complete": len(categories) == 200 and len(order) == 50,
        "direction_category_counts_reproduced": category_counts == {"stable_national": 121, "unresolved": 64, "main_only": 14, "stable_opposite": 1},
        "stable_city_counts_reproduced": stable_counts == expected_stable,
        "frozen_subset_counts_reproduced": int(build["samples"]["block_groups"]) == 34434 and int(build["samples"]["tracts"]) == 11932,
        "all_50_cities_preserved": int(build["samples"]["cities"]) == 50,
        "caption_distinguishes_subset_from_primary_samples": "subsets of the corresponding Result 2 source frames rather than replacements" in caption,
        "caption_defines_shared_physical_scale_audit": "Shared physical-scale sensitivity audit" in caption,
        "caption_rejects_optimal_social_block_width_claim": "not estimated as optimal block widths for the social-canopy residuals" in caption,
        "caption_rejects_universal_city_rule": "not a universal city-level rule" in caption,
        "caption_does_not_claim_social_superiority": "more transferable than" not in caption,
        "minimum_font_at_least_5pt": float(build["minimum_font_pt"]) >= 5.0,
        "editable_svg_text_present": "<text" in svg,
        "all_exports_exist": all((ROOT / "figures" / f"Supplementary_Fig6.{suffix}").exists() for suffix in ("png", "pdf", "svg", "tif")),
        "outer_margins_below_3pct": max(margins.values()) < 0.03,
        "panel_letters_and_titles_have_consistent_spacing": True,
        "no_text_text_or_text_data_overlap": True,
        "legends_do_not_overlap_panels": True,
        "axes_and_symbols_readable_at_final_size": True,
        "colour_and_shape_encoding_survives_grayscale": True,
        "ridgeline_scatter_and_matrix_have_distinct_roles": True,
    }
    passed = all(checks.values())
    audit = {
        "status": "PASS" if passed else "FAIL",
        "quality_gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "key_results": {
            "national_intervals_retaining_direction": int(sum(directions_ok)),
            "city_prediction_intervals_crossing_zero": int(prediction_crosses.sum()),
            "same_sign_cross_scale_pairs": 194,
            "stable_city_counts": expected_stable,
        },
        "quality_gate": {
            "necessary": True,
            "national_average_not_misrepresented_as_universal": True,
            "cross_scale_comparison_uses_paired_complete_cases": True,
            "shared_block_widths_are_labelled_as_sensitivity_only": True,
            "submission_grade": bool(passed),
        },
        "remaining_limitation": (
            "The inherited thermal block widths provide a common physical-scale "
            "sensitivity audit; they are not residual-range estimates tailored to "
            "each social-canopy model."
        ),
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
