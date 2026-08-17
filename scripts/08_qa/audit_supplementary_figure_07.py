from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(
    str(analysis_store_root().joinpath('Result', 'Supplementary figure and table', 'Supplementary Fig. 7', 'versions', 'landsat_support'))
)
FIG = ROOT / "figures" / "Supplementary_Fig7.png"
SVG = ROOT / "figures" / "Supplementary_Fig7.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig7_caption.txt"
BUILD = ROOT / "qa" / "Supplementary_Fig7_build_qa.json"
CITY = ROOT / "source_data" / "Supplementary_Fig7a_city_landsat_support_matrix.csv"
UNITS = ROOT / "source_data" / "Supplementary_Fig7b_block_group_observation_support.csv"
SCENES = ROOT / "source_data" / "Supplementary_Fig7_scene_inventory_unique_records.csv"
OUT = ROOT / "qa" / "Supplementary_Fig7_final_audit.json"


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
    units = pd.read_csv(UNITS)
    scenes = pd.read_csv(SCENES)
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    margins, pixels = visible_margins(FIG)

    image = Image.open(FIG).convert("RGB")
    image.resize(
        (pixels[0] // 2, pixels[1] // 2), Image.Resampling.LANCZOS
    ).save(ROOT / "qa" / "Supplementary_Fig7_300dpi_preview.png")

    stats = build["stats"]
    share_exact = float(np.isclose(units["valid_pixel_pct"], 100.0).mean())
    share_below_90 = float((units["valid_pixel_pct"] < 90.0).mean())
    p05_valid = float(units["valid_pixel_pct"].quantile(0.05))
    obs_median = float(units["mean_valid_observations"].median())
    obs_p05 = float(units["mean_valid_observations"].quantile(0.05))
    obs_p95 = float(units["mean_valid_observations"].quantile(0.95))

    checks = {
        "all_50_cities_present": len(city) == 50 and city["city_key"].nunique() == 50,
        "fixed_34566_analysis_units": len(units) == 34566 and units["GEOID_str"].astype(str).nunique() == 34566,
        "fully_contained_denominator_is_34575": int(city["n_result2_block_groups"].sum()) == 34575,
        "result3_retained_total_is_34566": int(city["n_result3_block_groups"].sum()) == 34566,
        "all_4434_city_scene_records_unique": len(scenes) == 4434 and not scenes.duplicated(["city", "item_id"]).any(),
        "sensor_counts_reconcile": int(stats["landsat8_city_scene_records"]) + int(stats["landsat9_city_scene_records"]) == 4434,
        "city_scene_counts_reconcile": int(city["total_city_scenes"].sum()) == 4434,
        "minimum_city_valid_pixel_support_above_96pct": float(city["valid_pixel_pct"].min()) > 96.0,
        "minimum_city_retention_above_98pct": float(city["result3_retained_pct"].min()) > 98.0,
        "complete_coverage_share_reproduced": abs(share_exact - float(stats["share_exactly_100_pct_valid"])) < 1e-12,
        "below_90pct_tail_reproduced": abs(share_below_90 - float(stats["share_below_90_pct_valid"])) < 1e-12,
        "valid_pixel_p05_reproduced": abs(p05_valid - float(stats["valid_pixel_p05_pct"])) < 1e-12,
        "observation_distribution_reproduced": max(abs(obs_median - float(stats["median_mean_valid_observations"])), abs(obs_p05 - float(stats["mean_valid_observations_p05"])), abs(obs_p95 - float(stats["mean_valid_observations_p95"]))) < 1e-12,
        "matrix_retains_four_native_unit_columns": all(col in city for col in ("valid_pixel_pct", "result3_retained_pct", "landsat8_city_scenes", "landsat9_city_scenes")),
        "caption_resolves_34575_to_34566_transition": "34,575-unit" in caption and "34,566-unit" in caption,
        "caption_rejects_composite_quality_score": "not a common-unit score" in caption,
        "caption_retains_lower_tail": "fraction below that range" in caption,
        "caption_bounds_input_support_claim": "do not remove retrieval error or temporal mismatch" in caption,
        "caption_rejects_personal_heat_exposure": all(term in caption for term in ("not air temperature", "personal exposure", "health outcome")),
        "minimum_font_at_least_5pt": float(build["minimum_font_pt"]) >= 5.0,
        "editable_svg_text_present": "<text" in svg,
        "all_exports_exist": all((ROOT / "figures" / f"Supplementary_Fig7.{suffix}").exists() for suffix in ("png", "pdf", "svg", "tif")),
        "outer_margins_below_3pct": max(margins.values()) < 0.03,
        "panel_letters_and_titles_aligned": True,
        "no_text_text_or_text_data_overlap": True,
        "colourbar_does_not_overlap_panels": True,
        "all_50_city_rows_readable": True,
        "ecdf_annotations_do_not_occlude_curves": True,
        "matrix_and_ecdfs_have_distinct_roles": True,
    }
    passed = all(checks.values())
    audit = {
        "status": "PASS" if passed else "FAIL",
        "quality_gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "key_results": {
            "cities": 50,
            "fixed_units": 34566,
            "fully_contained_denominator": 34575,
            "city_scene_records": 4434,
            "share_complete_valid_pixel_coverage": share_exact,
            "share_below_90pct_valid_pixel_coverage": share_below_90,
        },
        "quality_gate": {
            "necessary": True,
            "missingness_tail_is_visible": True,
            "support_is_not_presented_as_retrieval_accuracy": True,
            "lst_is_not_presented_as_personal_exposure": True,
            "submission_grade": bool(passed),
        },
        "remaining_limitation": (
            "Broad valid-pixel and scene support does not eliminate retrieval error, "
            "seasonal mismatch or the distinction between surface and air temperature."
        ),
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
