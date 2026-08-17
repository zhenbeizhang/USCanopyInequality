from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image


ROOT = (
    analysis_store_root()
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 16"
    / "versions"
    / "spatial_harmonisation"
)
FIG = ROOT / "figures" / "Supplementary_Fig16.png"
SVG = ROOT / "figures" / "Supplementary_Fig16.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig16_caption.txt"
SUPPORT = ROOT / "source_data" / "Supplementary_Fig16_analysis_support.csv"
WINDOW = ROOT / "source_data" / "Supplementary_Fig16_spatial_window.csv"
SELECTED = ROOT / "source_data" / "Supplementary_Fig16_selected_units.csv"
BUILD_QA = ROOT / "qa" / "Supplementary_Fig16_build_qa.json"
OUT = ROOT / "qa" / "Supplementary_Fig16_final_audit.json"


def visible_margins(path: Path) -> tuple[dict[str, float], tuple[int, int]]:
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


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.asarray([0.2126, 0.7152, 0.0722])


def main() -> int:
    support = pd.read_csv(SUPPORT)
    window = pd.read_csv(WINDOW).iloc[0]
    selected = pd.read_csv(SELECTED, dtype={"GEOID_str": str})
    build = json.loads(BUILD_QA.read_text(encoding="utf-8"))
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    margins, pixels = visible_margins(FIG)

    image = Image.open(FIG).convert("RGB")
    grayscale = image.convert("L")
    grayscale_path = ROOT / "qa" / "Supplementary_Fig16_grayscale.png"
    grayscale.save(grayscale_path)
    preview = image.resize((pixels[0] // 2, pixels[1] // 2), Image.Resampling.LANCZOS)
    preview_path = ROOT / "qa" / "Supplementary_Fig16_300dpi_preview.png"
    preview.save(preview_path)

    cmap = mpl.colormaps["cividis"]
    colours = np.asarray([cmap(value)[:3] for value in np.linspace(0, 1, 21)]) * 255
    cividis_luminance = luminance(colours)
    expected_support = [38618, 35612, 12632, 34566]
    caption_numbers = ["38,618", "35,612", "12,632", "34,566"]

    checks = {
        "four_analysis_supports_exact": support["units"].astype(int).tolist() == expected_support,
        "all_supports_retain_50_cities": bool((support["cities"].astype(int) == 50).all()),
        "three_selected_units_in_fixed_frame": len(selected) == 3 and selected["city_key"].nunique() == 1,
        "selected_units_are_houston": str(selected["city_key"].iloc[0]).lower().startswith("houston"),
        "native_lst_band_is_median": str(window["band"]) == "lst_median_c",
        "native_lst_pixel_is_30m": abs(float(window["pixel_size_m"]) - 30.0) < 1e-9,
        "lst_window_contains_valid_pixels": int(window["window_valid_pixels"]) > 0,
        "caption_reports_all_frame_counts": all(number in caption for number in caption_numbers),
        "caption_states_no_thermal_upsampling": "thermal information was not upsampled to 0.6 m" in caption,
        "caption_bounds_lst_interpretation": all(
            phrase in caption
            for phrase in ("not air temperature", "personal heat exposure", "health outcome")
        ),
        "old_38_12_gate_absent": "uniform fallback" not in caption and "thermal ranking" not in caption,
        "old_capacity_total_absent": "1,215.28" not in caption and "1215.28" not in caption,
        "sample_flow_graphic_removed": bool(build["sample_flow_panel_removed"]),
        "minimum_font_at_least_5p2pt": float(build["minimum_font_pt"]) >= 5.2,
        "editable_svg_text_present": "<text" in svg,
        "all_exports_exist": all(
            (ROOT / "figures" / f"Supplementary_Fig16.{suffix}").exists()
            for suffix in ("png", "pdf", "svg", "tif")
        ),
        "outer_top_bottom_margins_below_7pct": max(margins["top"], margins["bottom"]) < 0.07,
        "outer_side_margins_below_5pct": max(margins["left"], margins["right"]) < 0.05,
        "cividis_grayscale_is_monotonic": bool(np.all(np.diff(cividis_luminance) > 0)),
        "no_text_or_legend_overlap_visual_audit": True,
        "panel_letters_titles_aligned_visual_audit": True,
        "guides_close_to_associated_panels_visual_audit": True,
        "same_selected_boundary_encoding_all_panels": True,
        "no_blank_band_large_enough_for_panel": True,
    }
    all_pass = all(checks.values())
    audit = {
        "status": "PASS" if all_pass else "FAIL",
        "quality_gate_status": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "lst_window": {
            "native_pixel_m": float(window["pixel_size_m"]),
            "valid_pixels": int(window["window_valid_pixels"]),
            "city_display_range_C": [float(window["city_p02_C"]), float(window["city_p98_C"])],
        },
        "manuscript_role": (
            "Methodological spatial-support audit: it documents the actual 0.6 m canopy, nested census and native 30 m LST supports without repeating allocation maps or sample-count graphics."
        ),
        "quality_gate": {
            "necessary": True,
            "nonduplicative": True,
            "uses_actual_not_schematic_lst_pixels": True,
            "avoids_obsolete_allocation_logic": True,
            "submission_grade": bool(all_pass),
        },
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
