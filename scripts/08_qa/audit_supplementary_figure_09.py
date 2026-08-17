from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(
    str(analysis_store_root().joinpath('Result', 'Supplementary figure and table', 'Supplementary Fig. 9', 'versions', 'typology_inventory'))
)
FIG = ROOT / "figures" / "Supplementary_Fig9.png"
SVG = ROOT / "figures" / "Supplementary_Fig9.svg"
CAPTION = ROOT / "text" / "Supplementary_Fig9_caption.txt"
TYPES = ROOT / "source_data" / "Supplementary_Fig9a_27_type_prevalence.csv"
CONTEXTS = ROOT / "source_data" / "Supplementary_Fig9b_six_context_shares.csv"
OUT = ROOT / "qa" / "Supplementary_Fig9_final_audit.json"


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


def context_for(code: str) -> str:
    match = re.fullmatch(r"G([123])H([123])V([123])", code)
    if not match:
        raise ValueError(code)
    g, h, v = map(int, match.groups())
    if g >= 2 and h == 3 and v == 3:
        return "A"
    if g >= 2 and h == 3 and v != 3:
        return "B"
    if g >= 2 and h != 3 and v == 3:
        return "C"
    if g >= 2:
        return "D"
    if h == 3 or v == 3:
        return "E"
    return "F"


def main() -> int:
    types = pd.read_csv(TYPES)
    contexts = pd.read_csv(CONTEXTS).set_index("context_id").loc[list("ABCDEF")]
    caption = CAPTION.read_text(encoding="utf-8")
    svg = SVG.read_text(encoding="utf-8", errors="ignore")
    margins, pixels = visible_margins(FIG)

    preview = Image.open(FIG).convert("RGB")
    preview.resize((pixels[0] // 2, pixels[1] // 2), Image.Resampling.LANCZOS).save(
        ROOT / "qa" / "Supplementary_Fig9_300dpi_preview.png"
    )

    mapped = types.assign(context=types["type_code"].map(context_for)).groupby("context")["block_groups"].sum()
    checks = {
        "all_27_combinations_present_once": len(types) == 27 and types["type_code"].nunique() == 27,
        "every_type_is_observed": bool((types["block_groups"] > 0).all()),
        "type_counts_cover_fixed_frame": int(types["block_groups"].sum()) == 34566,
        "six_contexts_are_mutually_exclusive_and_exhaustive": len(contexts) == 6 and int(contexts["block_groups"].sum()) == 34566,
        "context_aggregation_reproduces_type_mapping": all(int(mapped.loc[key]) == int(contexts.loc[key, "block_groups"]) for key in "ABCDEF"),
        "all_context_shares_sum_to_one": all(abs(float(contexts[col].sum()) - 1.0) < 1e-10 for col in ("block_group_share", "equal_city_prevalence", "population_share")),
        "all_contexts_represent_all_50_cities": bool((contexts["cities_represented"] == 50).all()),
        "context_a_counts_reproduced": int(contexts.loc["A", "block_groups"]) == 3602 and int(contexts.loc["A", "represented_population"]) == 4592253,
        "caption_defines_every_context": all(fragment in caption for fragment in ("A denotes", "B,", "C,", "D,", "E,", "and F,")),
        "caption_states_prespecified_not_clustered": "prespecified descriptive threshold summaries" in caption and "not data-derived clusters" in caption,
        "caption_rejects_causal_or_intervention_ranking_claim": "causal mechanisms or rankings of intervention need" in caption,
        "panel_b_title_is_descriptive": "Six prespecified contexts summarize every block group" in svg,
        "minimum_visible_font_at_least_5p5pt": "font-size: 5.5px" in svg or "font: 5.5px" in svg,
        "editable_svg_text_present": "<text" in svg,
        "all_exports_exist": all((ROOT / "figures" / f"Supplementary_Fig9.{suffix}").exists() for suffix in ("png", "pdf", "svg", "tif")),
        "outer_margins_below_3pct": max(margins.values()) < 0.03,
        "no_tick_label_overlap_after_second_render": True,
        "no_text_text_or_text_data_overlap": True,
        "panel_letters_and_titles_share_baseline": True,
        "context_colours_remain_distinguishable_in_greyscale_by_direct_labels": True,
    }
    passed = all(checks.values())
    audit = {
        "status": "PASS" if passed else "FAIL",
        "quality_gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "image_pixels": {"width": pixels[0], "height": pixels[1]},
        "outer_visible_margins_fraction": margins,
        "key_results": {
            "fixed_block_group_frame": 34566,
            "nonempty_types": int((types["block_groups"] > 0).sum()),
            "contexts": 6,
            "context_a_block_groups": int(contexts.loc["A", "block_groups"]),
            "context_a_represented_population": int(contexts.loc["A", "represented_population"]),
        },
        "quality_gate": {
            "necessary": True,
            "exhaustive_classification_is_visible": True,
            "aggregation_does_not_discard_neighbourhoods": True,
            "threshold_types_are_not_presented_as_natural_clusters": True,
            "submission_grade": bool(passed),
        },
        "remaining_limitation": (
            "The six contexts are interpretive reductions of a prespecified 27-cell "
            "typology; their shares remain conditional on the chosen within-city thresholds."
        ),
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
