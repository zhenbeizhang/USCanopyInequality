from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "07_supplementary_figures"
    / "make_supplementary_figure_11.py"
)
spec = importlib.util.spec_from_file_location("suppfig11_audit", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def overlap_area(a, b) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def main() -> None:
    module.setup_style()
    intervals = pd.read_csv(module.R5_BUDGET / "tables" / "02_equal_city_budget_intervals.csv")
    allocation = pd.read_csv(module.OLD / "source_data" / "Supplementary_Fig11b_allocated_area_by_context.csv")
    draws = pd.read_csv(module.R5_WEIGHTING / "diagnostics" / "result5_weighting_two_level_draws.csv.gz")
    weighting_intervals = pd.read_csv(module.R5_WEIGHTING / "tables" / "result5_weighting_two_level_intervals.csv")

    figure = plt.figure(figsize=(7.20, 5.05), facecolor="white")
    module.budget_panel(figure, intervals)
    module.allocation_context_bars(figure, allocation)
    module.weighting_sensitivity_panel(figure, draws, weighting_intervals)
    figure.canvas.draw()

    renderer = figure.canvas.get_renderer()
    figure_box = figure.bbox
    texts = []
    clipped = []
    for artist in figure.findobj(match=mpl.text.Text):
        label = artist.get_text().strip()
        if not label or not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer=renderer)
        texts.append((label, box, artist.get_fontsize()))
        if box.x0 < figure_box.x0 - 1 or box.y0 < figure_box.y0 - 1 or box.x1 > figure_box.x1 + 1 or box.y1 > figure_box.y1 + 1:
            clipped.append(label)

    overlaps = []
    for index, (label_a, box_a, _) in enumerate(texts):
        for label_b, box_b, _ in texts[index + 1 :]:
            if overlap_area(box_a.expanded(0.96, 0.82), box_b.expanded(0.96, 0.82)) > 2:
                overlaps.append([label_a, label_b])
    plt.close(figure)

    audit = {
        "status": "PASS" if not clipped and not overlaps else "FAIL",
        "no_clipped_text": not clipped,
        "no_text_text_overlap": not overlaps,
        "minimum_font_pt": min(size for _, _, size in texts),
        "clipped": clipped,
        "overlaps": overlaps,
        "quality_gate_status": "PASS" if not clipped and not overlaps else "FAIL",
    }
    output = module.QA / "Supplementary_Fig11_artist_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise RuntimeError("Supplementary Fig. 11 visual audit failed")


if __name__ == "__main__":
    main()
