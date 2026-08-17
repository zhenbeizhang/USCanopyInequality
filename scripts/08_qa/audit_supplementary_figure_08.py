from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "07_supplementary_figures"
    / "make_supplementary_figure_08.py"
)
spec = importlib.util.spec_from_file_location("suppfig8_audit", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def overlap_area(a, b) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def main() -> None:
    module.style()
    data = module.load_data()
    figure = plt.figure(figsize=(module.WIDTH, module.HEIGHT), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.55),
        width_ratios=(1.02, 1.0),
        left=0.085,
        right=0.965,
        bottom=0.135,
        top=0.925,
        wspace=0.35,
        hspace=0.48,
    )
    module.panel_a(figure.add_subplot(grid[0, 0]), data)
    module.panel_b(figure.add_subplot(grid[0, 1]), data)
    module.panel_c(figure.add_subplot(grid[1, :]), data)
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
    for i, (label_a, box_a, _) in enumerate(texts):
        for label_b, box_b, _ in texts[i + 1 :]:
            if overlap_area(box_a.expanded(0.97, 0.84), box_b.expanded(0.97, 0.84)) > 2:
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
    (module.QA / "Supplementary_Fig8_artist_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise RuntimeError("Supplementary Fig. 8 visual audit failed")


if __name__ == "__main__":
    main()
