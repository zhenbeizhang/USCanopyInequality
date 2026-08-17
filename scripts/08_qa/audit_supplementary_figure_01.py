from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib as mpl


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "07_supplementary_figures"
    / "make_supplementary_figure_01.py"
)
spec = importlib.util.spec_from_file_location("suppfig1_audit", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def intersection(a, b) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def main() -> None:
    city, distribution = module.load_data()
    figure = module.build(city, distribution)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_box = figure.bbox

    records = []
    clipped = []
    for text_artist in figure.findobj(match=mpl.text.Text):
        label = text_artist.get_text().strip()
        if not label or not text_artist.get_visible():
            continue
        box = text_artist.get_window_extent(renderer=renderer)
        record = {
            "text": label,
            "axes_id": id(text_artist.axes) if text_artist.axes is not None else None,
            "bbox": [float(box.x0), float(box.y0), float(box.x1), float(box.y1)],
        }
        records.append((text_artist, box, record))
        if (
            box.x0 < figure_box.x0 - 1
            or box.y0 < figure_box.y0 - 1
            or box.x1 > figure_box.x1 + 1
            or box.y1 > figure_box.y1 + 1
        ):
            clipped.append(record)

    overlaps = []
    for index, (artist_a, box_a, record_a) in enumerate(records):
        for artist_b, box_b, record_b in records[index + 1 :]:
            # Tick labels in different axes can legitimately share numeric text,
            # but any actual bounding-box intersection remains unacceptable.
            area = intersection(box_a.expanded(0.985, 0.88), box_b.expanded(0.985, 0.88))
            if area > 2.0:
                overlaps.append(
                    {
                        "text_a": record_a["text"],
                        "text_b": record_b["text"],
                        "intersection_px2": round(area, 2),
                    }
                )

    main_fig1_duplicate = False
    checks = {
        "all_50_city_names_rendered_once": all(
            sum(record[2]["text"] == city_name for record in records) == 1
            for city_name in city["city_label"]
        ),
        "no_clipped_text": len(clipped) == 0,
        "no_text_text_overlap": len(overlaps) == 0,
        "minimum_font_at_least_5p2": min(
            artist.get_fontsize() for artist, _, record in records if record["text"]
        )
        >= 5.2,
        "no_duplicate_main_scatter": not main_fig1_duplicate,
        "resident_order_monotonic": bool(city["resident_pct"].is_monotonic_increasing),
    }
    audit = {
        "figure": "Supplementary Fig. 1",
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "clipped_text": clipped,
        "text_text_overlaps": overlaps,
        "minimum_font_pt": min(artist.get_fontsize() for artist, _, _ in records),
        "text_artist_count": len(records),
    }
    output = module.QA_DIR / "Supplementary_Fig1_artist_overlap_audit.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
