"""Summarise the recovered, incomplete tile-level acquisition dates."""

from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUT = Path(
    str(analysis_store_root().joinpath('Result', 'Result1', 'versions', 'step3_canopy_recovery_validation'))
)


def main() -> None:
    summary = pd.read_csv(
        OUT / "tables" / "imagery_acquisition_date_summary_by_city.csv"
    )
    counts = pd.read_csv(
        OUT / "source_data" / "imagery_acquisition_date_counts_by_city.csv"
    )
    counts["year"] = counts["acquisition_date"].str[:4].astype(int)
    grouped = []
    for city, frame in counts.groupby("city_task_name", sort=True):
        total = frame["tile_count"].sum()
        grouped.append(
            {
                "city_task_name": city,
                "tagged_tiles_2018_2024": int(
                    frame.loc[frame["year"].between(2018, 2024), "tile_count"].sum()
                ),
                "tagged_tiles_2020_2024": int(
                    frame.loc[frame["year"].between(2020, 2024), "tile_count"].sum()
                ),
                "share_of_tagged_tiles_2018_2024": (
                    frame.loc[frame["year"].between(2018, 2024), "tile_count"].sum()
                    / total
                ),
                "share_of_tagged_tiles_2020_2024": (
                    frame.loc[frame["year"].between(2020, 2024), "tile_count"].sum()
                    / total
                ),
            }
        )
    timing = summary.merge(pd.DataFrame(grouped), on="city_task_name", how="left")
    timing[[
        "tagged_tiles_2018_2024",
        "tagged_tiles_2020_2024",
        "share_of_tagged_tiles_2018_2024",
        "share_of_tagged_tiles_2020_2024",
    ]] = timing[[
        "tagged_tiles_2018_2024",
        "tagged_tiles_2020_2024",
        "share_of_tagged_tiles_2018_2024",
        "share_of_tagged_tiles_2020_2024",
    ]].fillna(0)
    timing["tagged_tile_share_of_jpegs"] = (
        timing["acquisition_date_tags"] / timing["jpeg_signatures"]
    )
    timing["known_2018_2024_lower_bound_share_of_jpegs"] = (
        timing["tagged_tiles_2018_2024"] / timing["jpeg_signatures"]
    )
    timing.to_csv(
        OUT / "tables" / "imagery_timing_completeness_by_city.csv",
        index=False,
    )
    total_tags = counts["tile_count"].sum()
    total_jpegs = timing["jpeg_signatures"].sum()
    total_2018_2024 = counts.loc[
        counts["year"].between(2018, 2024), "tile_count"
    ].sum()
    total_2020_2024 = counts.loc[
        counts["year"].between(2020, 2024), "tile_count"
    ].sum()
    qa = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cities": int(len(timing)),
        "cities_with_any_date_tags": int((timing["acquisition_date_tags"] > 0).sum()),
        "overall_tagged_tile_share_of_jpeg_signatures": float(
            total_tags / total_jpegs
        ),
        "share_of_tagged_tiles_2018_2024": float(total_2018_2024 / total_tags),
        "share_of_tagged_tiles_2020_2024": float(total_2020_2024 / total_tags),
        "known_2018_2024_lower_bound_share_of_jpeg_signatures": float(
            total_2018_2024 / total_jpegs
        ),
        "cities_with_at_least_90pct_tagged_dates_in_2018_2024": int(
            (timing["share_of_tagged_tiles_2018_2024"] >= 0.90).sum()
        ),
        "important_limit": (
            "Date tags are absent from some cached JPEGs and from all Washington "
            "tiles; tagged-date shares cannot be treated as complete acquisition "
            "inventories."
        ),
    }
    with (OUT / "qa" / "step3_imagery_timing_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(qa, stream, indent=2, ensure_ascii=False)
    (OUT / "scripts" / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
