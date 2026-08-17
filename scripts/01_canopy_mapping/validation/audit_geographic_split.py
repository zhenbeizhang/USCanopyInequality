"""Audit geographic separation of the recovered untouched test subset."""

from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(
    str(analysis_store_root().joinpath('Result', 'Result1', 'versions', 'step3_canopy_recovery_validation'))
)
INVENTORY = OUT / "source_data" / "annotation_split_inventory.csv"
BUFFER_M = 250.0


def box_distance(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    """Euclidean distance between one and many axis-aligned boxes."""
    dx = np.maximum.reduce(
        [many[:, 0] - one[2], one[0] - many[:, 2], np.zeros(len(many))]
    )
    dy = np.maximum.reduce(
        [many[:, 1] - one[3], one[1] - many[:, 3], np.zeros(len(many))]
    )
    return np.hypot(dx, dy)


def main() -> None:
    frame = pd.read_csv(INVENTORY)
    if set(frame["crs"]) != {"EPSG:3395"}:
        raise RuntimeError(f"Unexpected CRS values: {sorted(frame['crs'].unique())}")
    dev = frame.loc[
        frame["split"].isin(["train", "checkpoint_selection"]),
        ["filename", "base_id", "split", "left", "bottom", "right", "top"],
    ].copy()
    test = frame.loc[
        frame["split"].eq("model_selection_independent") & frame["fully_untouched"],
        ["filename", "base_id", "left", "bottom", "right", "top"],
    ].copy()
    dev_boxes = dev[["left", "bottom", "right", "top"]].to_numpy(float)
    rows = []
    for base_id, patches in test.groupby("base_id", sort=True):
        minimum = np.inf
        closest_index = -1
        closest_patch = None
        for _, patch in patches.iterrows():
            distances = box_distance(
                patch[["left", "bottom", "right", "top"]].to_numpy(float), dev_boxes
            )
            index = int(np.argmin(distances))
            if distances[index] < minimum:
                minimum = float(distances[index])
                closest_index = index
                closest_patch = patch["filename"]
        closest = dev.iloc[closest_index]
        rows.append(
            {
                "test_base_id": base_id,
                "test_patches": len(patches),
                "closest_test_patch": closest_patch,
                "minimum_distance_to_model_development_m": minimum,
                "within_250m_of_model_development": minimum < BUFFER_M,
                "overlaps_model_development": minimum == 0,
                "closest_development_split": closest["split"],
                "closest_development_base_id": closest["base_id"],
                "closest_development_patch": closest["filename"],
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        OUT / "tables" / "geographic_split_separation_by_test_base_tile.csv",
        index=False,
    )
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "crs": "EPSG:3395",
        "untouched_test_base_tiles": int(len(result)),
        "untouched_test_patches": int(len(test)),
        "overlapping_base_tiles": int(result["overlaps_model_development"].sum()),
        "base_tiles_within_250m": int(
            result["within_250m_of_model_development"].sum()
        ),
        "minimum_distance_m": float(
            result["minimum_distance_to_model_development_m"].min()
        ),
        "median_minimum_distance_m": float(
            result["minimum_distance_to_model_development_m"].median()
        ),
        "geographic_250m_separation_pass": bool(
            (~result["within_250m_of_model_development"]).all()
        ),
    }
    with (OUT / "qa" / "step3_geographic_split_audit.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    (OUT / "scripts" / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
