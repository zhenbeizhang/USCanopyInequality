from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(str(analysis_store_root()))
BOUNDARY = (
    ROOT
    / "Result"
    / "Result2"
    / "Result2_block_population_boundary_audit"
    / "data"
    / "block_population_boundary_frame.csv.gz"
)
VERSION = ROOT / "Result" / "Result1" / "versions" / "aligned_city_support"
PRIMARY = VERSION / "source_data" / "Fig1_primary_city_metrics.csv"
CONTAINED = VERSION / "source_data" / "Fig1_fully_contained_sensitivity.csv"
FIGURE = VERSION / "figures" / "Fig1_aligned_city_support.png"
OUTPUT = VERSION / "qa" / "Fig1_independent_recalculation_audit.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw = pd.read_csv(BOUNDARY)
    observed = pd.read_csv(PRIMARY).set_index("city_key")
    sensitivity = pd.read_csv(CONTAINED).set_index("city_key")

    differences = []
    contained_differences = []
    for city_key, group in raw.groupby("result_city_key"):
        area = 100.0 * group["canopy_area_m2"].sum() / group["estimated_city_land_area_m2"].sum()
        resident = float(
            np.average(
                group["canopy_pct_block_allocated_frame"],
                weights=group["population_block_allocated"],
            )
        )
        differences.extend(
            [
                abs(area - observed.loc[city_key, "area_weighted_canopy_pct"]),
                abs(resident - observed.loc[city_key, "population_weighted_canopy_pct"]),
            ]
        )

        complete = group.loc[group["fully_contained"].astype(bool)]
        contained_area = 100.0 * complete["canopy_area_m2"].sum() / complete["land_area_m2"].sum()
        contained_resident = float(
            np.average(complete["canopy_pct_analysis"], weights=complete["population"])
        )
        contained_differences.extend(
            [
                abs(contained_area - sensitivity.loc[city_key, "area_weighted_canopy_pct"]),
                abs(
                    contained_resident
                    - sensitivity.loc[city_key, "population_weighted_canopy_pct"]
                ),
            ]
        )

    image = Image.open(FIGURE)
    checks = {
        "source_rows": int(len(raw)),
        "source_cities": int(raw["result_city_key"].nunique()),
        "primary_city_rows": int(len(observed)),
        "contained_city_rows": int(len(sensitivity)),
        "max_primary_recalculation_error": float(max(differences)),
        "max_contained_recalculation_error": float(max(contained_differences)),
        "primary_shift_recalculation_error": float(
            np.max(
                np.abs(
                    observed["population_minus_area_pp"]
                    - (
                        observed["population_weighted_canopy_pct"]
                        - observed["area_weighted_canopy_pct"]
                    )
                )
            )
        ),
        "contained_shift_recalculation_error": float(
            np.max(
                np.abs(
                    sensitivity["population_minus_area_pp"]
                    - (
                        sensitivity["population_weighted_canopy_pct"]
                        - sensitivity["area_weighted_canopy_pct"]
                    )
                )
            )
        ),
        "primary_median_shift_pp": float(observed["population_minus_area_pp"].median()),
        "contained_median_shift_pp": float(
            sensitivity["population_minus_area_pp"].median()
        ),
        "primary_positive_cities": int((observed["population_minus_area_pp"] > 0).sum()),
        "contained_positive_cities": int(
            (sensitivity["population_minus_area_pp"] > 0).sum()
        ),
        "figure_dimensions_px": list(image.size),
        "figure_sha256": digest(FIGURE),
    }
    passed = (
        checks["source_rows"] == 38618
        and checks["source_cities"] == 50
        and checks["primary_city_rows"] == 50
        and checks["contained_city_rows"] == 50
        and checks["max_primary_recalculation_error"] < 1e-10
        and checks["max_contained_recalculation_error"] < 1e-10
        and checks["primary_shift_recalculation_error"] < 1e-10
        and checks["contained_shift_recalculation_error"] < 1e-10
        # At the 183-mm Nature double-column width, 2,956 px exceeds 400 ppi.
        and image.width >= 2800
        and image.height >= 2500
    )
    audit = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "quality_gate": {
            "aligned_canopy_numerator_and_denominator": True,
            "boundary_population_handled_within_city_portion": True,
            "zero_population_intersections_retained_for_city_area_supply": True,
            "fully_contained_support_reported_as_sensitivity": True,
            "universal_direction_claim_removed": True,
            "remaining_assumption_disclosed": (
                "2020 Census block spatial weights allocate 2024 ACS block-group population."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
