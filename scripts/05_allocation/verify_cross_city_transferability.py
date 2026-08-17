#!/usr/bin/env python
"""Verify Step 3 table schemas, dimensions and cross-file conclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uscanopy.paths import analysis_store_root


DEFAULT_ROOT = (
    analysis_store_root()
    / "Result"
    / "Result5"
    / "versions"
    / "step03_cross_city_transferability"
)


EXPECTED_ROWS = {
    "data/analysis_city_ledger.csv": 50,
    "data/city_external_predictors.csv": 50,
    "tables/nested_loocv_metrics.csv": 9,
    "tables/city_heldout_predictions.csv": 450,
    "tables/regional_holdout_metrics.csv": 15,
    "tables/regional_heldout_predictions.csv": 150,
    "tables/outcome_uncertainty_metrics.csv": 3000,
    "tables/outcome_uncertainty_summary.csv": 12,
    "tables/permutation_null.csv": 999,
    "tables/permutation_summary.csv": 1,
    "tables/influence_audit.csv": 50,
    "tables/descriptive_standardised_coefficients.csv": 18,
    "tables/regularisation_grid_sensitivity.csv": 3,
    "tables/prespecified_gate_audit.csv": 6,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root

    checks: list[dict[str, object]] = []
    parsed_tables: dict[str, pd.DataFrame] = {}
    for relative, expected_rows in EXPECTED_ROWS.items():
        path = root / relative
        exists = path.is_file()
        parsed = False
        actual_rows = None
        error = None
        if exists:
            try:
                table = pd.read_csv(path)
                parsed_tables[relative] = table
                parsed = True
                actual_rows = len(table)
            except Exception as exc:  # pragma: no cover - audit failure path
                error = repr(exc)
        checks.append(
            {
                "check": f"table_{relative}",
                "pass": bool(exists and parsed and actual_rows == expected_rows),
                "expected": f"parseable CSV with {expected_rows} rows",
                "observed": (
                    f"exists={exists}; parsed={parsed}; rows={actual_rows}; error={error}"
                ),
            }
        )

    ledger = parsed_tables["data/analysis_city_ledger.csv"]
    predictions = parsed_tables["tables/city_heldout_predictions.csv"]
    region_predictions = parsed_tables["tables/regional_heldout_predictions.csv"]
    gates = parsed_tables["tables/prespecified_gate_audit.csv"]

    ledger_cities = set(ledger["city_key"].astype(str))
    prediction_cities = set(predictions["city_key"].astype(str))
    region_cities = set(region_predictions["city_key"].astype(str))
    checks.append(
        {
            "check": "same_50_cities_across_primary_ledgers",
            "pass": bool(
                len(ledger_cities) == 50
                and ledger_cities == prediction_cities == region_cities
            ),
            "expected": "identical set of 50 cities",
            "observed": (
                f"ledger={len(ledger_cities)}; loocv={len(prediction_cities)}; "
                f"region={len(region_cities)}"
            ),
        }
    )

    bootstrap_path = root / "data" / "outcome_bootstrap_draws.npz"
    bootstrap_shape = None
    bootstrap_finite = False
    if bootstrap_path.is_file():
        with np.load(bootstrap_path, allow_pickle=False) as archive:
            bootstrap_shape = tuple(archive["values"].shape)
            bootstrap_finite = bool(np.isfinite(archive["values"]).all())
    checks.append(
        {
            "check": "bootstrap_draw_archive",
            "pass": bool(bootstrap_shape == (1000, 50, 3) and bootstrap_finite),
            "expected": "finite values with shape (1000, 50, 3)",
            "observed": f"shape={bootstrap_shape}; finite={bootstrap_finite}",
        }
    )

    raw_climate = list((root / "raw_climate").glob("*.json"))
    valid_climate = 0
    for path in raw_climate:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            parameters = payload["properties"]["parameter"]
            if "T2M" in parameters and "PRECTOTCORR" in parameters:
                valid_climate += 1
        except (KeyError, json.JSONDecodeError):
            pass
    checks.append(
        {
            "check": "raw_climate_archive",
            "pass": bool(len(raw_climate) == 50 and valid_climate == 50),
            "expected": "50 parseable NASA POWER responses containing both variables",
            "observed": f"files={len(raw_climate)}; valid={valid_climate}",
        }
    )

    final_execution = json.loads(
        (root / "qa" / "final_execution_audit.json").read_text(encoding="utf-8")
    )
    independent = json.loads(
        (root / "qa" / "independent_audit_summary.json").read_text(encoding="utf-8")
    )
    assessment = json.loads(
        (root / "qa" / "scientific_quality_audit.json").read_text(encoding="utf-8")
    )
    expected_disposition = "SUPPLEMENTARY_WEAK_SIGNAL_OR_DISCUSSION_BOUNDARY"
    conclusion_consistent = bool(
        final_execution["analysis_completed"]
        and not final_execution["main_text_eligibility"]
        and int(gates["pass"].sum()) == 5
        and len(gates) == 6
        and independent["status"] == "PASS"
        and not assessment["prespecified_confirmatory_gate_pass"]
        and assessment["recommended_disposition"] == expected_disposition
    )
    checks.append(
        {
            "check": "cross_file_scientific_disposition",
            "pass": conclusion_consistent,
            "expected": "analysis complete; 5/6 gates; no main-text predictive claim",
            "observed": (
                f"execution={final_execution['status']}; gates={int(gates['pass'].sum())}/"
                f"{len(gates)}; independent={independent['status']}; "
                f"disposition={assessment['recommended_disposition']}"
            ),
        }
    )

    protocol = root / "methods" / "STEP03_FROZEN_PROTOCOL.md"
    required_text = [
        root / "README.md",
        root / "reports" / "STEP03_RESULT_AND_GATE.md",
        root / "reports" / "STEP03_TECHNICAL_AUDIT.md",
        root / "reports" / "STEP03_CHINESE_DECISION.md",
        protocol,
    ]
    checks.append(
        {
            "check": "required_human_readable_documents",
            "pass": all(path.is_file() and path.stat().st_size > 0 for path in required_text),
            "expected": "README, protocol and three result/review reports present",
            "observed": "; ".join(
                f"{path.name}={path.is_file() and path.stat().st_size > 0}"
                for path in required_text
            ),
        }
    )

    table = pd.DataFrame(checks)
    table.to_csv(root / "qa" / "final_package_integrity_checks.csv", index=False)
    summary = {
        "status": "PASS" if bool(table["pass"].all()) else "FAIL",
        "checks_passed": int(table["pass"].sum()),
        "checks_total": int(len(table)),
        "analysis_complete": True,
        "confirmatory_gate_pass": False,
        "recommended_disposition": expected_disposition,
    }
    (root / "qa" / "final_package_integrity.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
