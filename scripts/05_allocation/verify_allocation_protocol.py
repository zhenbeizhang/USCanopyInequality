from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'step1_frozen_multicriteria_design')))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol_path = ROOT / "methods" / "Result5_FROZEN_ANALYSIS_PROTOCOL.md"
    ledger_path = ROOT / "methods" / "Result5_TERMINOLOGY_LEDGER.csv"
    design_path = ROOT / "qa" / "step1_design_audit.json"
    outcomes_path = ROOT / "tables" / "01_city_rule_and_pareto_outcomes.csv"
    national_path = ROOT / "tables" / "04_equal_city_rule_summary.csv"
    frontier_path = ROOT / "tables" / "05_equal_city_pareto_frontier.csv"

    protocol = protocol_path.read_text(encoding="utf-8")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    outcomes = pd.read_csv(outcomes_path)
    national = pd.read_csv(national_path)
    frontier = pd.read_csv(frontier_path)
    with ledger_path.open(encoding="utf-8", newline="") as stream:
        ledger = list(csv.DictReader(stream))

    checks = {
        "design_audit_passed": design["status"] == "PASS" and design["quality_gate_passed"],
        "fifty_cities": outcomes["city_key"].nunique() == 50,
        "all_expected_outcome_rows": len(outcomes) == 50 * (7 + 21),
        "seven_named_rules": len(national) == 7,
        "twenty_one_pareto_points": len(frontier) == 21,
        "budget_is_ten_percent": set(outcomes["budget_fraction"].round(12)) == {0.1},
        "protocol_exposes_population_oracle": "population oracle" in protocol.lower(),
        "protocol_separates_three_axes": all(
            phrase in protocol
            for phrase in (
                "heat-context-weighted",
                "social-condition-weighted",
                "population-weighted canopy-context increments",
            )
        ),
        "protocol_retires_legacy_metric": "retired from primary inference" in protocol,
        "protocol_denies_treatment_effect": "It is not a treatment effect" in protocol,
        "protocol_freezes_same_v_for_result4": "same frozen score will be used when Result 4" in protocol,
        "terminology_ledger_nonempty": len(ledger) >= 10,
        "terminology_ledger_unique_concepts": len({(r["concept"], r["permitted_term"]) for r in ledger}) == len(ledger),
    }

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (protocol_path, ledger_path, design_path, outcomes_path, national_path, frontier_path)
        },
        "outcome_rows": len(outcomes),
        "cities": int(outcomes["city_key"].nunique()),
        "named_rules": int(len(national)),
        "pareto_points": int(len(frontier)),
    }
    output = ROOT / "qa" / "step1_freeze_verification.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
