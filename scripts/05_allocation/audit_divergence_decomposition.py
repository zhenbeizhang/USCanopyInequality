from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from uscanopy.paths import analysis_store_root, repo_root


OUTCOME = "heat_context_opportunity_loss_under_social"
CAPACITY = "capacity_intensity_neutral_minus_observed"
GATE_TOTAL = "candidate_gate_total_matched_all_minus_same"
GATE_RATE = "candidate_gate_rate_matched_all_minus_same"
GATE_ROBUST = "candidate_gate_robust_across_capacity_matching"

RESULT5_VERSIONS = analysis_store_root() / "Result" / "Result5" / "versions"
DEFAULT_STEP1 = RESULT5_VERSIONS / "step01_city_tradeoff_outcomes"
DEFAULT_STEP2 = RESULT5_VERSIONS / "step02_divergence_decomposition"
DEFAULT_WEIGHTING = (
    RESULT5_VERSIONS
    / "weighting_heterogeneity_audit"
    / "tables"
    / "result5_weighting_two_level_intervals.csv"
)
DEFAULT_IMPLEMENTATION_NOTE = (
    repo_root() / "docs" / "protocols" / "DIVERGENCE_IMPLEMENTATION_NOTE.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1", type=Path, default=DEFAULT_STEP1)
    parser.add_argument("--step2", type=Path, default=DEFAULT_STEP2)
    parser.add_argument("--weighting", type=Path, default=DEFAULT_WEIGHTING)
    parser.add_argument(
        "--implementation-note", type=Path, default=DEFAULT_IMPLEMENTATION_NOTE
    )
    parser.add_argument(
        "--manuscript-snapshot-root",
        type=Path,
        default=None,
        help="Optional read-only manuscript directory for an internal hash audit.",
    )
    parser.add_argument(
        "--manuscript-snapshot-manifest",
        type=Path,
        default=None,
        help="Optional SHA-256 manifest paired with --manuscript-snapshot-root.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    step2 = args.step2.resolve()
    tables = step2 / "tables"
    qa = step2 / "qa"

    national = pd.read_csv(tables / "national_controlled_contrasts.csv")
    city = pd.read_csv(tables / "city_central_controlled_contrasts.csv")
    intervals = pd.read_csv(tables / "city_controlled_contrast_intervals.csv")
    outcomes = pd.read_csv(tables / "city_structure_budget_outcomes.csv")
    gate = pd.read_csv(qa / "main_text_gate.csv")

    selected = national.loc[national["outcome"].eq(OUTCOME)].copy()
    city_support = intervals.loc[intervals["outcome"].eq(OUTCOME)].copy()
    city_support["city_interval_positive"] = city_support["ci_low"] > 0
    city_support["city_interval_negative"] = city_support["ci_high"] < 0
    support = (
        city_support.groupby(["contrast", "budget_fraction"], as_index=False)
        .agg(
            city_intervals_positive=("city_interval_positive", "sum"),
            city_intervals_negative=("city_interval_negative", "sum"),
        )
    )
    selected = selected.merge(support, on=["contrast", "budget_fraction"], how="left", validate="one_to_one")

    budget = outcomes[["city_key", "budget_fraction", "budget_km2"]].drop_duplicates()
    central = city.loc[city["outcome"].eq(OUTCOME)].merge(
        budget, on=["city_key", "budget_fraction"], validate="many_to_one"
    )
    descriptive_rows = []
    loo_rows = []
    for (contrast, budget_fraction), group in central.groupby(["contrast", "budget_fraction"]):
        values = group["difference"].to_numpy(float)
        full = float(values.mean())
        loo = np.asarray([(values.sum() - value) / (len(values) - 1) for value in values])
        influential = int(np.argmax(np.abs(loo - full)))
        descriptive_rows.append(
            {
                "contrast": contrast,
                "budget_fraction": budget_fraction,
                "equal_city_mean": full,
                "budget_area_weighted_mean": float(np.average(values, weights=group["budget_km2"])),
                "median_city": float(np.median(values)),
                "q25_city": float(np.quantile(values, 0.25)),
                "q75_city": float(np.quantile(values, 0.75)),
                "cities_positive": int((values > 0).sum()),
                "cities_negative": int((values < 0).sum()),
                "leave_one_city_out_min": float(loo.min()),
                "leave_one_city_out_max": float(loo.max()),
                "maximum_absolute_leave_one_out_shift": float(np.max(np.abs(loo - full))),
                "most_influential_city": str(group.iloc[influential]["city_display"]),
                "largest_city_absolute_share": float(np.max(np.abs(values)) / np.sum(np.abs(values))),
                "five_largest_cities_absolute_share": float(
                    np.sort(np.abs(values))[-5:].sum() / np.sum(np.abs(values))
                ),
            }
        )
        for row_index, row in group.reset_index(drop=True).iterrows():
            loo_rows.append(
                {
                    "contrast": contrast,
                    "budget_fraction": budget_fraction,
                    "omitted_city": row["city_display"],
                    "leave_one_city_out_equal_city_mean": float(loo[row_index]),
                }
            )
    descriptive = pd.DataFrame(descriptive_rows)
    selected = selected.merge(
        descriptive,
        on=["contrast", "budget_fraction"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_audit"),
    )
    selected.to_csv(tables / "structural_finding_evidence_ledger.csv", index=False, encoding="utf-8-sig")
    descriptive.to_csv(tables / "city_heterogeneity_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(loo_rows).to_csv(qa / "leave_one_city_out_all.csv", index=False, encoding="utf-8-sig")

    weighting = pd.read_csv(args.weighting)
    weighting.to_csv(qa / "shared_population_weight_10pct_sensitivity.csv", index=False, encoding="utf-8-sig")

    snapshot_requested = (
        args.manuscript_snapshot_root is not None
        or args.manuscript_snapshot_manifest is not None
    )
    if snapshot_requested and (
        args.manuscript_snapshot_root is None
        or args.manuscript_snapshot_manifest is None
    ):
        raise ValueError(
            "Both manuscript snapshot arguments are required when either is supplied."
        )
    missing: list[str] = []
    mismatched: list[str] = []
    manifest_rows = 0
    if snapshot_requested:
        manifest = pd.read_csv(args.manuscript_snapshot_manifest)
        manifest_rows = len(manifest)
        for row in manifest.itertuples(index=False):
            path = args.manuscript_snapshot_root / Path(str(row.relative_path))
            if not path.is_file():
                missing.append(str(row.relative_path))
            elif sha256(path) != str(row.sha256):
                mismatched.append(str(row.relative_path))
    snapshot = {
        "status": "checked" if snapshot_requested else "not_requested",
        "manifest_rows": int(manifest_rows),
        "missing_files": missing,
        "sha256_mismatches": mismatched,
        "snapshot_unchanged": bool(snapshot_requested and not missing and not mismatched),
    }
    (qa / "manuscript_snapshot_recheck.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    main_passes = gate.loc[gate["passes_main_text_gate"]]
    required = {
        (CAPACITY, OUTCOME),
        (GATE_ROBUST, OUTCOME),
    }
    observed = set(zip(main_passes["contrast"], main_passes["outcome"], strict=False))
    capacity_rows = descriptive.loc[descriptive["contrast"].eq(CAPACITY)]
    total_rows = descriptive.loc[descriptive["contrast"].eq(GATE_TOTAL)]
    rate_rows = descriptive.loc[descriptive["contrast"].eq(GATE_RATE)]
    weight_difference = weighting.loc[
        weighting["metric"].eq("overlap_difference_equal_minus_population")
    ].iloc[0]
    bootstrap_files = sorted((step2 / "bootstrap_city").glob("*.npz"))
    bootstrap_shapes: set[tuple[int, ...]] = set()
    bootstrap_finite = True
    tie_finite_and_bounded = True
    for path in bootstrap_files:
        with np.load(path) as archive:
            values = archive["values"]
            ties = archive["heat_tie_fraction"]
        bootstrap_shapes.add(tuple(values.shape))
        bootstrap_finite = bootstrap_finite and bool(np.isfinite(values).all())
        tie_finite_and_bounded = tie_finite_and_bounded and bool(
            np.isfinite(ties).all() and (ties >= 0).all() and (ties <= 1).all()
        )
    primary_outcomes = outcomes[
        [
            "allocation_overlap",
            "heat_context_opportunity_loss_under_social",
            "social_context_opportunity_loss_under_heat",
        ]
    ].to_numpy(float)
    checks = {
        "exactly_two_prespecified_main_text_passes": observed == required,
        "capacity_effect_positive_under_all_leave_one_city_out_runs": bool(
            (capacity_rows["leave_one_city_out_min"] > 0).all()
        ),
        "candidate_gate_total_match_positive_under_all_leave_one_city_out_runs": bool(
            (total_rows["leave_one_city_out_min"] > 0).all()
        ),
        "candidate_gate_rate_match_positive_under_all_leave_one_city_out_runs": bool(
            (rate_rows["leave_one_city_out_min"] > 0).all()
        ),
        "candidate_gate_positive_in_at_least_46_of_50_cities_at_every_budget": bool(
            (total_rows["cities_positive"] >= 46).all()
            and (rate_rows["cities_positive"] >= 46).all()
        ),
        "candidate_gate_direction_stable_under_budget_area_weighting": bool(
            (total_rows["budget_area_weighted_mean"] > 0).all()
            and (rate_rows["budget_area_weighted_mean"] > 0).all()
        ),
        "shared_population_overlap_difference_10pct_interval_crosses_zero": bool(
            weight_difference["two_level_ci_low"] <= 0 <= weight_difference["two_level_ci_high"]
        ),
        "implementation_safeguards_documented": args.implementation_note.is_file(),
        "all_50_city_bootstrap_files_present": len(bootstrap_files) == 50,
        "all_city_bootstrap_arrays_have_frozen_shape": bootstrap_shapes == {(1000, 4, 3, 3)},
        "all_city_bootstrap_outcomes_are_finite": bootstrap_finite,
        "all_heat_tie_fractions_are_finite_and_bounded": tie_finite_and_bounded,
        "all_deterministic_outcomes_are_finite_and_bounded": bool(
            np.isfinite(primary_outcomes).all()
            and (primary_outcomes >= -1e-10).all()
            and (primary_outcomes <= 1 + 1e-10).all()
        ),
        "all_national_intervals_are_ordered_and_finite": bool(
            np.isfinite(national[["ci_low", "ci_high"]].to_numpy(float)).all()
            and (national["ci_low"] <= national["ci_high"]).all()
        ),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Controlled structural decomposition, not causal explanation or policy evaluation.",
        "checks": checks,
        "main_text_eligible_findings": [
            {
                "structure": "observed canopy-deficit capacity intensity",
                "outcome": OUTCOME,
                "qualification": "national structural sensitivity; city directions are heterogeneous",
            },
            {
                "structure": "positive canopy-deficit candidate gate",
                "outcome": OUTCOME,
                "qualification": "robust to total-matched and rate-matched null capacities",
            },
        ],
        "not_main_text_headlines": [
            "Overlap contrasts, because the total-matched 5% change misses the 0.02 threshold and 20% heat allocations are non-unique in some cities.",
            "Social-context opportunity loss under heat allocation, because some 20% heat allocations enter H=0 ties.",
            "The 10% shared-population-multiplier contrast, because its paired interval crosses zero and it was not extended confirmatorily across budgets.",
        ],
        "optional_manuscript_snapshot": snapshot,
    }
    (qa / "step02_final_technical_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise AssertionError([key for key, value in checks.items() if not value])

    shutil.copy2(
        args.implementation_note,
        step2 / "methods" / args.implementation_note.name,
    )
    shutil.copy2(Path(__file__), step2 / "methods" / Path(__file__).name)
    manifest_rows = []
    for path in sorted(item for item in step2.rglob("*") if item.is_file()):
        if path.name == "output_sha256.csv":
            continue
        manifest_rows.append(
            {
                "relative_path": path.relative_to(step2).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        step2 / "manifests" / "output_sha256.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
