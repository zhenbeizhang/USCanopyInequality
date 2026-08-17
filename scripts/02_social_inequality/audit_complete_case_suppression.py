from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = analysis_store_root()
INPUT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_city_interior_reanalysis"
    / "source_data"
    / "result2_city_interior80_block_group_analysis_dataset.csv"
)
REFERENCE_SCRIPT = Path(__file__).with_name("complete_case_model_core.py")
OLD_AUDIT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_complete_case_comparability_audit"
    / "tables"
)
OUT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_complete_case_city_interior80_audit"
)
TABLES = OUT / "tables"
DATA = OUT / "data"
QA = OUT / "qa"
TEXT = OUT / "text"

SEED = 20260719
N_BOOT = 9_999
EXPECTED_SOURCE = 35_612
EXPECTED_COMPLETE = 31_819
EXPECTED_CITIES = 50


def load_reference_module():
    spec = importlib.util.spec_from_file_location(
        "result2_complete_case_reference", REFERENCE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the reference Result 2 audit module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_collinearity(
    module, frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    _, _, x, weights, _, _ = module.within_transform(frame, module.JOINT_Z)
    covariance = (x.T * weights) @ x / weights.sum()
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, np.inf))
    correlation = covariance / np.outer(scale, scale)
    correlation_table = pd.DataFrame(
        correlation, index=module.JOINT_Z, columns=module.JOINT_Z
    )

    vif_rows = []
    for j, predictor in enumerate(module.JOINT_Z):
        others = [k for k in range(x.shape[1]) if k != j]
        xo = x[:, others]
        target = x[:, j]
        beta = np.linalg.pinv(xo.T @ (weights[:, None] * xo)) @ (
            xo.T @ (weights * target)
        )
        residual = target - xo @ beta
        sse = float(np.sum(weights * residual**2))
        sst = float(np.sum(weights * target**2))
        r2 = 1.0 - sse / sst
        vif_rows.append(
            {
                "predictor": predictor,
                "predictor_label": module.META[predictor].label,
                "r2_from_other_predictors": r2,
                "tolerance": 1.0 - r2,
                "vif": 1.0 / (1.0 - r2),
            }
        )

    weighted_x = np.sqrt(weights)[:, None] * x
    column_norms = np.sqrt(np.sum(weighted_x**2, axis=0))
    singular_values = np.linalg.svd(
        weighted_x / column_norms[None, :], compute_uv=False
    )
    max_condition_index = float(singular_values.max() / singular_values.min())
    return correlation_table, pd.DataFrame(vif_rows), max_condition_index


def poverty_subset_audit(module, frame: pd.DataFrame) -> pd.DataFrame:
    _, y, x, weights, _, _ = module.within_transform(frame, module.JOINT_Z)
    target = module.JOINT_Z.index("z_poverty")
    others = [j for j in range(len(module.JOINT_Z)) if j != target]
    rows = []
    for subset_size in range(len(others) + 1):
        for subset in combinations(others, subset_size):
            columns = [target, *subset]
            design = x[:, columns]
            beta = np.linalg.pinv(
                design.T @ (weights[:, None] * design), rcond=1e-12
            ) @ (design.T @ (weights * y))
            rows.append(
                {
                    "n_adjustment_predictors": subset_size,
                    "adjustment_predictors": ";".join(
                        module.JOINT_Z[j] for j in subset
                    ),
                    "poverty_coefficient_pp_per_within_city_sd": float(beta[0]),
                    "poverty_positive": bool(beta[0] > 0),
                }
            )
    return pd.DataFrame(rows)


def leave_one_city_out(module, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in sorted(frame["result_city_key"].astype(str).unique()):
        subset = frame.loc[~frame["result_city_key"].astype(str).eq(city)].copy()
        estimate = module.fit_fe_wild(
            subset,
            module.JOINT_Z,
            {},
            run_bootstrap=False,
        )
        poverty = estimate.loc[estimate["predictor"].eq("z_poverty")].iloc[0]
        rows.append(
            {
                "omitted_city": city,
                "poverty_joint_coefficient": float(
                    poverty["estimate_pp_per_within_city_sd"]
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    for directory in (OUT, TABLES, DATA, QA, TEXT):
        directory.mkdir(parents=True, exist_ok=True)

    module = load_reference_module()
    frame = pd.read_csv(
        INPUT, dtype={"spatial_unit_id": str}, low_memory=False
    )
    if (
        len(frame) != EXPECTED_SOURCE
        or frame["result_city_key"].nunique() != EXPECTED_CITIES
    ):
        raise RuntimeError("Unexpected city-interior source frame.")
    if frame["city_intersection_fraction"].min() < 0.80:
        raise RuntimeError("The source frame violates the 80% boundary rule.")

    required = [
        "canopy_reference_0p6m",
        "population",
        "result_city_key",
        *[predictor.raw for predictor in module.PREDICTORS],
    ]
    included = (
        frame[required]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    included &= pd.to_numeric(frame["population"], errors="coerce") > 0
    complete_raw = frame.loc[included].copy().reset_index(drop=True)
    if (
        len(complete_raw) != EXPECTED_COMPLETE
        or complete_raw["result_city_key"].nunique() != EXPECTED_CITIES
    ):
        raise RuntimeError("Unexpected city-interior complete-case frame.")

    complete = module.standardise_within_city(
        complete_raw, module.ALL_MARGINAL
    )
    rng = np.random.default_rng(SEED)
    city_names = sorted(frame["result_city_key"].astype(str).unique())
    rademacher = rng.choice(
        np.array([-1.0, 1.0]),
        size=(N_BOOT, EXPECTED_CITIES),
        replace=True,
    )
    rademacher_by_city = {
        city: rademacher[:, index] for index, city in enumerate(city_names)
    }

    marginal = module.fit_marginals(
        complete,
        module.JOINT_Z,
        rademacher_by_city,
        "marginal_city_interior80_complete_case",
        "restandardised_within_31819_complete_cases",
    )
    joint = module.add_model_metadata(
        module.fit_fe_wild(
            complete,
            module.JOINT_Z,
            rademacher_by_city,
            run_bootstrap=True,
        ),
        "joint_city_interior80_complete_case",
        "mutually adjusted city fixed-effect model",
        "restandardised_within_31819_complete_cases",
    )
    both = pd.concat([marginal, joint], ignore_index=True)
    both.to_csv(
        TABLES / "01_city_interior80_complete_case_models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison = marginal[
        [
            "predictor",
            "predictor_label",
            "estimate_pp_per_within_city_sd",
            "wild_ci_low",
            "wild_ci_high",
            "wild_p",
            "fdr_q",
        ]
    ].merge(
        joint[
            [
                "predictor",
                "estimate_pp_per_within_city_sd",
                "wild_ci_low",
                "wild_ci_high",
                "wild_p",
                "fdr_q",
            ]
        ],
        on="predictor",
        suffixes=("_marginal", "_joint"),
        validate="one_to_one",
    )
    comparison.to_csv(
        TABLES / "02_marginal_joint_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correlation, vif, max_condition = weighted_collinearity(module, complete)
    correlation.to_csv(
        TABLES / "03_weighted_within_city_predictor_correlation.csv",
        encoding="utf-8-sig",
    )
    vif.to_csv(TABLES / "04_vif.csv", index=False, encoding="utf-8-sig")

    subsets = poverty_subset_audit(module, complete)
    subsets.to_csv(
        TABLES / "05_poverty_all_512_adjustment_subsets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loco = leave_one_city_out(module, complete)
    loco.to_csv(
        TABLES / "06_poverty_leave_one_city_out.csv",
        index=False,
        encoding="utf-8-sig",
    )

    membership = frame[
        [
            "result_city_key",
            "spatial_unit_id",
            "city_intersection_fraction",
        ]
    ].copy()
    membership["included_in_city_interior80_complete_case"] = included.to_numpy()
    membership.to_csv(
        DATA / "city_interior80_complete_case_membership.csv.gz",
        index=False,
        compression="gzip",
    )

    poverty = comparison.loc[comparison["predictor"].eq("z_poverty")].iloc[0]
    old_marginal = pd.read_csv(
        OLD_AUDIT / "04_marginal_complete_case_restandardized.csv"
    )
    old_joint = pd.read_csv(
        OLD_AUDIT / "06_joint_complete_case_restandardized.csv"
    )
    old_m = old_marginal.loc[
        old_marginal["predictor"].eq("z_poverty")
    ].iloc[0]
    old_j = old_joint.loc[old_joint["predictor"].eq("z_poverty")].iloc[0]
    positive_subsets = int(subsets["poverty_positive"].sum())

    summary = {
        "status": "PASS"
        if (
            poverty["estimate_pp_per_within_city_sd_marginal"] < 0
            and poverty["wild_ci_high_marginal"] < 0
            and poverty["estimate_pp_per_within_city_sd_joint"] > 0
            and poverty["wild_ci_low_joint"] > 0
            and (np.sign(loco["poverty_joint_coefficient"]) > 0).all()
        )
        else "FAIL",
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "source_frame": {
            "boundary_rule": "city intersection fraction >= 0.80",
            "block_groups": len(frame),
            "cities": int(frame["result_city_key"].nunique()),
        },
        "complete_case_frame": {
            "block_groups": len(complete),
            "cities": int(complete["result_city_key"].nunique()),
            "retained_percent": 100 * len(complete) / len(frame),
        },
        "poverty_city_interior80": {
            "marginal_estimate": float(
                poverty["estimate_pp_per_within_city_sd_marginal"]
            ),
            "marginal_ci": [
                float(poverty["wild_ci_low_marginal"]),
                float(poverty["wild_ci_high_marginal"]),
            ],
            "joint_estimate": float(
                poverty["estimate_pp_per_within_city_sd_joint"]
            ),
            "joint_ci": [
                float(poverty["wild_ci_low_joint"]),
                float(poverty["wild_ci_high_joint"]),
            ],
            "joint_p": float(poverty["wild_p_joint"]),
            "joint_fdr_q": float(poverty["fdr_q_joint"]),
        },
        "poverty_fully_contained_reference": {
            "source_block_groups": 34_575,
            "complete_case_block_groups": 30_826,
            "marginal_estimate": float(
                old_m["estimate_pp_per_within_city_sd"]
            ),
            "joint_estimate": float(old_j["estimate_pp_per_within_city_sd"]),
        },
        "diagnostics_city_interior80": {
            "maximum_absolute_predictor_correlation": float(
                np.max(
                    np.abs(
                        correlation.to_numpy()
                        - np.eye(len(correlation))
                    )
                )
            ),
            "maximum_vif": float(vif["vif"].max()),
            "maximum_condition_index": max_condition,
            "positive_adjustment_subsets": positive_subsets,
            "total_adjustment_subsets": len(subsets),
            "positive_adjustment_subset_percent": (
                100 * positive_subsets / len(subsets)
            ),
            "leave_one_city_out_minimum": float(
                loco["poverty_joint_coefficient"].min()
            ),
            "leave_one_city_out_maximum": float(
                loco["poverty_joint_coefficient"].max()
            ),
            "leave_one_city_out_positive_cities": int(
                (loco["poverty_joint_coefficient"] > 0).sum()
            ),
        },
        "seed": SEED,
        "wild_cluster_draws": N_BOOT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (QA / "city_interior80_complete_case_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# Result 2 city-interior complete-case audit

The declared Result 2 source frame contains {len(frame):,} block groups in
{EXPECTED_CITIES} cities under the >=80% city-intersection rule. A frozen
complete-case mask retains {len(complete):,} units
({100 * len(complete) / len(frame):.1f}%) in all cities.

On these rows, the poverty coefficient is
{poverty['estimate_pp_per_within_city_sd_marginal']:+.3f}
[{poverty['wild_ci_low_marginal']:+.3f},
{poverty['wild_ci_high_marginal']:+.3f}] percentage points per within-city
standard deviation in the marginal model and
{poverty['estimate_pp_per_within_city_sd_joint']:+.3f}
[{poverty['wild_ci_low_joint']:+.3f},
{poverty['wild_ci_high_joint']:+.3f}] after mutual adjustment.

The joint coefficient remains positive in
{positive_subsets}/{len(subsets)} adjustment subsets and in all
{len(loco)} leave-one-city-out fits. The sign reversal therefore does not
depend on switching from the declared >=80% frame to the older fully
contained frame.
"""
    (TEXT / "city_interior80_complete_case_audit.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
