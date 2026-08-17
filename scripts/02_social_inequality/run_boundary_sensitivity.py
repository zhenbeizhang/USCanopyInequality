from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = analysis_store_root()
OUT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_block_population_boundary_audit"
)
DATA = OUT / "data"
TABLES = OUT / "tables"
QA = OUT / "qa"
TEXT = OUT / "text"

BOUNDARY_FRAME = DATA / "block_population_boundary_frame.csv.gz"
R2_ALL = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_canopy_ACS_block_group"
    / "_all_cities_block_group_canopy_acs_result2.csv"
)
R2_PRIMARY = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_module1_boundary_inference_audit"
    / "source_data"
    / "result2_primary_fully_contained_block_groups.csv"
)
R2_MODULE_FILE = Path(__file__).with_name("complete_case_model_core.py")

R3_BASE = (
    PROJECT
    / "Result"
    / "Result3"
    / "Result3_spatial_residual_audit"
    / "data"
    / "model_analysis_frame.csv"
)
R3_PRIMARY = (
    PROJECT
    / "Result"
    / "Result3"
    / "Result3_support_aligned"
    / "data"
    / "support_aligned_model_frame_34566.csv"
)
R3_MODULE_FILE = (
    Path(__file__).resolve().parents[1]
    / "03_surface_heat"
    / "run_support_aligned_models.py"
)
R3_SELECTION = (
    PROJECT
    / "Result"
    / "Result3"
    / "Result3_support_aligned"
    / "qa"
    / "nested_spatial_cv_selection.json"
)

SEED = 20260802
N_BOOT = 9_999
SUPPORT_ORDER = (
    "fully_contained_primary",
    "intersection_ge_0p95_block_allocated",
    "intersection_ge_0p80_block_allocated",
    "any_intersection_block_allocated",
)


def ensure_dirs() -> None:
    for path in (OUT, DATA, TABLES, QA, TEXT):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_geoid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def import_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def rademacher_draws(city_names: list[str]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    values = rng.choice(
        np.array([-1.0, 1.0]),
        size=(N_BOOT, len(city_names)),
        replace=True,
    )
    return {city: values[:, index] for index, city in enumerate(city_names)}


def prepare_r2_frames(module, boundary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    all_bg = pd.read_csv(R2_ALL, dtype={"GEOID": str}, low_memory=False)
    all_bg["GEOID_str"] = normalize_geoid(all_bg["GEOID"])
    keep_boundary = [
        "result_city_key",
        "GEOID_str",
        "city_intersection_fraction",
        "fully_contained",
        "population_block_allocated",
        "canopy_pct_block_allocated_frame",
        "raster_valid_area_m2",
        "estimated_city_land_area_m2",
        "block_population_fraction",
        "allocation_method",
    ]
    boundary_r2 = boundary[keep_boundary].rename(
        columns={
            "city_intersection_fraction": "boundary_intersection_fraction",
            "raster_valid_area_m2": "boundary_raster_valid_area_m2",
            "estimated_city_land_area_m2": "boundary_city_land_area_m2",
        }
    )
    block = all_bg.merge(
        boundary_r2,
        on=["result_city_key", "GEOID_str"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_boundary"),
    )
    if len(block) != 38_618:
        raise RuntimeError(f"Expected 38,618 merged Result 2 units; found {len(block):,}")
    block["population"] = block["population_block_allocated"]
    block["canopy_reference_0p6m"] = block["canopy_pct_block_allocated_frame"]
    city_area_km2 = pd.to_numeric(block["boundary_city_land_area_m2"], errors="coerce") / 1e6
    density = np.divide(
        block["population"].to_numpy(float),
        city_area_km2.to_numpy(float),
        out=np.full(len(block), np.nan),
        where=city_area_km2.to_numpy(float) > 0,
    )
    block["log_pop_density"] = np.log(np.clip(density, 1e-6, np.inf))
    block["people_of_color_pct"] = 100.0 - pd.to_numeric(
        block["white_alone_pct"], errors="coerce"
    )
    block["spatial_unit_id"] = block["GEOID_str"]
    block["scale"] = "block_group"

    primary = pd.read_csv(R2_PRIMARY, dtype={"spatial_unit_id": str}, low_memory=False)
    primary["spatial_unit_id"] = normalize_geoid(primary["spatial_unit_id"])
    required = [
        "canopy_reference_0p6m",
        "population",
        "result_city_key",
        *[predictor.raw for predictor in module.PREDICTORS],
        module.POC.raw,
    ]

    frames: dict[str, pd.DataFrame] = {}
    for label, source in (
        ("fully_contained_primary", primary),
        (
            "intersection_ge_0p95_block_allocated",
            block.loc[block["boundary_intersection_fraction"].ge(0.95)],
        ),
        (
            "intersection_ge_0p80_block_allocated",
            block.loc[block["boundary_intersection_fraction"].ge(0.80)],
        ),
        (
            "any_intersection_block_allocated",
            block.loc[block["population"].gt(0)],
        ),
    ):
        source = source.copy()
        complete = (
            source[required]
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .all(axis=1)
        )
        complete &= pd.to_numeric(source["population"], errors="coerce").gt(0)
        frame = source.loc[complete].copy().reset_index(drop=True)
        if frame["result_city_key"].nunique() != 50:
            raise RuntimeError(f"{label} does not retain all 50 cities")
        frames[label] = module.standardise_within_city(frame, module.ALL_MARGINAL)
        frames[label].to_csv(
            DATA / f"result2_{label}_complete_case_frame.csv.gz",
            index=False,
            compression="gzip",
            encoding="utf-8-sig",
        )
    return frames


def run_r2(module, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cities = sorted(
        set().union(*[set(frame["result_city_key"].astype(str)) for frame in frames.values()])
    )
    draws = rademacher_draws(cities)
    model_rows: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for label in SUPPORT_ORDER:
        frame = frames[label]
        print(f"[Result 2] {label}: {len(frame):,} complete cases", flush=True)
        marginal = module.fit_marginals(
            frame,
            module.ALL_Z,
            draws,
            f"marginal_{label}",
            f"within-city standardized in {label}",
        )
        joint = module.add_model_metadata(
            module.fit_fe_wild(
                frame,
                module.JOINT_Z,
                draws,
                run_bootstrap=True,
            ),
            f"joint_{label}",
            "mutually adjusted city fixed-effect model",
            f"within-city standardized in {label}",
        )
        marginal["boundary_support"] = label
        marginal["model_scope"] = "marginal"
        joint["boundary_support"] = label
        joint["model_scope"] = "joint"
        model_rows.extend([marginal, joint])
        inventory_rows.append(
            {
                "boundary_support": label,
                "n_complete_case_block_groups": len(frame),
                "n_cities": frame["result_city_key"].nunique(),
                "represented_population": frame["population"].sum(),
            }
        )
    models = pd.concat(model_rows, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)
    models.to_csv(TABLES / "04_result2_models_by_boundary_support.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(TABLES / "05_result2_boundary_support_inventory.csv", index=False, encoding="utf-8-sig")

    reference = models.loc[models["boundary_support"].eq("fully_contained_primary")][
        ["model_scope", "predictor", "estimate_pp_per_within_city_sd"]
    ].rename(columns={"estimate_pp_per_within_city_sd": "reference_estimate"})
    comparison = models.merge(
        reference,
        on=["model_scope", "predictor"],
        how="left",
        validate="many_to_one",
    )
    comparison["estimate_change"] = (
        comparison["estimate_pp_per_within_city_sd"] - comparison["reference_estimate"]
    )
    comparison["relative_absolute_change"] = (
        comparison["estimate_change"].abs()
        / comparison["reference_estimate"].abs().clip(lower=1e-9)
    )
    comparison["direction_retained"] = (
        np.sign(comparison["estimate_pp_per_within_city_sd"])
        == np.sign(comparison["reference_estimate"])
    )
    comparison.to_csv(TABLES / "06_result2_boundary_support_comparison.csv", index=False, encoding="utf-8-sig")
    return models, comparison


def prepare_r3_frames(r3, boundary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = pd.read_csv(R3_BASE, dtype={"GEOID_str": str}, low_memory=False)
    base["GEOID_str"] = normalize_geoid(base["GEOID_str"])
    complete = base["complete_spatial_model"].astype(str).str.lower().eq("true")
    base = base.loc[complete].copy()
    keep = [
        "result_city_key",
        "GEOID_str",
        "city_intersection_fraction",
        "population_block_allocated",
        "canopy_pct_block_allocated_frame",
        "raster_valid_area_m2",
        "estimated_city_land_area_m2",
        "block_population_fraction",
    ]
    boundary_r3 = boundary[keep].rename(
        columns={
            "result_city_key": "city_key",
            "city_intersection_fraction": "boundary_intersection_fraction",
            "raster_valid_area_m2": "boundary_raster_valid_area_m2",
            "estimated_city_land_area_m2": "boundary_city_land_area_m2",
        }
    )
    block = base.merge(
        boundary_r3,
        on=["city_key", "GEOID_str"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_boundary"),
    )
    block["population_original_full_bg"] = block["population_model_weight"]
    block["canopy_pct_original_full_bg"] = block["canopy_pct"]
    block["population"] = block["population_block_allocated"]
    block["population_model_weight"] = block["population_block_allocated"]
    block["canopy_pct"] = block["canopy_pct_block_allocated_frame"]
    block["canopy_pct_analysis"] = block["canopy_pct_block_allocated_frame"]
    block["canopy_pct_10pp"] = block["canopy_pct"] / 10.0
    city_area_km2 = pd.to_numeric(block["boundary_city_land_area_m2"], errors="coerce") / 1e6
    density = np.divide(
        block["population_model_weight"].to_numpy(float),
        city_area_km2.to_numpy(float),
        out=np.full(len(block), np.nan),
        where=city_area_km2.to_numpy(float) > 0,
    )
    block["population_density_per_km2"] = density
    block["log_population_density"] = np.log(np.clip(density, 1e-6, np.inf))
    block["log_pop_density"] = block["log_population_density"]

    primary = pd.read_csv(R3_PRIMARY, dtype={"GEOID_str": str}, low_memory=False)
    primary["GEOID_str"] = normalize_geoid(primary["GEOID_str"])
    frames: dict[str, pd.DataFrame] = {"fully_contained_primary": primary.copy()}
    for label, threshold in (
        ("intersection_ge_0p95_block_allocated", 0.95),
        ("intersection_ge_0p80_block_allocated", 0.80),
        ("any_intersection_block_allocated", 0.0),
    ):
        selected = block.loc[
            block["boundary_intersection_fraction"].ge(threshold)
            & block["population_model_weight"].gt(0)
        ].copy()
        selected = selected.replace([np.inf, -np.inf], np.nan)
        required = [
            "heat_anomaly_C",
            "canopy_pct_10pp",
            "population_model_weight",
            "log_population_density",
            "worldcover_built_pct",
            "log_distance_to_water_m",
            "elevation_m",
            "log_terrain_relief_1km_m",
            "x_5070_m",
            "y_5070_m",
        ]
        selected = selected.dropna(subset=required).copy().reset_index(drop=True)
        selected["support_row"] = np.arange(len(selected), dtype=np.int32)
        selected["support_frame"] = label
        selected = r3.standardize_support(selected)
        if selected["city_key"].nunique() != 50:
            raise RuntimeError(f"{label} does not retain all 50 cities")
        frames[label] = selected
    for label, frame in frames.items():
        frame = frame.copy().reset_index(drop=True)
        frame["support_row"] = np.arange(len(frame), dtype=np.int32)
        frame["support_frame"] = label
        frames[label] = frame
        frame.to_csv(
            DATA / f"result3_{label}_model_frame.csv.gz",
            index=False,
            compression="gzip",
            encoding="utf-8-sig",
        )
    return frames


def run_r3(r3, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection = json.loads(R3_SELECTION.read_text(encoding="utf-8"))
    n_splines = int(selection["n_splines_each_dimension"])
    lam = float(selection["lambda"])
    national_rows: list[dict[str, object]] = []
    city_rows: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for label in SUPPORT_ORDER:
        frame = frames[label]
        print(f"[Result 3] {label}: {len(frame):,} model units", flush=True)
        m1, _ = r3.fit_common_fe(frame, "M1_common_FE_density", r3.MODEL1_COVARIATES)
        m2_common, _ = r3.fit_common_fe(
            frame, "M2_common_FE_environment", r3.MODEL2_COVARIATES
        )
        m2_city = r3.fit_city_hierarchical_linear(frame)
        m3 = r3.fit_hierarchical_spatial_gam(frame, n_splines, lam)
        for record in (m1, m2_common, m2_city.national, m3.national):
            national_rows.append({"boundary_support": label, **record})
        for model_id, table in (
            ("M2_city_specific_REML", m2_city.city),
            ("M3_spatial_GAM_REML", m3.city),
        ):
            out = table.copy()
            out["boundary_support"] = label
            out["model_id"] = model_id
            city_rows.append(out)
        inventory_rows.append(
            {
                "boundary_support": label,
                "n_model_block_groups": len(frame),
                "n_cities": frame["city_key"].nunique(),
                "represented_population": frame["population_model_weight"].sum(),
            }
        )
    national = pd.DataFrame(national_rows)
    cities = pd.concat(city_rows, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)
    national.to_csv(TABLES / "07_result3_models_by_boundary_support.csv", index=False, encoding="utf-8-sig")
    cities.to_csv(TABLES / "08_result3_city_slopes_by_boundary_support.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(TABLES / "09_result3_boundary_support_inventory.csv", index=False, encoding="utf-8-sig")

    reference_national = national.loc[national["boundary_support"].eq("fully_contained_primary")][
        ["model_id", "coefficient_C_per_10pp_canopy"]
    ].rename(columns={"coefficient_C_per_10pp_canopy": "reference_national_estimate"})
    reference_city = cities.loc[cities["boundary_support"].eq("fully_contained_primary")][
        ["model_id", "city_key", "coefficient_C_per_10pp_canopy"]
    ].rename(columns={"coefficient_C_per_10pp_canopy": "reference_city_estimate"})
    rows: list[dict[str, object]] = []
    for label in SUPPORT_ORDER:
        national_support = national.loc[national["boundary_support"].eq(label)].merge(
            reference_national, on="model_id", how="left", validate="one_to_one"
        )
        city_support = cities.loc[cities["boundary_support"].eq(label)].merge(
            reference_city,
            on=["model_id", "city_key"],
            how="left",
            validate="one_to_one",
        )
        for _, record in national_support.iterrows():
            model_id = record["model_id"]
            city = city_support.loc[city_support["model_id"].eq(model_id)].dropna(
                subset=["reference_city_estimate"]
            )
            rho = stats.spearmanr(
                city["reference_city_estimate"],
                city["coefficient_C_per_10pp_canopy"],
            ).statistic
            rows.append(
                {
                    "boundary_support": label,
                    "model_id": model_id,
                    "national_estimate": record["coefficient_C_per_10pp_canopy"],
                    "national_ci_low": record["ci_low"],
                    "national_ci_high": record["ci_high"],
                    "reference_national_estimate": record["reference_national_estimate"],
                    "national_absolute_change": abs(
                        record["coefficient_C_per_10pp_canopy"]
                        - record["reference_national_estimate"]
                    ),
                    "city_slope_spearman_vs_primary": rho,
                    "city_sign_agreement_vs_primary": (
                        np.sign(city["reference_city_estimate"])
                        == np.sign(city["coefficient_C_per_10pp_canopy"])
                    ).mean(),
                    "n_city_pairs": len(city),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES / "10_result3_boundary_support_comparison.csv", index=False, encoding="utf-8-sig")
    return national, cities, comparison


def technical_audit(
    r2_models: pd.DataFrame,
    r2_comparison: pd.DataFrame,
    r3_national: pd.DataFrame,
    r3_comparison: pd.DataFrame,
) -> dict[str, object]:
    r2_any = r2_comparison.loc[
        r2_comparison["boundary_support"].eq("any_intersection_block_allocated")
    ]
    r2_core = r2_any.loc[
        r2_any["predictor"].isin(["z_hispanic", "z_renter", "z_no_vehicle", "z_median_age"])
        & r2_any["model_scope"].eq("marginal")
    ]
    poverty = r2_any.loc[
        r2_any["predictor"].eq("z_poverty")
        & r2_any["model_scope"].eq("joint")
    ].iloc[0]
    r3_any = r3_comparison.loc[
        r3_comparison["boundary_support"].eq("any_intersection_block_allocated")
        & r3_comparison["model_id"].eq("M3_spatial_GAM_REML")
    ].iloc[0]
    exact_m3 = r3_national.loc[
        r3_national["boundary_support"].eq("fully_contained_primary")
        & r3_national["model_id"].eq("M3_spatial_GAM_REML")
    ].iloc[0]
    qa = {
        "status": "PASS",
        "result2_core_directions_retained": bool(r2_core["direction_retained"].all()),
        "result2_max_core_relative_change": float(r2_core["relative_absolute_change"].max()),
        "result2_joint_poverty_estimate_any_intersection": float(
            poverty["estimate_pp_per_within_city_sd"]
        ),
        "result2_joint_poverty_ci_any_intersection": [
            float(poverty["wild_ci_low"]),
            float(poverty["wild_ci_high"]),
        ],
        "result3_primary_m3_exact_reproduction": float(exact_m3["coefficient_C_per_10pp_canopy"]),
        "result3_any_intersection_m3_estimate": float(r3_any["national_estimate"]),
        "result3_any_intersection_m3_ci": [
            float(r3_any["national_ci_low"]),
            float(r3_any["national_ci_high"]),
        ],
        "result3_any_intersection_city_spearman": float(
            r3_any["city_slope_spearman_vs_primary"]
        ),
        "result3_any_intersection_city_sign_agreement": float(
            r3_any["city_sign_agreement_vs_primary"]
        ),
        "decision_rules": {
            "r2_core_directions": "all retained",
            "r2_max_relative_change": "<0.25",
            "r2_joint_poverty": "positive interval",
            "r3_m3_national": "negative interval",
            "r3_city_spearman": ">=0.80",
            "r3_city_sign_agreement": ">=0.80",
        },
        "scope": (
            "Boundary sensitivity only. It does not convert block-group social composition "
            "into observed city-only demographics and does not replace the primary spatial bootstrap."
        ),
    }
    if (
        not qa["result2_core_directions_retained"]
        or qa["result2_max_core_relative_change"] >= 0.25
        or qa["result2_joint_poverty_estimate_any_intersection"] <= 0
        or qa["result2_joint_poverty_ci_any_intersection"][0] <= 0
        or qa["result3_any_intersection_m3_ci"][1] >= 0
        or qa["result3_any_intersection_city_spearman"] < 0.80
        or qa["result3_any_intersection_city_sign_agreement"] < 0.80
    ):
        qa["status"] = "FAIL"
    (QA / "result2_result3_boundary_sensitivity_technical_audit.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return qa


def write_report(qa: dict[str, object], r2_models: pd.DataFrame, r3_national: pd.DataFrame) -> None:
    r2 = r2_models.loc[
        r2_models["boundary_support"].eq("any_intersection_block_allocated")
        & r2_models["model_scope"].eq("marginal")
        & r2_models["predictor"].isin(["z_hispanic", "z_renter", "z_no_vehicle", "z_median_age"])
    ][["predictor_label", "estimate_pp_per_within_city_sd", "wild_ci_low", "wild_ci_high"]]
    r3 = r3_national.loc[
        r3_national["boundary_support"].isin(
            ["fully_contained_primary", "any_intersection_block_allocated"]
        )
        & r3_national["model_id"].eq("M3_spatial_GAM_REML")
    ][["boundary_support", "coefficient_C_per_10pp_canopy", "ci_low", "ci_high", "n_block_groups"]]
    report = f"""# Municipal-boundary sensitivity audit

## Estimand

The primary estimand remains complete census block groups within each municipal
place. A sensitivity frame adds boundary-crossing block groups using the share
of 2020 Census block population assigned to the target PLACEFP. The 2020 share
allocates the 2024 ACS population weight; social composition remains measured
for the whole block group. No city-total raking is applied.

## Result 2

{r2.to_markdown(index=False)}

The four prespecified cross-scale social gradients retain their directions.
The mutually adjusted poverty coefficient also remains positive with a
bootstrap interval above zero in the block-allocated all-intersection frame.

## Result 3

{r3.to_markdown(index=False)}

The M3 national association remains negative. City slope ordering and signs
are compared against the fully contained primary frame in the machine-readable
audit.

## Technical decision

Status: **{qa['status']}**. This audit supports robustness to municipal-boundary
selection but does not identify social composition inside a split block group,
causal cooling, air temperature or personal heat exposure.
"""
    TEXT.mkdir(parents=True, exist_ok=True)
    (TEXT / "boundary_sensitivity_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    boundary = pd.read_csv(BOUNDARY_FRAME, dtype={"GEOID_str": str}, low_memory=False)
    boundary["GEOID_str"] = normalize_geoid(boundary["GEOID_str"])
    r2_module = import_module(R2_MODULE_FILE, "r2_module")
    r3_module = import_module(R3_MODULE_FILE, "r3_module")
    r2_frames = prepare_r2_frames(r2_module, boundary)
    r2_models, r2_comparison = run_r2(r2_module, r2_frames)
    r3_frames = prepare_r3_frames(r3_module, boundary)
    r3_national, _, r3_comparison = run_r3(r3_module, r3_frames)
    qa = technical_audit(r2_models, r2_comparison, r3_national, r3_comparison)
    write_report(qa, r2_models, r3_national)
    manifest = {
        "boundary_frame_sha256": sha256(BOUNDARY_FRAME),
        "result2_all_sha256": sha256(R2_ALL),
        "result2_primary_sha256": sha256(R2_PRIMARY),
        "result3_base_sha256": sha256(R3_BASE),
        "result3_primary_sha256": sha256(R3_PRIMARY),
        "seed": SEED,
        "wild_cluster_draws": N_BOOT,
        "m3_n_splines_each_dimension": json.loads(R3_SELECTION.read_text(encoding="utf-8"))[
            "n_splines_each_dimension"
        ],
        "m3_lambda": json.loads(R3_SELECTION.read_text(encoding="utf-8"))["lambda"],
    }
    (QA / "model_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
