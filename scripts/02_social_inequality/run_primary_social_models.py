"""Module 1 audit for Result 2: boundary definition, weights and inference.

The primary analysis uses census units that are effectively fully contained in
the analytical city boundary (intersection fraction >= 0.9995). The tolerance
allows 0.05 percentage points of raster/area numerical discrepancy. The
previous 80% rule and a 95% rule are retained as sensitivity samples.

Only predictors are standardised within city. The canopy outcome remains in
percentage points. Each social predictor is fitted in a separate city fixed-
effect model; coefficients therefore describe overlapping bivariate social
gradients and are not mutually adjusted mechanisms.
"""

from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = analysis_store_root()
INPUT_ROOT = PROJECT / "Result" / "Result2" / "Result2_city_interior_reanalysis" / "source_data"
BG_INPUT = INPUT_ROOT / "result2_city_interior80_block_group_analysis_dataset.csv"
TRACT_INPUT = INPUT_ROOT / "result2_city_interior80_tract_analysis_dataset.csv"
CITY_LABELS = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_NatureCities_final"
    / "tables"
    / "result2_naturecities_final_city_equity_matrix.csv"
)
OUT_ROOT = PROJECT / "Result" / "Result2" / "Result2_module1_boundary_inference_audit"
SOURCE_DIR = OUT_ROOT / "source_data"
TABLE_DIR = OUT_ROOT / "tables"
QA_DIR = OUT_ROOT / "qa"
TEXT_DIR = OUT_ROOT / "text"
SCRIPT_DIR = OUT_ROOT / "scripts"
for directory in (SOURCE_DIR, TABLE_DIR, QA_DIR, TEXT_DIR, SCRIPT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


SEED = 20260719
N_BOOT = 9999
T_975_DF49 = 2.009575
FULLY_CONTAINED_TOLERANCE = 0.9995
SAMPLES = {
    "interior_80": 0.80,
    "interior_95": 0.95,
    "fully_contained": FULLY_CONTAINED_TOLERANCE,
}
WEIGHT_SCHEMES = ("population", "equal_unit", "city_balanced_population")
PRIMARY_SAMPLE = "fully_contained"
PRIMARY_WEIGHT = "population"


PREDICTORS = [
    ("z_low_income", "Low income", "Economic", "median_household_income_usd", -1.0),
    ("z_poverty", "Poverty", "Economic", "poverty_rate_pct", 1.0),
    ("z_median_age", "Older age", "Age", "median_age", 1.0),
    ("z_black", "Black", "Race/ethnicity", "black_alone_pct", 1.0),
    ("z_hispanic", "Hispanic/Latino", "Race/ethnicity", "hispanic_or_latino_pct", 1.0),
    ("z_asian", "Asian", "Race/ethnicity", "asian_alone_pct", 1.0),
    ("z_poc", "People of colour", "Race/ethnicity", "people_of_color_pct", 1.0),
    ("z_renter", "Renters", "Housing/mobility", "renter_occupied_pct", 1.0),
    ("z_no_vehicle", "No vehicle", "Housing/mobility", "no_vehicle_households_pct", 1.0),
    ("z_public_transport", "Public transit", "Housing/mobility", "public_transport_commute_pct", 1.0),
    ("z_density", "Density", "Urban form", "log_pop_density", 1.0),
]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return math.nan
    return float(np.average(x[valid], weights=w[valid]))


def standardise_within_city(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for z_name, _, _, raw_name, direction in PREDICTORS:
        out[z_name] = np.nan
        for _, idx in out.groupby("result_city_key", sort=False).groups.items():
            values = pd.to_numeric(out.loc[idx, raw_name], errors="coerce") * direction
            valid = values.notna() & np.isfinite(values)
            if valid.sum() < 2:
                continue
            sd = float(values.loc[valid].std(ddof=0))
            if not np.isfinite(sd) or sd <= 0:
                out.loc[values.index[valid], z_name] = 0.0
            else:
                out.loc[values.index[valid], z_name] = (
                    values.loc[valid] - float(values.loc[valid].mean())
                ) / sd
    return out


def add_oriented_quintiles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for _, label, _, raw_name, direction in PREDICTORS:
        q_name = "q_" + label.lower().replace("/", "_").replace(" ", "_")
        out[q_name] = np.nan
        for _, idx in out.groupby("result_city_key", sort=False).groups.items():
            values = pd.to_numeric(out.loc[idx, raw_name], errors="coerce") * direction
            valid = values.notna() & np.isfinite(values)
            if valid.sum() < 5:
                continue
            ranks = values.loc[valid].rank(method="first")
            out.loc[ranks.index, q_name] = pd.qcut(ranks, 5, labels=False).astype(float) + 1
    return out


def load_scale(path: Path, scale: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[frame["scale"].eq(scale)].copy()
    frame["result_city_key"] = frame["result_city_key"].astype(str)
    frame["spatial_unit_id"] = frame["spatial_unit_id"].astype(str)
    for col in ("population", "land_area_m2", "city_intersection_fraction", "canopy_reference_0p6m"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.loc[
        (frame["population"] > 0)
        & (frame["land_area_m2"] > 0)
        & frame["canopy_reference_0p6m"].between(0, 100, inclusive="both")
        & frame["city_intersection_fraction"].between(0, 1.01, inclusive="both")
    ].copy()
    if frame["result_city_key"].nunique() != 50:
        raise RuntimeError(f"{scale}: expected 50 cities before thresholding")
    return frame


def sample_scale(base: pd.DataFrame, sample_name: str, threshold: float) -> pd.DataFrame:
    out = base.loc[base["city_intersection_fraction"] >= threshold].copy()
    out["sample"] = sample_name
    if out["result_city_key"].nunique() != 50:
        raise RuntimeError(f"{sample_name}: threshold removed at least one city")
    out = standardise_within_city(out)
    return out.reset_index(drop=True)


def model_weights(frame: pd.DataFrame, scheme: str) -> np.ndarray:
    population = frame["population"].to_numpy(float)
    if scheme == "population":
        weights = population.copy()
    elif scheme == "equal_unit":
        weights = np.ones(len(frame), dtype=float)
    elif scheme == "city_balanced_population":
        city_totals = frame.groupby("result_city_key")["population"].transform("sum").to_numpy(float)
        weights = population / city_totals
    else:
        raise ValueError(scheme)
    weights = weights / np.nanmean(weights)
    return weights


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return BH rejection flags and adjusted q values without external dependencies."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted <= alpha, adjusted


def fit_single_gradient(
    frame: pd.DataFrame,
    predictor: str,
    weight_scheme: str,
    rademacher: np.ndarray,
) -> dict[str, float | int]:
    cols = ["canopy_reference_0p6m", predictor, "population", "result_city_key"]
    d = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d.loc[(d["population"] > 0)].copy()
    cities = sorted(d["result_city_key"].unique().tolist())
    city_index = {city: i for i, city in enumerate(cities)}
    cluster = d["result_city_key"].map(city_index).to_numpy(int)
    x = d[predictor].to_numpy(float)
    y = d["canopy_reference_0p6m"].to_numpy(float)
    w = model_weights(d, weight_scheme)

    sum_w = np.bincount(cluster, weights=w, minlength=len(cities))
    mean_x = np.bincount(cluster, weights=w * x, minlength=len(cities)) / sum_w
    mean_y = np.bincount(cluster, weights=w * y, minlength=len(cities)) / sum_w
    xc = x - mean_x[cluster]
    yc = y - mean_y[cluster]
    denominator = float(np.sum(w * xc * xc))
    beta = float(np.sum(w * xc * yc) / denominator)
    residual = yc - beta * xc

    scores = np.bincount(cluster, weights=w * xc * residual, minlength=len(cities))
    g = len(cities)
    n = len(d)
    k = g + 1
    cr1 = (g / (g - 1)) * ((n - 1) / (n - k))
    se_cr1 = float(np.sqrt(cr1 * np.sum(scores * scores) / (denominator * denominator)))
    if g != 50:
        raise RuntimeError(f"Expected 50 city clusters, found {g}")
    t_critical = T_975_DF49

    draws = rademacher[:, :g]
    beta_star = beta + draws.dot(scores) / denominator
    ci_low, ci_high = np.quantile(beta_star, [0.025, 0.975])

    null_scores = np.bincount(cluster, weights=w * xc * yc, minlength=g)
    beta_null = draws.dot(null_scores) / denominator
    p_wild = float((1 + np.sum(np.abs(beta_null) >= abs(beta))) / (len(beta_null) + 1))

    return {
        "estimate_pp_per_within_city_sd": beta,
        "wild_ci_low": float(ci_low),
        "wild_ci_high": float(ci_high),
        "wild_p": p_wild,
        "cr1_se": se_cr1,
        "cr1_ci_low": beta - t_critical * se_cr1,
        "cr1_ci_high": beta + t_critical * se_cr1,
        "n_spatial_units": n,
        "n_cities": g,
    }


def fit_all_models(samples: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rademacher = rng.choice(np.array([-1.0, 1.0]), size=(N_BOOT, 50), replace=True)
    rows: list[dict[str, object]] = []
    for (scale, sample_name), frame in samples.items():
        for weight_scheme in WEIGHT_SCHEMES:
            family_indices: list[int] = []
            family_p: list[float] = []
            for predictor, label, domain, raw, direction in PREDICTORS:
                result = fit_single_gradient(frame, predictor, weight_scheme, rademacher)
                rows.append(
                    {
                        "scale": scale,
                        "sample": sample_name,
                        "minimum_city_intersection_fraction": SAMPLES[sample_name],
                        "weight_scheme": weight_scheme,
                        "predictor": predictor,
                        "predictor_label": label,
                        "domain": domain,
                        "raw_variable": raw,
                        "orientation": "higher disadvantage/concentration" if direction > 0 else "lower income",
                        **result,
                    }
                )
                family_indices.append(len(rows) - 1)
                family_p.append(float(result["wild_p"]))
            reject, qvals = benjamini_hochberg(family_p, alpha=0.05)
            for idx, rejected, qval in zip(family_indices, reject, qvals):
                rows[idx]["fdr_q"] = float(qval)
                rows[idx]["significant_fdr05"] = bool(rejected)
    return pd.DataFrame(rows)


def bootstrap_city_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return math.nan, math.nan, math.nan
    indices = rng.integers(0, len(valid), size=(N_BOOT, len(valid)))
    boot = valid[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(valid.mean()), float(lo), float(hi)


def build_descriptive_outputs(primary_bg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(CITY_LABELS)[["city_key", "city_display"]].drop_duplicates()
    frame = add_oriented_quintiles(primary_bg)
    frame = frame.merge(
        labels,
        left_on="result_city_key",
        right_on="city_key",
        how="left",
        validate="many_to_one",
    )
    if frame["city_display"].isna().any():
        raise RuntimeError("City display labels are incomplete")

    city_summary_rows: list[dict[str, object]] = []
    for city_key, city in frame.groupby("result_city_key", sort=False):
        city_summary_rows.append(
            {
                "city_key": city_key,
                "city_display": city["city_display"].iloc[0],
                "n_block_groups": len(city),
                "population": float(city["population"].sum()),
                "population_weighted_canopy_pct": weighted_mean(
                    city["canopy_reference_0p6m"], city["population"]
                ),
            }
        )
    city_summary = pd.DataFrame(city_summary_rows)

    quintile_rows: list[dict[str, object]] = []
    for _, label, domain, _, _ in PREDICTORS:
        q_name = "q_" + label.lower().replace("/", "_").replace(" ", "_")
        for city_key, city in frame.dropna(subset=[q_name]).groupby("result_city_key", sort=False):
            for quintile, group in city.groupby(q_name):
                quintile_rows.append(
                    {
                        "city_key": city_key,
                        "city_display": city["city_display"].iloc[0],
                        "predictor_label": label,
                        "domain": domain,
                        "quintile": int(quintile),
                        "n_block_groups": len(group),
                        "population": float(group["population"].sum()),
                        "population_weighted_canopy_pct": weighted_mean(
                            group["canopy_reference_0p6m"], group["population"]
                        ),
                    }
                )
    city_quintiles = pd.DataFrame(quintile_rows)

    city_matrix = city_summary.copy()
    for _, label, _, _, _ in PREDICTORS:
        sub = city_quintiles.loc[city_quintiles["predictor_label"].eq(label)]
        wide = sub.pivot(index="city_key", columns="quintile", values="population_weighted_canopy_pct")
        contrast = (wide[5] - wide[1]).rename(label)
        city_matrix = city_matrix.merge(contrast, left_on="city_key", right_index=True, how="left", validate="one_to_one")
    gap_labels = [item[1] for item in PREDICTORS]
    city_matrix["mean_target_minus_reference_gap_pp"] = city_matrix[gap_labels].mean(axis=1)
    city_matrix["mean_social_canopy_deficit_pp"] = -city_matrix["mean_target_minus_reference_gap_pp"]
    canopy_median = float(city_matrix["population_weighted_canopy_pct"].median())
    deficit_median = float(city_matrix["mean_social_canopy_deficit_pp"].median())
    city_matrix["city_type"] = np.select(
        [
            (city_matrix["population_weighted_canopy_pct"] < canopy_median) & (city_matrix["mean_social_canopy_deficit_pp"] >= deficit_median),
            (city_matrix["population_weighted_canopy_pct"] >= canopy_median) & (city_matrix["mean_social_canopy_deficit_pp"] >= deficit_median),
            (city_matrix["population_weighted_canopy_pct"] < canopy_median) & (city_matrix["mean_social_canopy_deficit_pp"] < deficit_median),
        ],
        ["lower canopy / larger social deficit", "higher canopy / larger social deficit", "lower canopy / smaller social deficit"],
        default="higher canopy / smaller social deficit",
    )

    rng = np.random.default_rng(SEED + 1)
    national_rows: list[dict[str, object]] = []
    for label in ["Low income", "Hispanic/Latino", "Renters", "No vehicle"]:
        for quintile in range(1, 6):
            values = city_quintiles.loc[
                city_quintiles["predictor_label"].eq(label)
                & city_quintiles["quintile"].eq(quintile),
                "population_weighted_canopy_pct",
            ].to_numpy(float)
            mean, lo, hi = bootstrap_city_mean(values, rng)
            national_rows.append(
                {
                    "predictor_label": label,
                    "quintile": quintile,
                    "mean_canopy_pct": mean,
                    "city_bootstrap_ci_low": lo,
                    "city_bootstrap_ci_high": hi,
                    "n_cities": int(np.isfinite(values).sum()),
                }
            )
    national_curves = pd.DataFrame(national_rows)
    return city_matrix, city_quintiles, national_curves


def build_sample_inventory(samples: dict[tuple[str, str], pd.DataFrame], bases: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scale, base in bases.items():
        base_by_city = base.groupby("result_city_key").agg(
            units_available_above_80=("spatial_unit_id", "size"),
            population_available_above_80=("population", "sum"),
        )
        for sample_name in SAMPLES:
            selected = samples[(scale, sample_name)]
            grouped = selected.groupby("result_city_key").agg(
                retained_units=("spatial_unit_id", "size"),
                retained_population=("population", "sum"),
                minimum_observed_intersection=("city_intersection_fraction", "min"),
            )
            joined = base_by_city.join(grouped, how="left").reset_index()
            joined["scale"] = scale
            joined["sample"] = sample_name
            joined["threshold"] = SAMPLES[sample_name]
            joined["unit_retention_vs_80_pct"] = joined["retained_units"] / joined["units_available_above_80"] * 100
            joined["population_retention_vs_80_pct"] = joined["retained_population"] / joined["population_available_above_80"] * 100
            rows.extend(joined.to_dict("records"))
    return pd.DataFrame(rows)


def make_sensitivity_summary(models: pd.DataFrame) -> pd.DataFrame:
    primary = models.loc[
        models["sample"].eq(PRIMARY_SAMPLE)
        & models["weight_scheme"].eq(PRIMARY_WEIGHT)
    ].copy()
    rows: list[dict[str, object]] = []
    for scale in ("block_group", "tract"):
        for label in [item[1] for item in PREDICTORS]:
            all_rows = models.loc[
                models["scale"].eq(scale) & models["predictor_label"].eq(label)
            ]
            p = primary.loc[
                primary["scale"].eq(scale) & primary["predictor_label"].eq(label)
            ].iloc[0]
            estimates = all_rows["estimate_pp_per_within_city_sd"].to_numpy(float)
            signs = np.sign(estimates)
            primary_sign = np.sign(float(p["estimate_pp_per_within_city_sd"]))
            rows.append(
                {
                    "scale": scale,
                    "predictor_label": label,
                    "primary_estimate": float(p["estimate_pp_per_within_city_sd"]),
                    "primary_wild_ci_low": float(p["wild_ci_low"]),
                    "primary_wild_ci_high": float(p["wild_ci_high"]),
                    "primary_fdr_q": float(p["fdr_q"]),
                    "minimum_estimate_across_9_specifications": float(np.min(estimates)),
                    "maximum_estimate_across_9_specifications": float(np.max(estimates)),
                    "direction_retention_across_9_specifications_pct": float(np.mean(signs == primary_sign) * 100),
                    "n_specifications": len(estimates),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    bases = {
        "block_group": load_scale(BG_INPUT, "block_group"),
        "tract": load_scale(TRACT_INPUT, "tract"),
    }
    samples: dict[tuple[str, str], pd.DataFrame] = {}
    for scale, base in bases.items():
        for sample_name, threshold in SAMPLES.items():
            samples[(scale, sample_name)] = sample_scale(base, sample_name, threshold)

    models = fit_all_models(samples)
    models.to_csv(TABLE_DIR / "result2_boundary_weight_wild_cluster_models.csv", index=False, encoding="utf-8-sig")

    primary_bg = samples[("block_group", PRIMARY_SAMPLE)]
    primary_tract = samples[("tract", PRIMARY_SAMPLE)]
    primary_bg.to_csv(SOURCE_DIR / "result2_primary_fully_contained_block_groups.csv", index=False, encoding="utf-8-sig")
    primary_tract.to_csv(SOURCE_DIR / "result2_primary_fully_contained_tracts.csv", index=False, encoding="utf-8-sig")

    city_matrix, city_quintiles, national_curves = build_descriptive_outputs(primary_bg)
    city_matrix.to_csv(TABLE_DIR / "result2_primary_city_social_gap_and_type.csv", index=False, encoding="utf-8-sig")
    city_quintiles.to_csv(SOURCE_DIR / "result2_primary_city_quintile_curves.csv", index=False, encoding="utf-8-sig")
    national_curves.to_csv(SOURCE_DIR / "result2_primary_national_direct_gradient_curves.csv", index=False, encoding="utf-8-sig")

    inventory = build_sample_inventory(samples, bases)
    inventory.to_csv(TABLE_DIR / "result2_boundary_sample_inventory_by_city.csv", index=False, encoding="utf-8-sig")
    sensitivity = make_sensitivity_summary(models)
    sensitivity.to_csv(TABLE_DIR / "result2_primary_and_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    primary_models = models.loc[
        models["sample"].eq(PRIMARY_SAMPLE) & models["weight_scheme"].eq(PRIMARY_WEIGHT)
    ].copy()
    primary_models.to_csv(TABLE_DIR / "result2_primary_wild_cluster_models.csv", index=False, encoding="utf-8-sig")

    summary = {
        "primary_rule": "city_intersection_fraction >= 0.9995",
        "numerical_tolerance_percentage_points": (1 - FULLY_CONTAINED_TOLERANCE) * 100,
        "bootstrap_draws": N_BOOT,
        "bootstrap_seed": SEED,
        "primary_block_groups": len(primary_bg),
        "primary_tracts": len(primary_tract),
        "primary_cities_block_group": int(primary_bg["result_city_key"].nunique()),
        "primary_cities_tract": int(primary_tract["result_city_key"].nunique()),
        "original_80_block_groups": len(samples[("block_group", "interior_80")]),
        "original_80_tracts": len(samples[("tract", "interior_80")]),
        "primary_population_block_group": float(primary_bg["population"].sum()),
        "primary_population_tract": float(primary_tract["population"].sum()),
        "primary_fdr_significant_block_group": int(primary_models.loc[primary_models["scale"].eq("block_group"), "significant_fdr05"].sum()),
        "primary_fdr_significant_tract": int(primary_models.loc[primary_models["scale"].eq("tract"), "significant_fdr05"].sum()),
        "all_9_specifications_direction_retention_100pct": int((sensitivity["direction_retention_across_9_specifications_pct"] == 100).sum()),
        "sensitivity_rows": len(sensitivity),
        "estimand": "canopy percentage-point difference per one within-city standard deviation of one predictor",
        "interpretation_boundary": "separate single-predictor gradients; not mutually adjusted or causal mechanisms",
    }
    (QA_DIR / "module1_result2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    variable_ledger = pd.DataFrame(
        [
            {
                "quantity": "canopy outcome",
                "source_scope": "complete census-unit land area and city-clipped canopy numerator; primary units are effectively fully contained",
                "transformation": "none",
                "unit_in_model": "percentage points",
            },
            {
                "quantity": "ACS social predictor",
                "source_scope": "complete census unit",
                "transformation": "unweighted within-city z score; low income reverses median household income",
                "unit_in_model": "within-city standard deviations",
            },
            {
                "quantity": "population",
                "source_scope": "complete census unit",
                "transformation": "analytic weight in primary model",
                "unit_in_model": "persons",
            },
            {
                "quantity": "city intersection fraction",
                "source_scope": "city-clipped valid raster area divided by Census land-plus-water area",
                "transformation": "thresholded at 0.80, 0.95 and 0.9995",
                "unit_in_model": "dimensionless",
            },
        ]
    )
    variable_ledger.to_csv(QA_DIR / "result2_unit_and_denominator_ledger.csv", index=False, encoding="utf-8-sig")

    methods = f"""Result 2 audit specification

Primary spatial-unit rule: city_intersection_fraction >= {FULLY_CONTAINED_TOLERANCE:.4f}. This 0.05-percentage-point tolerance accommodates rasterisation and area-conversion discrepancies around unity. The 80% and 95% thresholds were retained only as sensitivity samples.

Canopy remained an unstandardised percentage outcome. Each social attribute was oriented so that larger values represented lower income or a larger group share, then standardised within city using the unweighted city mean and population standard deviation. Separate population-weighted city fixed-effect models were fitted for each prespecified predictor. Inference used {N_BOOT:,} city-level Rademacher wild-cluster residual-bootstrap draws across 50 cities. Benjamini-Hochberg correction was applied across the 11 predictors within each scale, boundary rule and weighting specification. Equal-unit and city-balanced population weights were compared with the primary population weights. Because predictors were fitted separately and are correlated, coefficients describe overlapping within-city social gradients rather than mutually independent or causal mechanisms.
"""
    (TEXT_DIR / "result2_methods_module1.txt").write_text(methods, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
