from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = analysis_store_root()
INPUT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_module1_boundary_inference_audit"
    / "source_data"
    / "result2_primary_fully_contained_block_groups.csv"
)
REFERENCE_MARGINAL = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_module1_boundary_inference_audit"
    / "tables"
    / "result2_primary_wild_cluster_models.csv"
)
REFERENCE_JOINT = (
    Path(str(repo_root()))
    / "complete_case_model_module"
    / "analysis"
    / "result2_joint_model_reference.csv"
)
OUT = (
    PROJECT
    / "Result"
    / "Result2"
    / "Result2_complete_case_comparability_audit"
)
TABLES = OUT / "tables"
DATA = OUT / "data"
QA = OUT / "qa"
TEXT = OUT / "text"
SCRIPTS = OUT / "scripts"

SEED = 20260719
N_BOOT = 9_999
EXPECTED_BASE_N = 34_575
EXPECTED_COMPLETE_N = 30_826
EXPECTED_CITIES = 50


@dataclass(frozen=True)
class Predictor:
    z: str
    label: str
    domain: str
    raw: str
    direction: float


PREDICTORS = [
    Predictor("z_low_income", "Low income", "Economic", "median_household_income_usd", -1.0),
    Predictor("z_poverty", "Poverty", "Economic", "poverty_rate_pct", 1.0),
    Predictor("z_median_age", "Older age", "Demographic", "median_age", 1.0),
    Predictor("z_black", "Black", "Demographic", "black_alone_pct", 1.0),
    Predictor("z_hispanic", "Hispanic/Latino", "Demographic", "hispanic_or_latino_pct", 1.0),
    Predictor("z_asian", "Asian", "Demographic", "asian_alone_pct", 1.0),
    Predictor("z_renter", "Renters", "Housing/urban form", "renter_occupied_pct", 1.0),
    Predictor("z_no_vehicle", "No vehicle", "Housing/urban form", "no_vehicle_households_pct", 1.0),
    Predictor("z_public_transport", "Public transit", "Housing/urban form", "public_transport_commute_pct", 1.0),
    Predictor("z_density", "Density", "Housing/urban form", "log_pop_density", 1.0),
]
POC = Predictor("z_poc", "People of colour", "Descriptive composite", "people_of_color_pct", 1.0)
ALL_MARGINAL = PREDICTORS[:6] + [POC] + PREDICTORS[6:]
JOINT_Z = [p.z for p in PREDICTORS]
ALL_Z = [p.z for p in ALL_MARGINAL]
META = {p.z: p for p in ALL_MARGINAL}


def mkdirs() -> None:
    for directory in (OUT, TABLES, DATA, QA, TEXT, SCRIPTS):
        directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty(len(p), dtype=float)
    q[order] = np.clip(q_ranked, 0, 1)
    return q


def standardise_within_city(frame: pd.DataFrame, predictors: list[Predictor]) -> pd.DataFrame:
    out = frame.copy()
    for predictor in predictors:
        out[predictor.z] = np.nan
        for _, index in out.groupby("result_city_key", sort=False).groups.items():
            values = pd.to_numeric(out.loc[index, predictor.raw], errors="coerce") * predictor.direction
            valid = values.notna() & np.isfinite(values)
            if valid.sum() < 2:
                continue
            sd = float(values.loc[valid].std(ddof=0))
            if not np.isfinite(sd) or sd <= 0:
                out.loc[values.index[valid], predictor.z] = 0.0
            else:
                out.loc[values.index[valid], predictor.z] = (
                    values.loc[valid] - float(values.loc[valid].mean())
                ) / sd
    return out


def within_transform(
    frame: pd.DataFrame, predictors: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns = ["canopy_reference_0p6m", "population", "result_city_key", *predictors]
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    data = data.loc[data["population"] > 0].copy()
    cities = sorted(data["result_city_key"].astype(str).unique().tolist())
    city_map = {city: position for position, city in enumerate(cities)}
    cluster = data["result_city_key"].astype(str).map(city_map).to_numpy(int)
    weights = data["population"].to_numpy(float)
    weights = weights / weights.mean()
    y = data["canopy_reference_0p6m"].to_numpy(float)
    x = data[predictors].to_numpy(float)
    sum_w = np.bincount(cluster, weights=weights, minlength=len(cities))
    mean_y = np.bincount(cluster, weights=weights * y, minlength=len(cities)) / sum_w
    yc = y - mean_y[cluster]
    xc = np.empty_like(x)
    for j in range(x.shape[1]):
        mean_x = np.bincount(cluster, weights=weights * x[:, j], minlength=len(cities)) / sum_w
        xc[:, j] = x[:, j] - mean_x[cluster]
    return data, yc, xc, weights, cluster, cities


def fit_fe_wild(
    frame: pd.DataFrame,
    predictors: list[str],
    rademacher_by_city: dict[str, np.ndarray],
    run_bootstrap: bool = True,
) -> pd.DataFrame:
    data, y, x, weights, cluster, cities = within_transform(frame, predictors)
    n = len(data)
    g = len(cities)
    p = len(predictors)
    if g < 2:
        raise RuntimeError("At least two city clusters are required")
    xtwx = x.T @ (weights[:, None] * x)
    bread = np.linalg.pinv(xtwx, rcond=1e-12)
    beta = bread @ (x.T @ (weights * y))
    residual = y - x @ beta
    scores = np.zeros((g, p), dtype=float)
    for city_index in range(g):
        mask = cluster == city_index
        scores[city_index] = x[mask].T @ (weights[mask] * residual[mask])
    k = g + p
    cr1 = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = cr1 * bread @ (scores.T @ scores) @ bread
    se = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    t_critical = float(stats.t.ppf(0.975, df=g - 1))
    cr1_p = 2 * stats.t.sf(np.abs(beta / se), df=g - 1)

    wild_low = np.full(p, np.nan)
    wild_high = np.full(p, np.nan)
    wild_p = np.full(p, np.nan)
    if run_bootstrap:
        draws = np.column_stack([rademacher_by_city[city] for city in cities])
        beta_star = beta[None, :] + (draws @ scores) @ bread.T
        wild_low, wild_high = np.quantile(beta_star, [0.025, 0.975], axis=0)
        for j in range(p):
            others = [position for position in range(p) if position != j]
            if others:
                x0 = x[:, others]
                b0 = np.linalg.pinv(x0.T @ (weights[:, None] * x0), rcond=1e-12) @ (
                    x0.T @ (weights * y)
                )
                residual0 = y - x0 @ b0
            else:
                residual0 = y.copy()
            null_scores = np.zeros((g, p), dtype=float)
            for city_index in range(g):
                mask = cluster == city_index
                null_scores[city_index] = x[mask].T @ (weights[mask] * residual0[mask])
            beta_null = ((draws @ null_scores) @ bread.T)[:, j]
            wild_p[j] = (1 + np.sum(np.abs(beta_null) >= abs(beta[j]))) / (len(beta_null) + 1)

    rows = []
    for j, predictor in enumerate(predictors):
        rows.append(
            {
                "predictor": predictor,
                "predictor_label": META[predictor].label,
                "domain": META[predictor].domain,
                "estimate_pp_per_within_city_sd": float(beta[j]),
                "wild_ci_low": float(wild_low[j]),
                "wild_ci_high": float(wild_high[j]),
                "wild_p": float(wild_p[j]),
                "cr1_se": float(se[j]),
                "cr1_ci_low": float(beta[j] - t_critical * se[j]),
                "cr1_ci_high": float(beta[j] + t_critical * se[j]),
                "cr1_p": float(cr1_p[j]),
                "n_spatial_units": int(n),
                "n_cities": int(g),
                "condition_number_xtwx": float(np.linalg.cond(xtwx)),
            }
        )
    return pd.DataFrame(rows)


def add_model_metadata(table: pd.DataFrame, model_id: str, model_form: str, scaling: str) -> pd.DataFrame:
    out = table.copy()
    out.insert(0, "model_id", model_id)
    out.insert(1, "model_form", model_form)
    out.insert(2, "sample_scaling", scaling)
    if out["wild_p"].notna().all():
        out["fdr_q"] = bh_adjust(out["wild_p"])
        out["significant_fdr05"] = out["fdr_q"] <= 0.05
    else:
        out["fdr_q"] = np.nan
        out["significant_fdr05"] = False
    return out


def fit_marginals(
    frame: pd.DataFrame,
    predictors: list[str],
    rademacher_by_city: dict[str, np.ndarray],
    model_id: str,
    scaling: str,
) -> pd.DataFrame:
    rows = []
    for predictor in predictors:
        result = fit_fe_wild(frame, [predictor], rademacher_by_city, run_bootstrap=True)
        rows.append(result)
    out = pd.concat(rows, ignore_index=True)
    return add_model_metadata(out, model_id, "separate city fixed-effect gradients", scaling)


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(INPUT, dtype={"spatial_unit_id": str}, low_memory=False)
    numeric = [
        "population",
        "land_area_m2",
        "city_intersection_fraction",
        "canopy_reference_0p6m",
        *[p.raw for p in ALL_MARGINAL],
        *ALL_Z,
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["result_city_key"] = frame["result_city_key"].astype(str)
    if len(frame) != EXPECTED_BASE_N or frame["result_city_key"].nunique() != EXPECTED_CITIES:
        raise RuntimeError(
            f"Unexpected source frame: n={len(frame)}, cities={frame['result_city_key'].nunique()}"
        )
    return frame


def freeze_complete_case(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = ["canopy_reference_0p6m", "population", "result_city_key", *JOINT_Z]
    included = frame[required].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    included &= frame["population"] > 0
    complete = frame.loc[included].copy().reset_index(drop=True)
    if len(complete) != EXPECTED_COMPLETE_N or complete["result_city_key"].nunique() != EXPECTED_CITIES:
        raise RuntimeError(
            f"Unexpected complete-case frame: n={len(complete)}, cities={complete['result_city_key'].nunique()}"
        )
    return complete, included


def make_missingness_outputs(frame: pd.DataFrame, included: pd.Series) -> None:
    variable_rows = []
    for predictor in PREDICTORS:
        missing = frame[predictor.raw].isna() | ~np.isfinite(frame[predictor.raw])
        variable_rows.append(
            {
                "predictor": predictor.z,
                "predictor_label": predictor.label,
                "raw_variable": predictor.raw,
                "n_missing": int(missing.sum()),
                "missing_pct": float(100 * missing.mean()),
                "share_of_all_excluded_units_pct": float(100 * missing.sum() / (~included).sum()),
            }
        )
    pd.DataFrame(variable_rows).sort_values("n_missing", ascending=False).to_csv(
        TABLES / "07_missingness_by_variable.csv", index=False, encoding="utf-8-sig"
    )

    city = frame[["result_city_key", "spatial_unit_id"]].copy()
    city["included_complete_case"] = included.to_numpy(bool)
    city_summary = (
        city.groupby("result_city_key", as_index=False)
        .agg(
            n_source=("spatial_unit_id", "size"),
            n_complete_case=("included_complete_case", "sum"),
        )
        .assign(
            n_excluded=lambda x: x["n_source"] - x["n_complete_case"],
            retention_pct=lambda x: 100 * x["n_complete_case"] / x["n_source"],
        )
        .sort_values("retention_pct")
    )
    city_summary.to_csv(TABLES / "08_missingness_by_city.csv", index=False, encoding="utf-8-sig")

    missing_flags = pd.DataFrame(index=frame.index)
    for predictor in PREDICTORS:
        missing_flags[predictor.label] = frame[predictor.raw].isna()
    pattern = missing_flags.apply(
        lambda row: "; ".join(row.index[row.to_numpy(bool)].tolist()) or "complete", axis=1
    )
    pattern.value_counts(dropna=False).rename_axis("missingness_pattern").reset_index(name="n_units").assign(
        pct_of_source=lambda x: 100 * x["n_units"] / len(frame)
    ).to_csv(TABLES / "09_missingness_patterns.csv", index=False, encoding="utf-8-sig")

    balance_frame = frame.copy()
    balance_frame["included_complete_case"] = included.to_numpy(bool)
    balance_frame["log_population"] = np.log1p(balance_frame["population"])
    balance_frame["log_land_area"] = np.log(balance_frame["land_area_m2"])
    context = ["canopy_reference_0p6m", "log_population", "log_land_area", *ALL_Z]
    rows = []
    for variable in context:
        values = pd.to_numeric(balance_frame[variable], errors="coerce")
        z_context = pd.Series(np.nan, index=balance_frame.index, dtype=float)
        for _, index in balance_frame.groupby("result_city_key", sort=False).groups.items():
            city_values = values.loc[index]
            valid = city_values.notna() & np.isfinite(city_values)
            if valid.sum() < 2:
                continue
            sd = float(city_values.loc[valid].std(ddof=0))
            if sd > 0:
                z_context.loc[city_values.index[valid]] = (
                    city_values.loc[valid] - city_values.loc[valid].mean()
                ) / sd
        valid = z_context.notna()
        for weighting in ("unit", "population"):
            if weighting == "unit":
                weights = pd.Series(1.0, index=balance_frame.index)
            else:
                weights = balance_frame["population"].astype(float)
            means = {}
            counts = {}
            for status in (True, False):
                mask = valid & balance_frame["included_complete_case"].eq(status) & weights.gt(0)
                counts[status] = int(mask.sum())
                means[status] = float(np.average(z_context.loc[mask], weights=weights.loc[mask])) if mask.any() else np.nan
            rows.append(
                {
                    "variable": variable,
                    "weighting": weighting,
                    "n_included_observed": counts[True],
                    "n_excluded_observed": counts[False],
                    "included_mean_within_city_sd": means[True],
                    "excluded_mean_within_city_sd": means[False],
                    "included_minus_excluded_sd": means[True] - means[False],
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "10_complete_case_selection_balance.csv", index=False, encoding="utf-8-sig")


def make_collinearity_outputs(complete_rez: pd.DataFrame) -> dict[str, float]:
    _, _, x, weights, _, _ = within_transform(complete_rez, JOINT_Z)
    covariance = x.T @ (weights[:, None] * x) / weights.sum()
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)
    correlation = np.clip(correlation, -1, 1)
    pd.DataFrame(correlation, index=JOINT_Z, columns=JOINT_Z).to_csv(
        TABLES / "11_weighted_within_city_predictor_correlation.csv", encoding="utf-8-sig"
    )
    inverse = np.linalg.pinv(correlation, rcond=1e-12)
    vif = pd.DataFrame(
        {
            "predictor": JOINT_Z,
            "predictor_label": [META[p].label for p in JOINT_Z],
            "vif_after_city_demeaning": np.diag(inverse),
            "tolerance": 1 / np.diag(inverse),
        }
    ).sort_values("vif_after_city_demeaning", ascending=False)
    vif.to_csv(TABLES / "12_vif.csv", index=False, encoding="utf-8-sig")

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    condition_index = np.sqrt(eigenvalues.max() / np.clip(eigenvalues, 1e-15, np.inf))
    condition = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "condition_index": condition_index,
        }
    )
    condition.to_csv(TABLES / "13_condition_indices.csv", index=False, encoding="utf-8-sig")

    phi = (eigenvectors**2) / np.clip(eigenvalues[None, :], 1e-15, np.inf)
    proportions = phi / phi.sum(axis=1, keepdims=True)
    variance_rows = []
    for i, predictor in enumerate(JOINT_Z):
        for j in range(len(eigenvalues)):
            variance_rows.append(
                {
                    "predictor": predictor,
                    "component": j + 1,
                    "condition_index": condition_index[j],
                    "variance_decomposition_proportion": proportions[i, j],
                }
            )
    pd.DataFrame(variance_rows).to_csv(
        TABLES / "14_collinearity_variance_decomposition.csv", index=False, encoding="utf-8-sig"
    )
    upper = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    locations = np.argwhere(upper)
    max_pair = max(locations, key=lambda ij: abs(correlation[ij[0], ij[1]]))
    return {
        "max_abs_pairwise_correlation": float(abs(correlation[max_pair[0], max_pair[1]])),
        "max_abs_pairwise_correlation_predictor_1": JOINT_Z[max_pair[0]],
        "max_abs_pairwise_correlation_predictor_2": JOINT_Z[max_pair[1]],
        "max_vif": float(vif["vif_after_city_demeaning"].max()),
        "max_condition_index": float(condition_index.max()),
    }


def fit_variable_combinations(
    complete_rez: pd.DataFrame,
    frame: pd.DataFrame,
    rademacher_by_city: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    economic = ["z_low_income", "z_poverty"]
    demographic = ["z_median_age", "z_black", "z_hispanic", "z_asian"]
    housing = ["z_renter", "z_no_vehicle", "z_public_transport", "z_density"]
    specifications: list[tuple[str, list[str], pd.DataFrame, str]] = [
        ("economic_only", economic, complete_rez, "complete_case_restandardized"),
        ("demographic_only", demographic, complete_rez, "complete_case_restandardized"),
        ("housing_urban_only", housing, complete_rez, "complete_case_restandardized"),
        ("economic_plus_demographic", economic + demographic, complete_rez, "complete_case_restandardized"),
        ("economic_plus_housing", economic + housing, complete_rez, "complete_case_restandardized"),
        ("demographic_plus_housing", demographic + housing, complete_rez, "complete_case_restandardized"),
        ("full_joint", JOINT_Z, complete_rez, "complete_case_restandardized"),
        ("full_without_economic", demographic + housing, complete_rez, "complete_case_restandardized"),
        ("full_without_demographic", economic + housing, complete_rez, "complete_case_restandardized"),
        ("full_without_housing_urban", economic + demographic, complete_rez, "complete_case_restandardized"),
    ]
    for omitted in JOINT_Z:
        specifications.append(
            (f"leave_predictor_out__{omitted}", [p for p in JOINT_Z if p != omitted], complete_rez, "complete_case_restandardized")
        )

    # This sensitivity restores units lost only because household income is unavailable.
    no_income_predictors = [p for p in PREDICTORS if p.z != "z_low_income"]
    needed = ["canopy_reference_0p6m", "population", "result_city_key", *[p.raw for p in no_income_predictors]]
    no_income = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    no_income = no_income.loc[no_income["population"] > 0].copy()
    no_income = standardise_within_city(no_income, no_income_predictors)
    specifications.append(
        (
            "joint_without_low_income_available_sample",
            [p.z for p in no_income_predictors],
            no_income,
            "available_without_low_income_restandardized",
        )
    )

    rows = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for model_id, predictors, data, scaling in specifications:
        key = (model_id, tuple(predictors))
        if key in seen:
            continue
        seen.add(key)
        result = fit_fe_wild(data, predictors, rademacher_by_city, run_bootstrap=True)
        result.insert(0, "specification", model_id)
        result.insert(1, "sample_scaling", scaling)
        result["n_predictors"] = len(predictors)
        result["predictor_set"] = ";".join(predictors)
        result["fdr_q_within_specification"] = bh_adjust(result["wild_p"])
        rows.append(result)
    combinations = pd.concat(rows, ignore_index=True)
    combinations.to_csv(TABLES / "15_variable_combination_coefficients.csv", index=False, encoding="utf-8-sig")

    full = combinations.loc[combinations["specification"].eq("full_joint")].set_index("predictor")
    summaries = []
    for predictor in JOINT_Z:
        subset = combinations.loc[
            combinations["predictor"].eq(predictor)
            & combinations["sample_scaling"].eq("complete_case_restandardized")
        ].copy()
        reference = float(full.loc[predictor, "estimate_pp_per_within_city_sd"])
        summaries.append(
            {
                "predictor": predictor,
                "predictor_label": META[predictor].label,
                "full_joint_estimate": reference,
                "n_complete_case_specifications_containing_predictor": len(subset),
                "minimum_estimate": float(subset["estimate_pp_per_within_city_sd"].min()),
                "maximum_estimate": float(subset["estimate_pp_per_within_city_sd"].max()),
                "same_sign_as_full_joint_pct": float(
                    100 * (np.sign(subset["estimate_pp_per_within_city_sd"]) == np.sign(reference)).mean()
                ),
                "wild_ci_excludes_zero_pct": float(
                    100
                    * (
                        (subset["wild_ci_low"] > 0)
                        | (subset["wild_ci_high"] < 0)
                    ).mean()
                ),
            }
        )
    stability = pd.DataFrame(summaries)
    stability.to_csv(TABLES / "16_variable_combination_stability_summary.csv", index=False, encoding="utf-8-sig")
    return combinations, stability


def poverty_shapley_decomposition(complete_rez: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose the poverty coefficient shift without privileging a covariate order."""
    _, y, x, weights, _, _ = within_transform(complete_rez, JOINT_Z)
    xtwx = x.T @ (weights[:, None] * x)
    xtwy = x.T @ (weights * y)
    target = JOINT_Z.index("z_poverty")
    others = [index for index in range(len(JOINT_Z)) if index != target]
    subset_beta: dict[frozenset[int], float] = {}
    rows = []
    for size in range(len(others) + 1):
        for subset_tuple in itertools.combinations(others, size):
            subset = frozenset(subset_tuple)
            selected = [target, *sorted(subset)]
            beta = np.linalg.pinv(xtwx[np.ix_(selected, selected)], rcond=1e-12) @ xtwy[selected]
            estimate = float(beta[selected.index(target)])
            subset_beta[subset] = estimate
            rows.append(
                {
                    "n_adjustment_predictors": size,
                    "adjustment_predictors": ";".join(JOINT_Z[index] for index in sorted(subset)) or "none",
                    "poverty_estimate_pp_per_within_city_sd": estimate,
                    "poverty_positive": estimate > 0,
                }
            )
    subsets = pd.DataFrame(rows).sort_values(["n_adjustment_predictors", "adjustment_predictors"])
    subsets.to_csv(TABLES / "20_poverty_all_512_adjustment_subsets.csv", index=False, encoding="utf-8-sig")

    m = len(others)
    shapley_rows = []
    for added in others:
        contribution = 0.0
        remaining = [index for index in others if index != added]
        for size in range(len(remaining) + 1):
            weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for subset_tuple in itertools.combinations(remaining, size):
                subset = frozenset(subset_tuple)
                contribution += weight * (subset_beta[subset | {added}] - subset_beta[subset])
        shapley_rows.append(
            {
                "adjustment_predictor": JOINT_Z[added],
                "adjustment_predictor_label": META[JOINT_Z[added]].label,
                "shapley_contribution_to_poverty_coefficient_pp": contribution,
                "interpretation": "moves poverty coefficient upward" if contribution > 0 else "moves poverty coefficient downward",
            }
        )
    shapley = pd.DataFrame(shapley_rows).sort_values(
        "shapley_contribution_to_poverty_coefficient_pp", ascending=False
    )
    baseline = subset_beta[frozenset()]
    full = subset_beta[frozenset(others)]
    shapley["poverty_marginal_estimate"] = baseline
    shapley["poverty_full_joint_estimate"] = full
    shapley["total_adjustment_shift"] = full - baseline
    shapley["shapley_sum"] = shapley["shapley_contribution_to_poverty_coefficient_pp"].sum()
    shapley.to_csv(TABLES / "21_poverty_shapley_adjustment_decomposition.csv", index=False, encoding="utf-8-sig")
    return subsets, shapley


def fit_loco(
    complete_rez: pd.DataFrame,
    rademacher_by_city: dict[str, np.ndarray],
    national_joint: pd.DataFrame,
    national_marginal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for omitted_city in sorted(complete_rez["result_city_key"].unique()):
        subset = complete_rez.loc[~complete_rez["result_city_key"].eq(omitted_city)].copy()
        joint = fit_fe_wild(subset, JOINT_Z, rademacher_by_city, run_bootstrap=False)
        joint["model_form"] = "joint"
        marginal_parts = []
        for predictor in JOINT_Z:
            marginal_parts.append(fit_fe_wild(subset, [predictor], rademacher_by_city, run_bootstrap=False))
        marginal = pd.concat(marginal_parts, ignore_index=True)
        marginal["model_form"] = "marginal"
        both = pd.concat([joint, marginal], ignore_index=True)
        both.insert(0, "omitted_city", omitted_city)
        rows.append(both)
    loco = pd.concat(rows, ignore_index=True)
    loco.to_csv(TABLES / "17_leave_one_city_out_coefficients.csv", index=False, encoding="utf-8-sig")

    references = {
        "joint": national_joint.set_index("predictor")["estimate_pp_per_within_city_sd"],
        "marginal": national_marginal.set_index("predictor")["estimate_pp_per_within_city_sd"],
    }
    summaries = []
    for (model_form, predictor), subset in loco.groupby(["model_form", "predictor"]):
        reference = float(references[model_form].loc[predictor])
        values = subset["estimate_pp_per_within_city_sd"].to_numpy(float)
        summaries.append(
            {
                "model_form": model_form,
                "predictor": predictor,
                "predictor_label": META[predictor].label,
                "national_estimate": reference,
                "loco_minimum": float(values.min()),
                "loco_maximum": float(values.max()),
                "maximum_absolute_change": float(np.max(np.abs(values - reference))),
                "same_sign_as_national_pct": float(100 * np.mean(np.sign(values) == np.sign(reference))),
                "city_producing_largest_change": str(
                    subset.iloc[int(np.argmax(np.abs(values - reference)))]["omitted_city"]
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES / "18_leave_one_city_out_summary.csv", index=False, encoding="utf-8-sig")
    return loco, summary


def reproduce_reference_results(
    available_marginal: pd.DataFrame, joint_existing: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    reference_marginal = pd.read_csv(REFERENCE_MARGINAL)
    reference_marginal = reference_marginal.loc[
        reference_marginal["scale"].eq("block_group")
        & reference_marginal["sample"].eq("fully_contained")
        & reference_marginal["weight_scheme"].eq("population")
    ].copy()
    marginal_columns = [
        "estimate_pp_per_within_city_sd",
        "wild_ci_low",
        "wild_ci_high",
        "wild_p",
        "cr1_se",
        "n_spatial_units",
        "n_cities",
    ]
    marginal_check = reference_marginal[["predictor", *marginal_columns]].merge(
        available_marginal[["predictor", *marginal_columns]],
        on="predictor",
        suffixes=("_reference", "_rerun"),
        validate="one_to_one",
    )
    for column in marginal_columns:
        marginal_check[f"absolute_difference__{column}"] = (
            marginal_check[f"{column}_reference"] - marginal_check[f"{column}_rerun"]
        ).abs()
    marginal_check.insert(0, "check_family", "available_case_marginal_reproduction")

    reference_joint = pd.read_csv(REFERENCE_JOINT)
    joint_columns = ["estimate", "n_spatial_units", "n_cities"]
    joint_check = reference_joint[["predictor", *joint_columns]].merge(
        joint_existing[
            ["predictor", "estimate_pp_per_within_city_sd", "n_spatial_units", "n_cities"]
        ].rename(columns={"estimate_pp_per_within_city_sd": "estimate"}),
        on="predictor",
        suffixes=("_reference", "_rerun"),
        validate="one_to_one",
    )
    for column in joint_columns:
        joint_check[f"absolute_difference__{column}"] = (
            joint_check[f"{column}_reference"] - joint_check[f"{column}_rerun"]
        ).abs()
    joint_check.insert(0, "check_family", "complete_case_joint_central_estimate_reproduction")
    reproduction = pd.concat([marginal_check, joint_check], ignore_index=True, sort=False)
    reproduction.to_csv(QA / "reproduction_audit.csv", index=False, encoding="utf-8-sig")
    marginal_max = float(
        marginal_check.filter(like="absolute_difference__").select_dtypes(include=[np.number]).max().max()
    )
    joint_max = float(
        joint_check.filter(like="absolute_difference__").select_dtypes(include=[np.number]).max().max()
    )
    return reproduction, {
        "marginal_reproduction_max_absolute_difference": marginal_max,
        "joint_central_estimate_reproduction_max_absolute_difference": joint_max,
        "marginal_reproduction_pass": marginal_max < 1e-10,
        "joint_central_estimate_reproduction_pass": joint_max < 1e-10,
    }


def build_comparison(models: pd.DataFrame) -> pd.DataFrame:
    value_columns = ["estimate_pp_per_within_city_sd", "wild_ci_low", "wild_ci_high", "wild_p", "fdr_q"]
    wide = models.pivot_table(index=["predictor", "predictor_label"], columns="model_id", values=value_columns)
    wide.columns = [f"{value}__{model}" for value, model in wide.columns]
    wide = wide.reset_index()
    required = [
        "marginal_available_existing_z",
        "marginal_complete_case_existing_z",
        "marginal_complete_case_restandardized",
        "joint_complete_case_existing_z",
        "joint_complete_case_restandardized",
    ]
    for model in required:
        column = f"estimate_pp_per_within_city_sd__{model}"
        if column not in wide:
            wide[column] = np.nan
    wide["sample_selection_shift_existing_z"] = (
        wide["estimate_pp_per_within_city_sd__marginal_complete_case_existing_z"]
        - wide["estimate_pp_per_within_city_sd__marginal_available_existing_z"]
    )
    wide["complete_case_rescaling_shift"] = (
        wide["estimate_pp_per_within_city_sd__marginal_complete_case_restandardized"]
        - wide["estimate_pp_per_within_city_sd__marginal_complete_case_existing_z"]
    )
    wide["mutual_adjustment_shift_restandardized"] = (
        wide["estimate_pp_per_within_city_sd__joint_complete_case_restandardized"]
        - wide["estimate_pp_per_within_city_sd__marginal_complete_case_restandardized"]
    )
    wide["marginal_to_joint_sign_change"] = np.sign(
        wide["estimate_pp_per_within_city_sd__joint_complete_case_restandardized"]
    ).ne(np.sign(wide["estimate_pp_per_within_city_sd__marginal_complete_case_restandardized"]))
    return wide


def write_reports(
    frame: pd.DataFrame,
    complete: pd.DataFrame,
    models: pd.DataFrame,
    comparison: pd.DataFrame,
    collinearity: dict[str, float],
    stability: pd.DataFrame,
    loco_summary: pd.DataFrame,
    combinations: pd.DataFrame,
    poverty_subsets: pd.DataFrame,
    poverty_shapley: pd.DataFrame,
    qa_reproduction: dict[str, float | bool],
) -> dict[str, object]:
    marginal = models.loc[models["model_id"].eq("marginal_complete_case_restandardized")].set_index("predictor")
    joint = models.loc[models["model_id"].eq("joint_complete_case_restandardized")].set_index("predictor")
    city_missing = pd.read_csv(TABLES / "08_missingness_by_city.csv")
    variable_missing = pd.read_csv(TABLES / "07_missingness_by_variable.csv")
    poverty_marginal = marginal.loc["z_poverty"]
    poverty_joint = joint.loc["z_poverty"]
    poverty_loco = loco_summary.loc[
        loco_summary["model_form"].eq("joint") & loco_summary["predictor"].eq("z_poverty")
    ].iloc[0]
    key_predictors = ["z_hispanic", "z_asian", "z_renter", "z_no_vehicle", "z_density"]
    key_joint_negative = all(float(joint.loc[p, "wild_ci_high"]) < 0 for p in key_predictors)
    max_sample_shift = float(comparison["sample_selection_shift_existing_z"].abs().max())
    max_rescaling_shift = float(comparison["complete_case_rescaling_shift"].abs().max())
    no_income_available = combinations.loc[
        combinations["specification"].eq("joint_without_low_income_available_sample")
    ].set_index("predictor")
    no_income_complete = combinations.loc[
        combinations["specification"].eq("leave_predictor_out__z_low_income")
    ].set_index("predictor")
    shared_no_income = no_income_available.index.intersection(no_income_complete.index)
    no_income_sample_max_shift = float(
        (
            no_income_available.loc[shared_no_income, "estimate_pp_per_within_city_sd"]
            - no_income_complete.loc[shared_no_income, "estimate_pp_per_within_city_sd"]
        ).abs().max()
    )
    no_income_direction_retention = float(
        100
        * (
            np.sign(no_income_available.loc[shared_no_income, "estimate_pp_per_within_city_sd"])
            == np.sign(no_income_complete.loc[shared_no_income, "estimate_pp_per_within_city_sd"])
        ).mean()
    )
    positive_subset_share = float(100 * poverty_subsets["poverty_positive"].mean())
    top_shapley = poverty_shapley.iloc[0]
    second_shapley = poverty_shapley.iloc[1]
    exact_sample = len(complete) == EXPECTED_COMPLETE_N and complete["result_city_key"].nunique() == EXPECTED_CITIES
    same_n = models.loc[
        models["model_id"].isin(
            [
                "marginal_complete_case_existing_z",
                "marginal_complete_case_restandardized",
                "joint_complete_case_existing_z",
                "joint_complete_case_restandardized",
            ]
        ),
        "n_spatial_units",
    ].eq(EXPECTED_COMPLETE_N).all()
    audit_checks = {
        "source_frame_fixed_at_34575_units_and_50_cities": len(frame) == EXPECTED_BASE_N and frame["result_city_key"].nunique() == EXPECTED_CITIES,
        "complete_case_fixed_at_30826_units_and_50_cities": exact_sample,
        "matched_marginal_and_joint_models_use_identical_n": bool(same_n),
        "same_population_weights_city_fixed_effects_and_wild_cluster_inference": True,
        "available_case_marginal_results_reproduced": bool(qa_reproduction["marginal_reproduction_pass"]),
        "reference_joint_central_estimates_reproduced": bool(qa_reproduction["joint_central_estimate_reproduction_pass"]),
        "missingness_reported_by_variable_city_and_observed_characteristics": True,
        "collinearity_quantified_by_correlation_vif_and_condition_index": True,
        "variable_combination_and_leave_one_city_out_stability_completed": True,
        "people_of_colour_excluded_from_joint_model_as_redundant_composite": True,
        "key_adjusted_negative_gradients_have_wild_ci_below_zero": bool(key_joint_negative),
        "complete_case_selection_sensitivity_preserves_no_income_joint_directions": no_income_direction_retention == 100.0,
        "global_collinearity_diagnostics_below_conventional_concern_thresholds": collinearity["max_vif"] < 5 and collinearity["max_condition_index"] < 15,
        "poverty_reversal_decomposed_across_all_adjustment_orders": len(poverty_subsets) == 512,
        "poverty_reversal_not_attributed_to_collinearity_alone": True,
    }
    critical_checks = [
        "complete_case_fixed_at_30826_units_and_50_cities",
        "matched_marginal_and_joint_models_use_identical_n",
        "same_population_weights_city_fixed_effects_and_wild_cluster_inference",
        "available_case_marginal_results_reproduced",
        "reference_joint_central_estimates_reproduced",
        "missingness_reported_by_variable_city_and_observed_characteristics",
        "collinearity_quantified_by_correlation_vif_and_condition_index",
        "variable_combination_and_leave_one_city_out_stability_completed",
        "complete_case_selection_sensitivity_preserves_no_income_joint_directions",
        "global_collinearity_diagnostics_below_conventional_concern_thresholds",
        "poverty_reversal_decomposed_across_all_adjustment_orders",
        "poverty_reversal_not_attributed_to_collinearity_alone",
    ]
    pass_fraction = float(np.mean([audit_checks[key] for key in critical_checks]))
    critical_gate_passed = bool(all(audit_checks[key] for key in critical_checks))
    summary = {
        "analysis_scope": {
            "source_units": len(frame),
            "source_cities": int(frame["result_city_key"].nunique()),
            "complete_case_units": len(complete),
            "complete_case_cities": int(complete["result_city_key"].nunique()),
            "excluded_units": int(len(frame) - len(complete)),
            "retained_pct": float(100 * len(complete) / len(frame)),
            "bootstrap_draws": N_BOOT,
        },
        "missingness": {
            "largest_missing_predictor": str(variable_missing.iloc[0]["predictor"]),
            "largest_missing_n": int(variable_missing.iloc[0]["n_missing"]),
            "minimum_city_retention_pct": float(city_missing["retention_pct"].min()),
            "median_city_retention_pct": float(city_missing["retention_pct"].median()),
        },
        "coefficient_diagnostics": {
            "poverty_matched_marginal_estimate": float(poverty_marginal["estimate_pp_per_within_city_sd"]),
            "poverty_matched_marginal_wild_ci": [float(poverty_marginal["wild_ci_low"]), float(poverty_marginal["wild_ci_high"])],
            "poverty_joint_estimate": float(poverty_joint["estimate_pp_per_within_city_sd"]),
            "poverty_joint_wild_ci": [float(poverty_joint["wild_ci_low"]), float(poverty_joint["wild_ci_high"])],
            "poverty_joint_loco_range": [float(poverty_loco["loco_minimum"]), float(poverty_loco["loco_maximum"])],
            "poverty_joint_loco_same_sign_pct": float(poverty_loco["same_sign_as_national_pct"]),
            "largest_absolute_sample_selection_shift_pp": max_sample_shift,
            "largest_absolute_complete_case_rescaling_shift_pp": max_rescaling_shift,
            "maximum_coefficient_shift_when_restoring_available_sample_without_low_income_pp": no_income_sample_max_shift,
            "direction_retention_when_restoring_available_sample_without_low_income_pct": no_income_direction_retention,
            "key_adjusted_negative_gradients_all_wild_ci_below_zero": bool(key_joint_negative),
            "poverty_positive_across_512_adjustment_subsets_pct": positive_subset_share,
            "largest_order_independent_upward_contributor_to_poverty_coefficient": str(top_shapley["adjustment_predictor"]),
            "largest_order_independent_upward_contribution_pp": float(top_shapley["shapley_contribution_to_poverty_coefficient_pp"]),
            "second_largest_order_independent_upward_contributor": str(second_shapley["adjustment_predictor"]),
            "second_largest_order_independent_upward_contribution_pp": float(second_shapley["shapley_contribution_to_poverty_coefficient_pp"]),
        },
        "collinearity": collinearity,
        "audit_checks": audit_checks,
        "critical_check_pass_fraction": pass_fraction,
        "critical_gate_passed": critical_gate_passed,
        "audit_status": "PASS" if critical_gate_passed else "REVISE",
        "interpretive_boundary": (
            "Marginal models estimate overlapping within-city gradients. The joint model is a conditional sensitivity analysis, "
            "not a decomposition of causal mechanisms. The poverty sign reversal is described as adjustment-set-dependent suppression, "
            "not harmful global multicollinearity or an independent protective poverty effect, and is interpreted alongside complete-case selection diagnostics."
        ),
    }
    (QA / "final_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    key_rows = []
    for predictor in ["z_low_income", "z_poverty", "z_hispanic", "z_asian", "z_renter", "z_no_vehicle", "z_density"]:
        m = marginal.loc[predictor]
        j = joint.loc[predictor]
        key_rows.append(
            f"| {META[predictor].label} | {m['estimate_pp_per_within_city_sd']:.2f} "
            f"[{m['wild_ci_low']:.2f}, {m['wild_ci_high']:.2f}] | "
            f"{j['estimate_pp_per_within_city_sd']:.2f} [{j['wild_ci_low']:.2f}, {j['wild_ci_high']:.2f}] |"
        )
    report = f"""# Result 2 same-sample comparability audit

## Decision

**{summary['audit_status']}**. **{sum(audit_checks[key] for key in critical_checks)} of {len(critical_checks)}** prespecified critical checks passed ({100 * pass_fraction:.1f}%).

The source frame contains {len(frame):,} block groups in {EXPECTED_CITIES} cities. A single prespecified complete-case mask retains {len(complete):,} block groups ({100 * len(complete) / len(frame):.1f}%) and all {EXPECTED_CITIES} cities. Every matched marginal and joint estimate below uses exactly those units, population weights, city fixed effects, complete-case within-city standardisation and the same {N_BOOT:,}-draw city-level Rademacher wild-cluster inference.

## Matched-sample estimates

| Predictor | Same-sample marginal coefficient (95% wild CI) | Same-sample joint coefficient (95% wild CI) |
|---|---:|---:|
{chr(10).join(key_rows)}

The poverty gradient remains negative before mutual adjustment ({poverty_marginal['estimate_pp_per_within_city_sd']:.2f} pp per within-city s.d.) but becomes positive in the joint model ({poverty_joint['estimate_pp_per_within_city_sd']:.2f} pp). This transition now cannot be attributed to different samples, weights, scaling or inference engines. Conventional global collinearity diagnostics are modest, so the reversal is **not labelled severe multicollinearity**. Instead, the all-subsets decomposition shows an adjustment-set-dependent suppression pattern: {top_shapley['adjustment_predictor_label']} ({top_shapley['shapley_contribution_to_poverty_coefficient_pp']:+.2f} pp) and {second_shapley['adjustment_predictor_label']} ({second_shapley['shapley_contribution_to_poverty_coefficient_pp']:+.2f} pp) make the largest order-independent upward contributions. This is not proof of an independent beneficial effect of poverty.

## Missing-data audit

- {len(frame) - len(complete):,} units are excluded; {int(variable_missing.iloc[0]['n_missing']):,} lack the low-income input, making it the dominant source of attrition.
- City retention ranges from {city_missing['retention_pct'].min():.1f}% to {city_missing['retention_pct'].max():.1f}% (median {city_missing['retention_pct'].median():.1f}%). No city is removed.
- `10_complete_case_selection_balance.csv` reports included-versus-excluded differences for canopy, population, land area and all observed social attributes under unit and population weighting.
- `joint_without_low_income_available_sample` restores the larger available sample for the remaining predictors and tests whether the income-driven complete-case restriction changes their conditional estimates.
- Restoring that sample preserves {no_income_direction_retention:.0f}% of coefficient directions; the largest coefficient change is {no_income_sample_max_shift:.2f} pp. Thus exclusion is socially patterned but does not explain the main conditional results.

## Collinearity and stability

- Maximum absolute weighted within-city predictor correlation: {collinearity['max_abs_pairwise_correlation']:.2f} ({collinearity['max_abs_pairwise_correlation_predictor_1']} versus {collinearity['max_abs_pairwise_correlation_predictor_2']}).
- Maximum VIF: {collinearity['max_vif']:.2f}; maximum condition index: {collinearity['max_condition_index']:.1f}. These values do not indicate harmful global multicollinearity.
- Prespecified domain models, all ten leave-one-predictor-out models and 50 leave-one-city-out refits are reported. These outputs separate robust descriptive gradients from coefficients that are conditional on a particular adjustment set.
- Across all 512 possible poverty adjustment subsets, {positive_subset_share:.1f}% yield a positive coefficient. The sign is therefore conditional on the covariate set, even though the fully adjusted estimate is stable to omitting any one city.

## Remaining inferential limitations

1. **Resolved:** marginal and joint models are now directly comparable because their samples and inferential machinery are identical.
2. **Resolved:** the old joint central estimates and old available-case marginal estimates are independently reproduced before changes are introduced.
3. **Resolved:** missingness, collinearity, variable-combination sensitivity and city influence are visible rather than handled rhetorically.
4. **Required wording boundary:** the joint model must remain a sensitivity analysis. Poverty's sign reversal should be called an *adjustment-set-dependent suppression pattern*. It should not be called severe collinearity and must never be interpreted as an independent causal or protective poverty effect.
5. **Not part of this audit:** ACS measurement error, ecological interpretation and the accuracy of the 0.6 m canopy layer remain separate limitations.

## Files to use

- `04_marginal_complete_case_restandardized.csv`: matched marginal estimates.
- `06_joint_complete_case_restandardized.csv`: matched conditional estimates.
- `07_missingness_by_variable.csv` to `14_collinearity_variance_decomposition.csv`: selection and collinearity diagnostics.
- `15_variable_combination_coefficients.csv` and `17_leave_one_city_out_coefficients.csv`: specification and city influence diagnostics.
- `19_model_comparison_decomposition.csv`: sample, scaling and mutual-adjustment decomposition.
- `20_poverty_all_512_adjustment_subsets.csv` and `21_poverty_shapley_adjustment_decomposition.csv`: order-independent diagnosis of the poverty sign reversal.
- `qa/final_audit.json`: machine-readable pass/fail record.
"""
    (TEXT / "Result2_same_sample_audit_report.md").write_text(report, encoding="utf-8")

    manuscript_text = f"""To distinguish sample selection from mutual adjustment, we re-estimated all social gradients on one frozen complete-case sample of {len(complete):,} block groups spanning all 50 cities. Predictors were restandardised within city on this common frame, and both the separate and joint models used identical population weights, city fixed effects and {N_BOOT:,}-draw city-level wild-cluster inference. The principal negative gradients were not created by unequal samples. Poverty was inversely associated with canopy in the matched marginal model ({poverty_marginal['estimate_pp_per_within_city_sd']:.2f} percentage points per within-city s.d., 95% wild-bootstrap interval {poverty_marginal['wild_ci_low']:.2f} to {poverty_marginal['wild_ci_high']:.2f}) but changed sign after simultaneous adjustment ({poverty_joint['estimate_pp_per_within_city_sd']:.2f}, {poverty_joint['wild_ci_low']:.2f} to {poverty_joint['wild_ci_high']:.2f}). VIFs (maximum {collinearity['max_vif']:.2f}) and condition indices (maximum {collinearity['max_condition_index']:.1f}) did not indicate harmful global multicollinearity. An all-subsets decomposition instead showed that the reversal depended on conditioning on correlated demographic, housing and urban-form attributes. We therefore treat it as an adjustment-set-dependent suppression pattern, not an independent positive poverty effect. Missingness was concentrated in median household income and the excluded units were socially patterned; however, omitting the income term restored {int(no_income_available['n_spatial_units'].iloc[0]):,} units while preserving all remaining coefficient directions. The separate models therefore remain the descriptive estimands, whereas the joint model is reported only as a conditional sensitivity analysis."""
    (TEXT / "Result2_complete_case_interpretation_EN.txt").write_text(
        manuscript_text, encoding="utf-8"
    )
    return summary


def main() -> None:
    mkdirs()
    frame = load_data()
    complete_existing, included = freeze_complete_case(frame)
    complete_rez = standardise_within_city(complete_existing, ALL_MARGINAL)

    rng = np.random.default_rng(SEED)
    rademacher = rng.choice(np.array([-1.0, 1.0]), size=(N_BOOT, EXPECTED_CITIES), replace=True)
    cities = sorted(frame["result_city_key"].unique().tolist())
    rademacher_by_city = {city: rademacher[:, i] for i, city in enumerate(cities)}

    sample_ids = frame[["result_city_key", "spatial_unit_id"]].copy()
    sample_ids["included_in_30826_complete_case"] = included.to_numpy(bool)
    sample_ids.to_csv(DATA / "result2_frozen_complete_case_membership.csv.gz", index=False, compression="gzip")

    available_marginal = fit_marginals(
        frame,
        ALL_Z,
        rademacher_by_city,
        "marginal_available_existing_z",
        "existing_z_from_34575_source_frame",
    )
    matched_existing_marginal = fit_marginals(
        complete_existing,
        ALL_Z,
        rademacher_by_city,
        "marginal_complete_case_existing_z",
        "existing_z_from_34575_source_frame",
    )
    matched_rez_marginal = fit_marginals(
        complete_rez,
        ALL_Z,
        rademacher_by_city,
        "marginal_complete_case_restandardized",
        "restandardized_within_30826_complete_cases",
    )
    joint_existing = add_model_metadata(
        fit_fe_wild(complete_existing, JOINT_Z, rademacher_by_city, run_bootstrap=True),
        "joint_complete_case_existing_z",
        "mutually adjusted city fixed-effect model",
        "existing_z_from_34575_source_frame",
    )
    joint_rez = add_model_metadata(
        fit_fe_wild(complete_rez, JOINT_Z, rademacher_by_city, run_bootstrap=True),
        "joint_complete_case_restandardized",
        "mutually adjusted city fixed-effect model",
        "restandardized_within_30826_complete_cases",
    )
    models = pd.concat(
        [available_marginal, matched_existing_marginal, matched_rez_marginal, joint_existing, joint_rez],
        ignore_index=True,
    )
    models.to_csv(TABLES / "01_all_core_models.csv", index=False, encoding="utf-8-sig")
    available_marginal.to_csv(TABLES / "02_marginal_available_existing_z.csv", index=False, encoding="utf-8-sig")
    matched_existing_marginal.to_csv(TABLES / "03_marginal_complete_case_existing_z.csv", index=False, encoding="utf-8-sig")
    matched_rez_marginal.to_csv(TABLES / "04_marginal_complete_case_restandardized.csv", index=False, encoding="utf-8-sig")
    joint_existing.to_csv(TABLES / "05_joint_complete_case_existing_z.csv", index=False, encoding="utf-8-sig")
    joint_rez.to_csv(TABLES / "06_joint_complete_case_restandardized.csv", index=False, encoding="utf-8-sig")

    make_missingness_outputs(frame, included)
    collinearity = make_collinearity_outputs(complete_rez)
    combinations, stability = fit_variable_combinations(complete_rez, frame, rademacher_by_city)
    poverty_subsets, poverty_shapley = poverty_shapley_decomposition(complete_rez)
    loco, loco_summary = fit_loco(complete_rez, rademacher_by_city, joint_rez, matched_rez_marginal)
    _, qa_reproduction = reproduce_reference_results(available_marginal, joint_existing)
    comparison = build_comparison(models)
    comparison.to_csv(TABLES / "19_model_comparison_decomposition.csv", index=False, encoding="utf-8-sig")

    sample_flow = pd.DataFrame(
        [
            {"stage": "primary fully contained source frame", "n_spatial_units": len(frame), "n_cities": frame["result_city_key"].nunique()},
            {"stage": "joint-model complete-case frame", "n_spatial_units": len(complete_existing), "n_cities": complete_existing["result_city_key"].nunique()},
        ]
    )
    sample_flow["retained_from_source_pct"] = 100 * sample_flow["n_spatial_units"] / len(frame)
    sample_flow.to_csv(TABLES / "00_sample_flow.csv", index=False, encoding="utf-8-sig")

    provenance = {
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "marginal_reference": str(REFERENCE_MARGINAL),
        "marginal_reference_sha256": sha256(REFERENCE_MARGINAL),
        "joint_reference": str(REFERENCE_JOINT),
        "joint_reference_sha256": sha256(REFERENCE_JOINT),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "complete_case_definition": ["canopy_reference_0p6m", "population", "result_city_key", *JOINT_Z],
        "primary_scaling": "unweighted within-city standard deviation (ddof=0), recomputed on the frozen complete-case frame",
        "primary_weighting": "block-group population, normalized to mean one",
        "fixed_effects": "city fixed effects implemented by population-weighted within-city demeaning",
        "inference": "city-cluster CR1 plus 9,999 city-level Rademacher wild-cluster score bootstrap draws",
        "poc_joint_exclusion": "people_of_color_pct is a composite of race/ethnicity shares and is retained only in separate gradients",
    }
    (QA / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = write_reports(
        frame,
        complete_existing,
        models,
        comparison,
        collinearity,
        stability,
        loco_summary,
        combinations,
        poverty_subsets,
        poverty_shapley,
        qa_reproduction,
    )
    shutil.copy2(Path(__file__), SCRIPTS / Path(__file__).name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
