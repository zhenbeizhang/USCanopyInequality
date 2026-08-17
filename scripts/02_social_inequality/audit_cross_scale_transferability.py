from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
import statsmodels.api as sm


BASE = Path(str(analysis_store_root().joinpath('Result', 'Result2', 'Result2_city_slope_transferability_audit')))
PREDICTORS = ("hispanic", "renter", "no_vehicle", "median_age")


def independent_reml(beta: np.ndarray, se: np.ndarray) -> tuple[float, float]:
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    valid = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    beta = beta[valid]
    se = se[valid]

    def negative_restricted_log_likelihood(log_tau: float) -> float:
        tau2 = float(np.exp(2.0 * log_tau))
        variance = se**2 + tau2
        precision = 1.0 / variance
        mean = float(np.dot(precision, beta) / precision.sum())
        return 0.5 * (
            np.log(variance).sum()
            + math.log(precision.sum())
            + np.sum((beta - mean) ** 2 / variance)
        )

    fitted = optimize.minimize_scalar(
        negative_restricted_log_likelihood,
        bounds=(-10.0, 6.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    tau = float(np.exp(fitted.x))
    variance = se**2 + tau**2
    precision = 1.0 / variance
    mean = float(np.dot(precision, beta) / precision.sum())
    return mean, tau


def fit_statsmodels(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    geography = str(frame["geography"].iloc[0])
    for predictor in PREDICTORS:
        for city_key, sub in frame.groupby("city_key", sort=True):
            y = sub["canopy_pct"].to_numpy(float)
            z = sub[f"z_{predictor}"].to_numpy(float)
            weights = sub["population"].to_numpy(float)
            fit = sm.WLS(
                y,
                sm.add_constant(z, has_constant="add"),
                weights=weights,
            ).fit(cov_type="HC3")
            rows.append(
                {
                    "geography": geography,
                    "predictor": predictor,
                    "city_key": city_key,
                    "independent_slope": float(fit.params[1]),
                    "independent_hc3_se": float(fit.bse[1]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    analytic_city = pd.read_csv(BASE / "tables/01_city_social_slopes_analytic.csv")
    analytic_national = pd.read_csv(BASE / "tables/02_national_social_reml_analytic.csv")
    bootstrap_national = pd.read_csv(
        BASE / "tables/03_national_social_spatial_bootstrap.csv"
    )
    bootstrap_city = pd.read_csv(
        BASE / "tables/04_city_social_spatial_bootstrap_intervals.csv"
    )
    national_draws = pd.read_csv(
        BASE / "diagnostics/national_social_bootstrap_draws.csv.gz"
    )
    city_draws = pd.read_csv(
        BASE / "diagnostics/city_social_bootstrap_draws.csv.gz"
    )

    independent_frames = []
    for geography, filename in (
        ("block_group", "data/frozen_block_group_common4_frame.csv.gz"),
        ("tract", "data/frozen_tract_common4_frame.csv.gz"),
    ):
        frame = pd.read_csv(BASE / filename, dtype={"unit_id": str}, low_memory=False)
        assert frame["geography"].eq(geography).all()
        independent_frames.append(fit_statsmodels(frame))
    independent = pd.concat(independent_frames, ignore_index=True)
    comparison = analytic_city.merge(
        independent,
        on=["geography", "predictor", "city_key"],
        how="inner",
        validate="one_to_one",
    )
    comparison["absolute_slope_difference"] = (
        comparison["raw_slope_pp_per_within_city_sd"]
        - comparison["independent_slope"]
    ).abs()
    comparison["absolute_se_difference"] = (
        comparison["raw_hc3_se"] - comparison["independent_hc3_se"]
    ).abs()

    reml_rows = []
    for (geography, predictor), sub in comparison.groupby(
        ["geography", "predictor"], sort=True
    ):
        mean, tau = independent_reml(
            sub["independent_slope"].to_numpy(float),
            sub["independent_hc3_se"].to_numpy(float),
        )
        reference = analytic_national.loc[
            analytic_national["geography"].eq(geography)
            & analytic_national["predictor"].eq(predictor)
        ].iloc[0]
        reml_rows.append(
            {
                "geography": geography,
                "predictor": predictor,
                "independent_reml_mu": mean,
                "reference_reml_mu": float(
                    reference["reml_mu_pp_per_within_city_sd"]
                ),
                "absolute_mu_difference": abs(
                    mean - float(reference["reml_mu_pp_per_within_city_sd"])
                ),
                "independent_tau": tau,
                "reference_tau": float(reference["tau_between_city"]),
                "absolute_tau_difference": abs(
                    tau - float(reference["tau_between_city"])
                ),
            }
        )
    reml = pd.DataFrame(reml_rows)

    national_quantile_rows = []
    for keys, sub in national_draws.groupby(
        ["geography", "predictor", "block_scale"], sort=True
    ):
        geography, predictor, block_scale = keys
        low, median, high = np.percentile(
            sub["two_level_reml_mu"].to_numpy(float), [2.5, 50, 97.5]
        )
        reference = bootstrap_national.loc[
            bootstrap_national["geography"].eq(geography)
            & bootstrap_national["predictor"].eq(predictor)
            & bootstrap_national["block_scale"].eq(block_scale)
        ].iloc[0]
        national_quantile_rows.append(
            {
                "geography": geography,
                "predictor": predictor,
                "block_scale": block_scale,
                "max_absolute_quantile_difference": max(
                    abs(low - float(reference["two_level_reml_ci_low"])),
                    abs(median - float(reference["two_level_reml_median"])),
                    abs(high - float(reference["two_level_reml_ci_high"])),
                ),
            }
        )
    national_quantiles = pd.DataFrame(national_quantile_rows)

    city_quantile_rows = []
    for keys, sub in city_draws.groupby(
        ["geography", "predictor", "block_scale", "city_key"], sort=True
    ):
        geography, predictor, block_scale, city_key = keys
        low, median, high = np.percentile(
            sub["partially_pooled_city_slope"].to_numpy(float), [2.5, 50, 97.5]
        )
        reference = bootstrap_city.loc[
            bootstrap_city["geography"].eq(geography)
            & bootstrap_city["predictor"].eq(predictor)
            & bootstrap_city["block_scale"].eq(block_scale)
            & bootstrap_city["city_key"].eq(city_key)
        ].iloc[0]
        city_quantile_rows.append(
            {
                "geography": geography,
                "predictor": predictor,
                "block_scale": block_scale,
                "city_key": city_key,
                "max_absolute_quantile_difference": max(
                    abs(low - float(reference["partially_pooled_bootstrap_ci_low"])),
                    abs(
                        median
                        - float(reference["partially_pooled_bootstrap_median"])
                    ),
                    abs(high - float(reference["partially_pooled_bootstrap_ci_high"])),
                ),
            }
        )
    city_quantiles = pd.DataFrame(city_quantile_rows)

    checks = {
        "independent_statsmodels_compares_all_400_city_slopes": len(comparison) == 400,
        "maximum_absolute_city_slope_difference_below_1e_10": float(
            comparison["absolute_slope_difference"].max()
        )
        < 1e-10,
        "maximum_absolute_hc3_se_difference_below_1e_10": float(
            comparison["absolute_se_difference"].max()
        )
        < 1e-10,
        "independent_reml_mu_difference_below_2e_6": float(
            reml["absolute_mu_difference"].max()
        )
        < 2e-6,
        "independent_reml_tau_difference_below_2e_6": float(
            reml["absolute_tau_difference"].max()
        )
        < 2e-6,
        "saved_national_bootstrap_quantiles_reproduce": float(
            national_quantiles["max_absolute_quantile_difference"].max()
        )
        < 1e-12,
        "saved_city_bootstrap_quantiles_reproduce": float(
            city_quantiles["max_absolute_quantile_difference"].max()
        )
        < 1e-12,
    }
    print(
        json.dumps(
            {
                "max_city_slope_difference": float(
                    comparison["absolute_slope_difference"].max()
                ),
                "max_hc3_se_difference": float(
                    comparison["absolute_se_difference"].max()
                ),
                "max_reml_mu_difference": float(
                    reml["absolute_mu_difference"].max()
                ),
                "max_reml_tau_difference": float(
                    reml["absolute_tau_difference"].max()
                ),
                "max_national_quantile_difference": float(
                    national_quantiles["max_absolute_quantile_difference"].max()
                ),
                "max_city_quantile_difference": float(
                    city_quantiles["max_absolute_quantile_difference"].max()
                ),
            },
            indent=2,
        )
    )
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"Independent audit failed: {failed}")

    comparison.to_csv(
        BASE / "qa/independent_statsmodels_city_slope_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reml.to_csv(
        BASE / "qa/independent_reml_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    output = {
        "implementation": (
            "Independent statsmodels WLS-HC3 city fits, independently coded "
            "restricted-likelihood optimizer, and direct quantiles from saved draws."
        ),
        "checks": checks,
        "maximum_absolute_city_slope_difference": float(
            comparison["absolute_slope_difference"].max()
        ),
        "maximum_absolute_hc3_se_difference": float(
            comparison["absolute_se_difference"].max()
        ),
        "maximum_absolute_reml_mu_difference": float(
            reml["absolute_mu_difference"].max()
        ),
        "maximum_absolute_reml_tau_difference": float(
            reml["absolute_tau_difference"].max()
        ),
        "maximum_national_quantile_difference": float(
            national_quantiles["max_absolute_quantile_difference"].max()
        ),
        "maximum_city_quantile_difference": float(
            city_quantiles["max_absolute_quantile_difference"].max()
        ),
        "pass": True,
    }
    (BASE / "qa/independent_implementation_check.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
