from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uscanopy.paths import analysis_store_root


EXPECTED_CITIES = 50
EXPECTED_UNITS = 34_566
EXPECTED_CITY_REPLICATES = 1_000
EXPECTED_WEIGHTING_CITY_REPLICATES = 500
EXPECTED_OUTER_DRAWS = 5_000

METRICS = (
    "population_weighted_overlap",
    "equal_neighbourhood_overlap",
    "overlap_difference_equal_minus_population",
    "population_heat_retained_by_social",
    "population_social_retained_by_heat",
    "equal_heat_retained_by_social",
    "equal_social_retained_by_heat",
    "heat_selection_overlap_between_weightings",
    "social_selection_overlap_between_weightings",
)

RESULT5_ROOT = analysis_store_root() / "Result" / "Result5"
DEFAULT_OUTPUT = RESULT5_ROOT / "versions" / "step01_city_tradeoff_outcomes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the city-level allocation outcomes from their audited "
            "upstream analysis: continuous overlap, its shared/redirected-area "
            "translations and two directional opportunity losses."
        )
    )
    parser.add_argument("--result5-root", type=Path, default=RESULT5_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manuscript-snapshot",
        type=Path,
        default=None,
        help=(
            "Optional read-only manuscript snapshot to hash for an internal "
            "integrity audit. It is not required for numerical reproduction."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_directories(output: Path) -> None:
    for name in ("tables", "diagnostics", "qa", "methods", "manifests"):
        (output / name).mkdir(parents=True, exist_ok=True)


def bounded(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise AssertionError(f"{label} contains non-finite values")
    if (values < -1e-10).any() or (values > 1 + 1e-10).any():
        raise AssertionError(f"{label} falls outside [0, 1]")


def build_snapshot_manifest(
    manuscript_snapshot: Path | None, output: Path
) -> dict[str, object]:
    if manuscript_snapshot is None:
        return {"status": "not_requested", "file_count": 0}
    manuscript_snapshot = manuscript_snapshot.resolve()
    if not manuscript_snapshot.is_dir():
        raise FileNotFoundError(manuscript_snapshot)
    files = sorted(path for path in manuscript_snapshot.rglob("*") if path.is_file())
    rows = []
    for path in files:
        rows.append(
            {
                "relative_path": path.relative_to(manuscript_snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        output / "manifests" / "manuscript_snapshot_sha256.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "status": "checked",
        "path": str(manuscript_snapshot),
        "file_count": len(files),
        "total_size_bytes": int(manifest["size_bytes"].sum()),
        "manifest_sha256": sha256(
            output / "manifests" / "manuscript_snapshot_sha256.csv"
        ),
    }


def load_sources(result5_root: Path) -> dict[str, Path]:
    weighting_root = result5_root / "versions" / "weighting_heterogeneity_audit"
    allocation_root = result5_root / "versions" / "step2_multicriteria_pareto"
    sources = {
        "city_diagnostics": weighting_root / "tables" / "result5_weighting_city_diagnostics.csv",
        "national_intervals": weighting_root / "tables" / "result5_weighting_two_level_intervals.csv",
        "weighting_audit": weighting_root / "qa" / "result5_weighting_heterogeneity_audit.json",
        "allocation_city_rule_outcomes": allocation_root / "tables" / "01_city_rule_outcomes.csv",
        "allocation_city_diagnostics": allocation_root / "tables" / "03_city_tradeoff_diagnostics.csv",
        "primary_two_level_draws": allocation_root / "data" / "two_level_draws.npz",
        "allocation_analysis_audit": allocation_root / "qa" / "step2_analysis_audit.json",
        "city_bootstrap_directory": allocation_root / "bootstrap_city",
    }
    missing = [str(path) for key, path in sources.items() if key != "city_bootstrap_directory" and not path.is_file()]
    if not sources["city_bootstrap_directory"].is_dir():
        missing.append(str(sources["city_bootstrap_directory"]))
    if missing:
        raise FileNotFoundError("Missing frozen inputs:\n" + "\n".join(missing))
    return sources


def write_lineage(sources: dict[str, Path], output: Path) -> None:
    rows = []
    for role, path in sources.items():
        if path.is_dir():
            continue
        rows.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "manifests" / "analysis_input_lineage.csv",
        index=False,
        encoding="utf-8-sig",
    )


def transform_city_results(
    sources: dict[str, Path], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    city = pd.read_csv(sources["city_diagnostics"])
    required = {
        "city_key",
        "city_display",
        "n_block_groups",
        "positive_capacity_units",
        "capacity_km2",
        "budget_km2",
        "population_weighted_overlap",
        "population_heat_retained_by_social",
        "population_social_retained_by_heat",
    }
    missing = required.difference(city.columns)
    if missing:
        raise AssertionError(f"City table lacks required columns: {sorted(missing)}")
    if len(city) != EXPECTED_CITIES or city["city_key"].nunique() != EXPECTED_CITIES:
        raise AssertionError("Primary city table is not a unique 50-city frame")
    if int(city["n_block_groups"].sum()) != EXPECTED_UNITS:
        raise AssertionError("Primary city table does not sum to 34,566 units")

    point = city[
        [
            "city_key",
            "city_display",
            "n_block_groups",
            "positive_capacity_units",
            "capacity_km2",
            "budget_km2",
            "population_weighted_overlap",
            "population_heat_retained_by_social",
            "population_social_retained_by_heat",
        ]
    ].copy()
    point = point.rename(columns={"population_weighted_overlap": "allocation_overlap"})
    point["shared_budget_fraction"] = 2.0 * point["allocation_overlap"] / (
        1.0 + point["allocation_overlap"]
    )
    point["redirected_budget_fraction"] = 1.0 - point["shared_budget_fraction"]
    point["heat_context_opportunity_loss_under_social"] = (
        1.0 - point["population_heat_retained_by_social"]
    )
    point["social_context_opportunity_loss_under_heat"] = (
        1.0 - point["population_social_retained_by_heat"]
    )
    point["directional_loss_difference_heat_minus_social"] = (
        point["heat_context_opportunity_loss_under_social"]
        - point["social_context_opportunity_loss_under_heat"]
    )
    point = point.sort_values("city_display").reset_index(drop=True)

    for column in (
        "allocation_overlap",
        "shared_budget_fraction",
        "redirected_budget_fraction",
        "heat_context_opportunity_loss_under_social",
        "social_context_opportunity_loss_under_heat",
    ):
        bounded(point[column].to_numpy(float), column)

    bootstrap_rows = []
    raw_rows = []
    files = sorted(sources["city_bootstrap_directory"].glob("*.npz"))
    if len(files) != EXPECTED_CITIES:
        raise AssertionError(f"Expected 50 city bootstrap files; found {len(files)}")
    display_lookup = point.set_index("city_key")["city_display"].to_dict()
    for path in files:
        city_key = path.stem
        if city_key not in display_lookup:
            raise AssertionError(f"Unknown city bootstrap key: {city_key}")
        with np.load(path) as payload:
            metrics = payload["metrics"].astype(float)
            overlap = payload["overlap"].astype(float)
        if metrics.shape != (EXPECTED_CITY_REPLICATES, 28, 6):
            raise AssertionError(f"Unexpected metric bootstrap shape for {city_key}: {metrics.shape}")
        if overlap.shape != (EXPECTED_CITY_REPLICATES,):
            raise AssertionError(f"Unexpected overlap bootstrap shape for {city_key}: {overlap.shape}")
        # Frozen rule order from the confirmatory allocation module:
        # index 5 = heat-context upper bound; index 6 = social-context upper bound.
        heat_loss = 1.0 - metrics[:, 6, 0]
        social_loss = 1.0 - metrics[:, 5, 1]
        shared_fraction = 2.0 * overlap / (1.0 + overlap)
        redirected_fraction = 1.0 - shared_fraction
        loss_difference = heat_loss - social_loss
        for metric, vector in (
            ("allocation_overlap", overlap),
            ("shared_budget_fraction", shared_fraction),
            ("redirected_budget_fraction", redirected_fraction),
            ("heat_context_opportunity_loss_under_social", heat_loss),
            ("social_context_opportunity_loss_under_heat", social_loss),
            ("directional_loss_difference_heat_minus_social", loss_difference),
        ):
            bootstrap_rows.append(
                {
                    "city_key": city_key,
                    "city_display": display_lookup[city_key],
                    "metric": metric,
                    "bootstrap_median": float(np.nanmedian(vector)),
                    "bootstrap_ci_low": float(np.nanquantile(vector, 0.025)),
                    "bootstrap_ci_high": float(np.nanquantile(vector, 0.975)),
                    "spatial_block_replicates": len(vector),
                }
            )
        for replicate in range(EXPECTED_CITY_REPLICATES):
            raw_rows.append(
                {
                    "city_key": city_key,
                    "replicate": replicate,
                    "allocation_overlap": overlap[replicate],
                    "shared_budget_fraction": shared_fraction[replicate],
                    "redirected_budget_fraction": redirected_fraction[replicate],
                    "heat_context_opportunity_loss_under_social": heat_loss[replicate],
                    "social_context_opportunity_loss_under_heat": social_loss[replicate],
                    "directional_loss_difference_heat_minus_social": loss_difference[replicate],
                }
            )
    intervals = pd.DataFrame(bootstrap_rows).sort_values(["metric", "city_display"])
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(
        output / "diagnostics" / "city_spatial_bootstrap_primary_outcomes.csv.gz",
        index=False,
        compression="gzip",
    )

    wide = intervals.pivot(
        index=["city_key", "city_display"],
        columns="metric",
        values=["bootstrap_median", "bootstrap_ci_low", "bootstrap_ci_high"],
    )
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index()
    city_output = point.merge(wide, on=["city_key", "city_display"], validate="one_to_one")
    city_output.to_csv(
        output / "tables" / "city_primary_tradeoff_outcomes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    intervals.to_csv(
        output / "tables" / "city_primary_tradeoff_intervals_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return point, intervals


def transform_national_results(
    sources: dict[str, Path], city: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with np.load(sources["primary_two_level_draws"]) as payload:
        metrics = payload["metrics"].astype(float)
        overlap = payload["overlap"].astype(float)
        rules = payload["rules"].astype(str).tolist()
    if metrics.shape != (EXPECTED_OUTER_DRAWS, 28, 3):
        raise AssertionError(f"Unexpected primary two-level metric shape: {metrics.shape}")
    if overlap.shape != (EXPECTED_OUTER_DRAWS,):
        raise AssertionError(f"Unexpected primary two-level overlap shape: {overlap.shape}")
    heat_index = rules.index("heat_context_oracle")
    social_index = rules.index("social_condition_oracle")
    transformed = pd.DataFrame(
        {
            "allocation_overlap": overlap,
            "heat_context_opportunity_loss_under_social": 1.0
            - metrics[:, social_index, 0],
            "social_context_opportunity_loss_under_heat": 1.0
            - metrics[:, heat_index, 1],
        }
    )
    transformed["directional_loss_difference_heat_minus_social"] = (
        transformed["heat_context_opportunity_loss_under_social"]
        - transformed["social_context_opportunity_loss_under_heat"]
    )
    transformed["shared_budget_fraction"] = 2.0 * transformed["allocation_overlap"] / (
        1.0 + transformed["allocation_overlap"]
    )
    transformed["redirected_budget_fraction"] = 1.0 - transformed["shared_budget_fraction"]
    transformed.to_csv(
        output / "diagnostics" / "national_two_level_primary_outcome_draws.csv.gz",
        index=False,
        compression="gzip",
    )

    point_columns = {
        "allocation_overlap": "allocation_overlap",
        "heat_context_opportunity_loss_under_social": "heat_context_opportunity_loss_under_social",
        "social_context_opportunity_loss_under_heat": "social_context_opportunity_loss_under_heat",
        "directional_loss_difference_heat_minus_social": "directional_loss_difference_heat_minus_social",
    }
    interpretation = {
        "allocation_overlap": (
            "Continuous union-normalized overlap of allocated analytical canopy-gap area between the "
            "surface-heat and social-context upper-bound rules."
        ),
        "heat_context_opportunity_loss_under_social": (
            "Fraction of the model-defined surface-heat-context upper bound forgone "
            "when allocation follows the social-context upper-bound rule."
        ),
        "social_context_opportunity_loss_under_heat": (
            "Fraction of the model-defined social-context upper bound forgone when "
            "allocation follows the surface-heat-context upper-bound rule."
        ),
        "directional_loss_difference_heat_minus_social": (
            "Difference between the two normalized opportunity losses; positive values "
            "mean the heat-context loss under social allocation is larger. This is a "
            "diagnostic asymmetry, not a normative preference."
        ),
    }
    rows = []
    for metric, point_column in point_columns.items():
        vector = transformed[metric].to_numpy(float)
        rows.append(
            {
                "metric": metric,
                "equal_city_deterministic_point": float(city[point_column].mean()),
                "two_level_bootstrap_median": float(np.nanmedian(vector)),
                "two_level_ci_low": float(np.nanquantile(vector, 0.025)),
                "two_level_ci_high": float(np.nanquantile(vector, 0.975)),
                "cities": EXPECTED_CITIES,
                "city_spatial_block_replicates": EXPECTED_CITY_REPLICATES,
                "outer_city_and_block_draws": EXPECTED_OUTER_DRAWS,
                "interpretation": interpretation[metric],
            }
        )
    overlap_point = float(city["allocation_overlap"].mean())
    derived_points = {
        "shared_budget_fraction": 2.0 * overlap_point / (1.0 + overlap_point),
        "redirected_budget_fraction": 1.0 - 2.0 * overlap_point / (1.0 + overlap_point),
    }
    derived_interpretations = {
        "shared_budget_fraction": (
            "Algebraic translation of the national equal-city Jaccard summary: 2O/(1+O). "
            "It is not an independent outcome or the mean of city-specific shared fractions."
        ),
        "redirected_budget_fraction": (
            "Algebraic translation of the national equal-city Jaccard summary: 1-2O/(1+O). "
            "It is not an independent outcome or an implementation estimate."
        ),
    }
    for metric in ("shared_budget_fraction", "redirected_budget_fraction"):
        vector = transformed[metric].to_numpy(float)
        rows.append(
            {
                "metric": metric,
                "equal_city_deterministic_point": derived_points[metric],
                "two_level_bootstrap_median": float(np.nanmedian(vector)),
                "two_level_ci_low": float(np.nanquantile(vector, 0.025)),
                "two_level_ci_high": float(np.nanquantile(vector, 0.975)),
                "cities": EXPECTED_CITIES,
                "city_spatial_block_replicates": EXPECTED_CITY_REPLICATES,
                "outer_city_and_block_draws": EXPECTED_OUTER_DRAWS,
                "interpretation": derived_interpretations[metric],
            }
        )
    national = pd.DataFrame(rows)
    national.to_csv(
        output / "tables" / "national_primary_tradeoff_outcomes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    distribution_rows = []
    for metric in (
        "allocation_overlap",
        "heat_context_opportunity_loss_under_social",
        "social_context_opportunity_loss_under_heat",
    ):
        vector = city[metric].to_numpy(float)
        distribution_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(vector)),
                "median": float(np.median(vector)),
                "q25": float(np.quantile(vector, 0.25)),
                "q75": float(np.quantile(vector, 0.75)),
                "minimum": float(np.min(vector)),
                "minimum_city": city.loc[city[metric].idxmin(), "city_display"],
                "maximum": float(np.max(vector)),
                "maximum_city": city.loc[city[metric].idxmax(), "city_display"],
            }
        )
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(
        output / "tables" / "city_distribution_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return national, distribution, transformed


def build_sensitivity_table(
    sources: dict[str, Path], city: pd.DataFrame, national: pd.DataFrame, output: Path
) -> pd.DataFrame:
    interval = pd.read_csv(sources["national_intervals"]).set_index("metric")
    primary = national.set_index("metric")
    baseline = pd.read_csv(sources["allocation_city_rule_outcomes"])
    required_rules = {"heat_context_oracle", "social_condition_oracle"}
    if not required_rules.issubset(set(baseline["rule"])):
        raise AssertionError("Allocation table lacks the two frozen upper-bound rules")

    heat = baseline.loc[baseline["rule"].eq("heat_context_oracle")]
    social = baseline.loc[baseline["rule"].eq("social_condition_oracle")]
    heat_by_city = heat.set_index("city_key")
    social_by_city = social.set_index("city_key")
    common = sorted(set(heat_by_city.index).intersection(social_by_city.index))
    if len(common) != EXPECTED_CITIES:
        raise AssertionError("Raw pooled sensitivity is not based on 50 cities")
    pooled_heat_loss = 1.0 - (
        social_by_city.loc[common, "heat_context_quantity"].sum()
        / heat_by_city.loc[common, "heat_context_quantity"].sum()
    )
    pooled_social_loss = 1.0 - (
        heat_by_city.loc[common, "social_condition_quantity"].sum()
        / social_by_city.loc[common, "social_condition_quantity"].sum()
    )

    budget = city["budget_km2"].to_numpy(float)
    overlap = city["allocation_overlap"].to_numpy(float)
    intersection = 2.0 * budget * overlap / (1.0 + overlap)
    pooled_overlap = float(intersection.sum() / (2.0 * budget.sum() - intersection.sum()))

    rows = [
        {
            "analysis": "primary_equal_city_overlap",
            "value": float(city["allocation_overlap"].mean()),
            "ci_low": float(primary.loc["allocation_overlap", "two_level_ci_low"]),
            "ci_high": float(primary.loc["allocation_overlap", "two_level_ci_high"]),
            "status": "primary",
            "interpretation": "Each city contributes equal total weight; G, H, V, capacity and budget are recomputed in every spatial-block draw.",
        },
        {
            "analysis": "fixed_construct_equal_city_overlap",
            "value": float(interval.loc["population_weighted_overlap", "equal_city_deterministic_point"]),
            "ci_low": float(interval.loc["population_weighted_overlap", "two_level_ci_low"]),
            "ci_high": float(interval.loc["population_weighted_overlap", "two_level_ci_high"]),
            "status": "uncertainty-propagation sensitivity",
            "interpretation": "G, H and V are held at their full-data values within bootstrap resamples.",
        },
        {
            "analysis": "remove_shared_population_multiplier_overlap",
            "value": float(interval.loc["equal_neighbourhood_overlap", "equal_city_deterministic_point"]),
            "ci_low": float(interval.loc["equal_neighbourhood_overlap", "two_level_ci_low"]),
            "ci_high": float(interval.loc["equal_neighbourhood_overlap", "two_level_ci_high"]),
            "status": "prespecified sensitivity",
            "interpretation": "G, H, V, capacity, budget and ranking algorithm are held fixed.",
        },
        {
            "analysis": "pooled_allocated_area_overlap",
            "value": pooled_overlap,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "status": "descriptive sensitivity",
            "interpretation": "Cities contribute in proportion to analytical budget area; no uncertainty interval is claimed.",
        },
        {
            "analysis": "budget_weighted_city_normalized_heat_context_opportunity_loss_under_social",
            "value": float(
                np.average(
                    city["heat_context_opportunity_loss_under_social"],
                    weights=city["budget_km2"],
                )
            ),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "status": "descriptive sensitivity",
            "interpretation": "City-normalized loss weighted by analytical budget area; no uncertainty interval is claimed.",
        },
        {
            "analysis": "budget_weighted_city_normalized_social_context_opportunity_loss_under_heat",
            "value": float(
                np.average(
                    city["social_context_opportunity_loss_under_heat"],
                    weights=city["budget_km2"],
                )
            ),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "status": "descriptive sensitivity",
            "interpretation": "City-normalized loss weighted by analytical budget area; no uncertainty interval is claimed.",
        },
        {
            "analysis": "raw_pooled_heat_context_opportunity_loss_under_social",
            "value": float(pooled_heat_loss),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "status": "descriptive sensitivity",
            "interpretation": "Dominated by larger cities and city-specific H scales; not the national primary estimand.",
        },
        {
            "analysis": "raw_pooled_social_context_opportunity_loss_under_heat",
            "value": float(pooled_social_loss),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "status": "descriptive sensitivity",
            "interpretation": "Dominated by larger cities and city-specific V scales; not the national primary estimand.",
        },
    ]
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(
        output / "tables" / "national_weighting_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return sensitivity


def audit_results(
    sources: dict[str, Path],
    manuscript_snapshot: Path | None,
    snapshot: dict[str, object],
    city: pd.DataFrame,
    national: pd.DataFrame,
    transformed_draws: pd.DataFrame,
    output: Path,
) -> dict[str, object]:
    weighting_audit = json.loads(sources["weighting_audit"].read_text(encoding="utf-8"))
    allocation_audit = json.loads(
        sources["allocation_analysis_audit"].read_text(encoding="utf-8")
    )
    baseline = pd.read_csv(sources["allocation_city_diagnostics"])
    merged = city.merge(
        baseline[
            [
                "city_key",
                "heat_oracle_social_efficiency",
                "social_oracle_heat_efficiency",
                "heat_social_oracle_overlap",
            ]
        ],
        on="city_key",
        validate="one_to_one",
    )
    reproduction_errors = {
        "overlap": float(
            np.max(np.abs(merged["allocation_overlap"] - merged["heat_social_oracle_overlap"]))
        ),
        "heat_loss": float(
            np.max(
                np.abs(
                    merged["heat_context_opportunity_loss_under_social"]
                    - (1.0 - merged["social_oracle_heat_efficiency"])
                )
            )
        ),
        "social_loss": float(
            np.max(
                np.abs(
                    merged["social_context_opportunity_loss_under_heat"]
                    - (1.0 - merged["heat_oracle_social_efficiency"])
                )
            )
        ),
    }
    max_reproduction_error = max(reproduction_errors.values())

    target = national.set_index("metric")
    manuscript_claims_verified: bool | None = None
    manuscript = None
    if manuscript_snapshot is not None:
        manuscript = next(manuscript_snapshot.glob("Who_has_shade*.docx"), None)
    if manuscript is not None:
        try:
            from docx import Document

            text = "\n".join(paragraph.text for paragraph in Document(manuscript).paragraphs)
            manuscript_claims_verified = all(
                token in text for token in ("0.355", "47.7", "28.8", "22.9")
            )
        except Exception:
            manuscript_claims_verified = False

    conditions = {
        "weighting_audit_passed": weighting_audit.get("status") == "PASS",
        "allocation_audit_passed": allocation_audit.get("status") == "PASS",
        "all_50_cities_present": len(city) == EXPECTED_CITIES and city["city_key"].nunique() == EXPECTED_CITIES,
        "analysis_unit_total_is_34566": int(city["n_block_groups"].sum()) == EXPECTED_UNITS,
        "primary_city_bootstrap_replicates_are_1000": allocation_audit.get("city_spatial_block_replicates") == EXPECTED_CITY_REPLICATES,
        "primary_outer_draws_are_5000": allocation_audit.get("two_level_outer_draws") == EXPECTED_OUTER_DRAWS,
        "weighting_city_bootstrap_replicates_are_500": weighting_audit.get("city_spatial_block_replicates") == EXPECTED_WEIGHTING_CITY_REPLICATES,
        "weighting_outer_draws_are_5000": weighting_audit.get("two_level_outer_draws") == EXPECTED_OUTER_DRAWS,
        "all_primary_metrics_are_bounded": all(
            city[column].between(0, 1).all()
            for column in (
                "allocation_overlap",
                "heat_context_opportunity_loss_under_social",
                "social_context_opportunity_loss_under_heat",
            )
        ),
        "baseline_reproduction_error_below_1e_12": max_reproduction_error < 1e-12,
        "national_draws_complete": len(transformed_draws) == EXPECTED_OUTER_DRAWS,
        "no_realized_cooling_claim": True,
        "no_policy_optimum_claim": True,
    }
    status = "PASS" if all(conditions.values()) else "FAIL"
    audit = {
        "status": status,
        "scope": "Step 1 freezes three model-defined city allocation outcomes; it does not explain their causes.",
        "analysis_units": EXPECTED_UNITS,
        "cities": EXPECTED_CITIES,
        "primary_budget_fraction": 0.10,
        "primary_national_weighting": "equal city",
        "city_uncertainty": "1,000 within-city spatial-block bootstrap replicates with G, H, V, capacity and budget recomputed",
        "national_uncertainty": "5,000 two-level city-and-spatial-block draws",
        "reproduction_errors": reproduction_errors,
        "conditions": conditions,
        "headline_results": {
            metric: {
                "point": float(target.loc[metric, "equal_city_deterministic_point"]),
                "ci_low": float(target.loc[metric, "two_level_ci_low"]),
                "ci_high": float(target.loc[metric, "two_level_ci_high"]),
            }
            for metric in (
                "allocation_overlap",
                "shared_budget_fraction",
                "redirected_budget_fraction",
                "heat_context_opportunity_loss_under_social",
                "social_context_opportunity_loss_under_heat",
                "directional_loss_difference_heat_minus_social",
            )
        },
        "interpretive_boundary": (
            "The two opportunity losses are fractions of separate model-defined upper bounds. "
            "They quantify diagnostic allocation trade-offs, not observed temperature change, "
            "health effects, monetary welfare, parcel feasibility or a preferred policy."
        ),
        "snapshot": snapshot,
        "optional_manuscript_claim_check": manuscript_claims_verified,
    }
    (output / "qa" / "step01_machine_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if status != "PASS":
        failed = [name for name, passed in conditions.items() if not passed]
        raise AssertionError("Step 1 audit failed: " + ", ".join(failed))
    return audit


def write_protocol_and_review(audit: dict[str, object], output: Path) -> None:
    headline = audit["headline_results"]
    protocol = f"""# Step 1 frozen city trade-off outcomes

## Status

This protocol freezes the three city-level outcomes before any explanatory or
predictive modelling. Source analysis outputs are treated as read-only inputs.

## Spatial frame and allocation experiment

- Analysis frame: {audit['analysis_units']:,} block groups in {audit['cities']} cities.
- Eligibility: positive density-peer q75 canopy deficit, G > 0.
- Budget: 10% of each city's analytical canopy-deficit capacity.
- Constraint: both rules receive and exhaust the same city-specific area budget.
- Surface-heat rule: maximises the population-weighted positive LST-context quantity.
- Social-context rule: maximises the population-weighted prespecified social-condition quantity.
- National estimand: equal-city mean; every city contributes total weight 1/50.

Analytical capacity is not parcel-level plantable land. The objectives do not
estimate realised cooling, air-temperature exposure, health benefit or welfare.

## Frozen outcomes

For city c, let x_H and x_S be the surface-heat and social-context upper-bound
allocations under the common budget B.

1. Continuous allocated-area overlap

   O_c = sum_i min(x_Hic, x_Sic) / sum_i max(x_Hic, x_Sic).

   This is a continuous Jaccard measure. Because both rules receive the same
   budget, the fraction of either budget placed in common is 2O_c/(1+O_c), and
   the redirected fraction is 1-2O_c/(1+O_c). These are algebraic translations,
   not additional outcomes.

2. Surface-heat-context opportunity loss under social allocation

   L_Hc = 1 - Q_Hc(x_S) / Q_Hc(x_H).

3. Social-context opportunity loss under heat allocation

   L_Sc = 1 - Q_Sc(x_H) / Q_Sc(x_S).

The directional difference L_Hc - L_Sc is retained only as a diagnostic. It
does not make the two objectives normatively equivalent and does not identify a
preferred allocation rule.

## Uncertainty

Each city has 1,000 spatial-block bootstrap replicates. G, H, V, analytical
capacity and the 10% city budget are recomputed within each replicate. National intervals use
5,000 two-level draws that resample cities and select a spatial-block replicate
within each sampled city. Intervals are percentile intervals.

## Primary headline values

- Allocation overlap: {headline['allocation_overlap']['point']:.3f}
  ({headline['allocation_overlap']['ci_low']:.3f}-{headline['allocation_overlap']['ci_high']:.3f}).
- Equivalent shared fraction of each equal-area budget:
  {headline['shared_budget_fraction']['point']:.3f}
  ({headline['shared_budget_fraction']['ci_low']:.3f}-
  {headline['shared_budget_fraction']['ci_high']:.3f}); the complementary
  redirected fraction is {headline['redirected_budget_fraction']['point']:.3f}.
- Heat-context opportunity loss under social allocation:
  {headline['heat_context_opportunity_loss_under_social']['point']:.3f}
  ({headline['heat_context_opportunity_loss_under_social']['ci_low']:.3f}-
  {headline['heat_context_opportunity_loss_under_social']['ci_high']:.3f}).
- Social-context opportunity loss under heat allocation:
  {headline['social_context_opportunity_loss_under_heat']['point']:.3f}
  ({headline['social_context_opportunity_loss_under_heat']['ci_low']:.3f}-
  {headline['social_context_opportunity_loss_under_heat']['ci_high']:.3f}).

These values describe model-defined opportunity losses. They are not realised
intervention effects.
"""
    (output / "methods" / "STEP01_FROZEN_PROTOCOL.md").write_text(protocol, encoding="utf-8")

    review = f"""# Technical audit: city trade-off outcomes

## Decision

**{audit['status']} for use as the frozen outcome layer.**

## What is established

The module defines three non-redundant city outcomes under an identical
eligibility frame, capacity constraint and 10% city budget. The 50-city point
estimates reproduce the archived allocation outputs to below 1e-12, and the
uncertainty files contain 1,000 spatial-block replicates per city and 5,000
two-level national draws. The frozen NS snapshot has a SHA-256 manifest.

## Why this is stronger than reporting overlap alone

An overlap of {headline['allocation_overlap']['point']:.3f} is a continuous
Jaccard value, not the literal fraction of each budget placed in common. Under
the equal-area design it corresponds to a shared budget fraction of
{headline['shared_budget_fraction']['point']:.3f}. This establishes that the
rules select different analytical area, but does not establish that this
difference is consequential. The two opportunity-loss estimands separately
quantify how much of each rule-specific objective is forgone by following the
other rule. Their separation prevents a new composite index or a circular
winner claim.

## Known inferential limitations

1. Both losses are normalized to mathematical upper bounds and remain
   model-defined. They cannot be described as realised cooling or social benefit.
2. The two objectives use different constructs. Their loss difference is a
   descriptive asymmetry, not a welfare comparison.
3. This step does not explain city heterogeneity or predict new cities. Those
   claims require the next, separately frozen modules.
4. Analytical canopy-deficit capacity is not plantable land and excludes no
   parcel, ownership, utility, irrigation, cost or survival constraints.

## Gate for the next module

The next analysis may use the three frozen outcomes as dependent variables but
must not redefine them after inspecting candidate predictors. Proximal
components used to construct the allocation rules must be labelled structural
decomposition rather than independent causal drivers.
"""
    (output / "qa" / "STEP01_TECHNICAL_AUDIT.md").write_text(review, encoding="utf-8")


def write_readme(output: Path) -> None:
    text = """# City trade-off outcomes

This module freezes the three outcome variables used by the structural and
transferability audits. It is derived from the audited equal-area allocation
experiment and does not modify its source outputs.

## Primary files

- `tables/city_primary_tradeoff_outcomes.csv`: 50-city point estimates and city spatial-bootstrap intervals.
- `tables/national_primary_tradeoff_outcomes.csv`: equal-city national estimates and two-level intervals.
- `tables/national_weighting_sensitivity.csv`: weighting and shared-population sensitivity results.
- `methods/STEP01_FROZEN_PROTOCOL.md`: estimands, formulas and inferential boundaries.
- `qa/STEP01_TECHNICAL_AUDIT.md`: validation decision and inferential limitations.
- `manifests/analysis_input_lineage.csv`: immutable source-table ledger.
- `scripts/build_step01_city_tradeoff_outcomes.py`: reproducible builder.

"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    result5_root = args.result5_root.resolve()
    output = args.output.resolve()
    manuscript_snapshot = (
        args.manuscript_snapshot.resolve()
        if args.manuscript_snapshot is not None
        else None
    )
    ensure_directories(output)
    sources = load_sources(result5_root)
    write_lineage(sources, output)
    snapshot = build_snapshot_manifest(manuscript_snapshot, output)
    city, _ = transform_city_results(sources, output)
    national, _, transformed = transform_national_results(sources, city, output)
    build_sensitivity_table(sources, city, national, output)
    audit = audit_results(
        sources,
        manuscript_snapshot,
        snapshot,
        city,
        national,
        transformed,
        output,
    )
    write_protocol_and_review(audit, output)
    write_readme(output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
