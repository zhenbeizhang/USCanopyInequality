from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


PROJECT = analysis_store_root()
INPUT = (
    PROJECT
    / "Result"
    / "Result4"
    / "versions"
    / "joint_context_boundary_spatial_audit"
    / "tables"
    / "09_result5_national_boundary_sensitivity.csv"
)
ROOT = (
    PROJECT
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 12"
    / "versions"
    / "boundary_support_matrix"
)
FIG_DIR = ROOT / "figures"
SOURCE_DIR = ROOT / "source_data"
METHOD_DIR = ROOT / "methods"
QA_DIR = ROOT / "qa"
TEXT_DIR = ROOT / "text"
for directory in (FIG_DIR, SOURCE_DIR, METHOD_DIR, QA_DIR, TEXT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

STEM = FIG_DIR / "Supplementary_Fig12"
BLACK = "#171717"
NEGATIVE = "#3D6F8D"
ZERO = "#F4F1EC"
POSITIVE = "#D17652"

SUPPORTS = [
    ("intersection_ge_0p95_block_allocated", "≥95%"),
    ("intersection_ge_0p80_block_allocated", "≥80%"),
    ("any_intersection_block_allocated", "Any"),
]
METRICS = [
    ("heat_context_efficiency", "Surface heat"),
    ("social_condition_efficiency", "Social context"),
    ("population_efficiency", "Resident context"),
]
PRIMARY = "fully_contained_primary"
LAMBDAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.2,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.5,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def parse_lambda(rule: pd.Series) -> pd.Series:
    values = rule.str.extract(r"^pareto_(\d+\.\d+)$", expand=False)
    return pd.to_numeric(values, errors="coerce")


def prepare() -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray, dict[str, float]]:
    data = pd.read_csv(INPUT)
    data["lambda"] = parse_lambda(data["rule"])
    pareto = data.loc[data["lambda"].notna()].copy()
    if pareto["support"].nunique() != 4 or len(pareto) != 4 * len(LAMBDAS):
        raise RuntimeError("Boundary sensitivity table is incomplete")
    if not np.allclose(np.sort(pareto["lambda"].unique()), LAMBDAS):
        raise RuntimeError("Unexpected Pareto lambda grid")

    primary = pareto.loc[pareto["support"].eq(PRIMARY)].set_index("lambda")
    rows: list[dict[str, float | str]] = []
    matrix_rows: list[np.ndarray] = []
    labels: list[str] = []
    row_maxima: list[float] = []
    for support_code, support_label in SUPPORTS:
        local = pareto.loc[pareto["support"].eq(support_code)].set_index("lambda")
        for metric, metric_label in METRICS:
            differences = (
                local.loc[LAMBDAS, metric].to_numpy(float)
                - primary.loc[LAMBDAS, metric].to_numpy(float)
            )
            matrix_rows.append(differences)
            labels.append(f"{support_label} · {metric_label}")
            row_maxima.append(float(np.max(np.abs(differences))))
            for lam, difference in zip(LAMBDAS, differences):
                rows.append(
                    {
                        "support": support_code,
                        "support_label": support_label,
                        "metric": metric,
                        "metric_label": metric_label,
                        "lambda": float(lam),
                        "difference_from_fully_contained": float(difference),
                    }
                )
    matrix = np.vstack(matrix_rows)
    source = pd.DataFrame(rows)
    midpoint_index = int(np.where(np.isclose(LAMBDAS, 0.5))[0][0])
    summary = {
        "maximum_absolute_change": float(np.max(np.abs(matrix))),
        "maximum_absolute_change_at_lambda_0p5": float(np.max(np.abs(matrix[:, midpoint_index]))),
        "maximum_budget_error_m2": float(pd.to_numeric(data["maximum_budget_error_m2"]).abs().max()),
        "minimum_total_budget_km2": float(pd.to_numeric(data["total_budget_km2"]).min()),
        "maximum_total_budget_km2": float(pd.to_numeric(data["total_budget_km2"]).max()),
    }
    return source, matrix, labels, np.asarray(row_maxima), summary


def make_colormap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "boundary_difference", [NEGATIVE, ZERO, POSITIVE], N=256
    )


def make_figure(matrix: np.ndarray, labels: list[str], row_maxima: np.ndarray, summary: dict[str, float]) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.20, 3.34), facecolor="white")
    ax = fig.add_axes([0.140, 0.230, 0.760, 0.700])
    limit = 0.018
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    mesh = ax.pcolormesh(
        np.arange(len(LAMBDAS) + 1),
        np.arange(len(labels) + 1),
        matrix,
        cmap=make_colormap(),
        norm=norm,
        edgecolors="white",
        linewidth=0.45,
        antialiased=True,
        shading="flat",
    )
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, len(LAMBDAS))
    ax.set_ylim(len(labels), 0)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=4.0)

    shown = [0.0, 0.25, 0.5, 0.75, 1.0]
    shown_idx = [int(np.where(np.isclose(LAMBDAS, value))[0][0]) for value in shown]
    ax.set_xticks(np.asarray(shown_idx) + 0.5)
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.tick_params(axis="x", length=2.2, width=0.55, pad=2.0)
    ax.set_xlabel("Weight on surface-heat objective (λ)", labelpad=3.0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y in (3, 6):
        ax.axhline(y, color=BLACK, lw=0.72, zorder=5)
    midpoint_index = int(np.where(np.isclose(LAMBDAS, 0.5))[0][0])
    ax.add_patch(
        Rectangle(
            (midpoint_index, 0),
            1,
            len(labels),
            facecolor="none",
            edgecolor=BLACK,
            linewidth=1.0,
            zorder=6,
            clip_on=False,
        )
    )

    ax.text(
        1.027,
        1.025,
        "max |Δ|",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        fontweight="bold",
        clip_on=False,
    )
    for row, value in enumerate(row_maxima):
        ax.text(
            1.027,
            1 - (row + 0.5) / len(labels),
            f"{value:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.4,
            clip_on=False,
        )

    fig.text(
        0.140,
        0.970,
        "Boundary-support sensitivity of normalized objective fractions",
        ha="left",
        va="top",
        fontsize=6.9,
        fontweight="bold",
        color=BLACK,
    )
    cax = fig.add_axes([0.140, 0.070, 0.760, 0.035])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_ticks([-0.015, 0.0, 0.015])
    cbar.set_ticklabels(["−0.015", "0", "+0.015"])
    cbar.ax.tick_params(labelsize=5.2, length=1.8, width=0.5, pad=1.0)
    cbar.set_label("Change in fraction of objective-specific maximum", fontsize=5.6, labelpad=1.5)
    cbar.outline.set_linewidth(0.5)

    fig.savefig(STEM.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), dpi=600, facecolor="white")
    fig.savefig(
        STEM.with_suffix(".tif"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> int:
    source, matrix, labels, row_maxima, summary = prepare()
    source.to_csv(SOURCE_DIR / "Supplementary_Fig12_boundary_objective_differences.csv", index=False)
    pd.DataFrame([summary]).to_csv(SOURCE_DIR / "Supplementary_Fig12_summary.csv", index=False)
    make_figure(matrix, labels, row_maxima, summary)

    caption = (
        "Supplementary Fig. 12 | Multiobjective allocation quantities are insensitive to prespecified municipal-boundary supports. "
        "Rows compare three alternatives—block groups with at least 95% or 80% of area inside the municipal boundary and all intersecting block groups—with the fully contained primary support. "
        "Columns trace the Pareto path from the social-context objective (λ = 0) to the surface-heat objective (λ = 1). "
        "Colours show the deterministic change in the fraction of the surface-heat, social-context or resident-context maximum after rerunning the same city-specific 10% equal-area allocation within each support; the outlined column marks the illustrative equal-weight rule (λ = 0.5), and right-hand values give the maximum absolute change along each row. "
        f"The maximum absolute difference was {summary['maximum_absolute_change']:.3f} across the full path and {summary['maximum_absolute_change_at_lambda_0p5']:.3f} at λ = 0.5. "
        "These support contrasts are a specification audit rather than sampling-confidence intervals. Values are model-defined allocation quantities, not observed cooling, personal exposure or health benefit."
    )
    (TEXT_DIR / "Supplementary_Fig12_caption.txt").write_text(caption + "\n", encoding="utf-8")

    qa = {
        "status": "PASS",
        "supports_including_reference": 4,
        "alternative_supports": 3,
        "pareto_lambda_values": int(len(LAMBDAS)),
        "objectives": 3,
        "matrix_cells": int(matrix.size),
        "maximum_absolute_change": summary["maximum_absolute_change"],
        "maximum_absolute_change_at_lambda_0p5": summary["maximum_absolute_change_at_lambda_0p5"],
        "maximum_budget_error_m2": summary["maximum_budget_error_m2"],
        "minimum_font_pt": 5.2,
        "old_38_12_framework_used": False,
        "uncertainty_claimed": False,
        "figure_contract": {
            "core_claim": "Equal-area multiobjective allocation conclusions change little under prespecified boundary supports.",
            "nonduplicative_role": "Boundary-support specification audit; Supplementary Fig. 11 instead addresses budget sensitivity.",
            "strongest_objection": "Normalized objective values might appear stable only because the same objective defines the oracle.",
            "response": "The figure does not compare objectives or claim superiority; it tests each objective's transfer across spatial supports relative to its own fixed primary-support result.",
        },
    }
    (QA_DIR / "Supplementary_Fig12_build_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
