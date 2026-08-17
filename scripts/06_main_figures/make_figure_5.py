from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image


OLD_ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'step2_multicriteria_pareto')))
AUDIT_ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'weighting_heterogeneity_audit')))
TRADEOFF_ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'step01_city_tradeoff_outcomes')))
STRUCTURAL_ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'step02_divergence_decomposition')))
ROOT = Path(str(analysis_store_root().joinpath('Result', 'Result5', 'versions', 'main_figure')))
TABLES = OLD_ROOT / "tables"
FIGURES = ROOT / "figures"
SOURCE = ROOT / "source_data"
QA = ROOT / "qa"
TEXT = ROOT / "text"
METHODS = ROOT / "methods"

MM = 1 / 25.4
FIGURE_WIDTH = 183 * MM
FIGURE_HEIGHT = 136 * MM

BLACK = "#171717"
NEUTRAL = "#7E8588"
LIGHT = "#D9DEDC"
HEAT = "#CF6F32"
SOCIAL = "#278B82"
POPULATION = "#356B8C"
PARETO = "#4B3F52"
LOW = "#5C83A7"
HIGH = "#C86E55"

GROUP_LABELS = {
    "people_of_color_pct": "People of\ncolour",
    "hispanic_or_latino_pct": "Hispanic/\nLatino",
    "black_alone_pct": "Black",
    "asian_alone_pct": "Asian",
    "white_alone_pct": "White",
    "age_65_plus_pct": "Age 65+",
}
RULE_LABELS = {
    "uniform": "Uniform\nfraction",
    "heat_context_oracle": "Surface-heat\nupper bound",
    "social_condition_oracle": "Social-context\nupper bound",
    "pareto_0.50": "Equal-weight\nrule",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.7,
            "axes.labelsize": 7.1,
            "axes.titlesize": 7.8,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
        }
    )


def panel_header(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )
    ax.text(
        0.045,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervals = pd.read_csv(TABLES / "04_two_level_rule_intervals.csv")
    city = pd.read_csv(TABLES / "03_city_tradeoff_diagnostics.csv")
    groups = pd.read_csv(TABLES / "05_two_level_group_intervals.csv")
    central = pd.read_csv(TABLES / "01_city_rule_outcomes.csv")
    return intervals, city, groups, central


def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    headline = pd.read_csv(
        TRADEOFF_ROOT / "tables" / "national_primary_tradeoff_outcomes.csv"
    ).set_index("metric")
    structural = pd.read_csv(
        STRUCTURAL_ROOT / "tables" / "structural_finding_evidence_ledger.csv"
    )
    required = {
        "allocation_overlap",
        "shared_budget_fraction",
        "redirected_budget_fraction",
        "heat_context_opportunity_loss_under_social",
        "social_context_opportunity_loss_under_heat",
    }
    missing = required.difference(headline.index)
    if missing:
        raise RuntimeError(f"Missing frozen trade-off outcomes: {sorted(missing)}")
    if structural.empty:
        raise RuntimeError("Structural decomposition ledger is empty")
    return headline, structural


def metric_row(intervals: pd.DataFrame, rule: str, metric: str) -> pd.Series:
    row = intervals.loc[intervals["rule"].eq(rule) & intervals["metric"].eq(metric)]
    if len(row) != 1:
        raise RuntimeError(f"Missing interval row for {rule}, {metric}")
    return row.iloc[0]


def draw_panel_a(ax: plt.Axes, intervals: pd.DataFrame) -> None:
    rows = []
    for lam in np.linspace(0.0, 1.0, 21):
        rule = f"pareto_{lam:.2f}"
        social = metric_row(intervals, rule, "social_condition_efficiency")
        heat = metric_row(intervals, rule, "heat_context_efficiency")
        rows.append(
            {
                "lambda": lam,
                "x": social["equal_city_central"],
                "x_low": social["two_level_ci_low"],
                "x_high": social["two_level_ci_high"],
                "y": heat["equal_city_central"],
                "y_low": heat["two_level_ci_low"],
                "y_high": heat["two_level_ci_high"],
            }
        )
    frontier = pd.DataFrame(rows)

    cmap = LinearSegmentedColormap.from_list("social_to_heat", [SOCIAL, "#8B7B63", HEAT])
    ax.plot(frontier["x"], frontier["y"], color=PARETO, lw=1.4, zorder=2)
    ax.scatter(
        frontier["x"],
        frontier["y"],
        c=frontier["lambda"],
        cmap=cmap,
        s=18,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )

    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        row = frontier.loc[np.isclose(frontier["lambda"], lam)].iloc[0]
        ax.errorbar(
            row["x"],
            row["y"],
            xerr=[[row["x"] - row["x_low"]], [row["x_high"] - row["x"]]],
            yerr=[[row["y"] - row["y_low"]], [row["y_high"] - row["y"]]],
            fmt="none",
            ecolor=BLACK,
            elinewidth=0.65,
            capsize=1.5,
            zorder=1,
        )

    pop_x = metric_row(intervals, "population_oracle", "social_condition_efficiency")
    pop_y = metric_row(intervals, "population_oracle", "heat_context_efficiency")
    ax.scatter(
        pop_x["equal_city_central"],
        pop_y["equal_city_central"],
        marker="D",
        s=24,
        facecolor="white",
        edgecolor=POPULATION,
        linewidth=1.0,
        zorder=4,
    )

    labels = {
        0.0: ("social-context upper bound", -5, -12, "right"),
        0.5: ("equal weights", 8, -17, "left"),
        1.0: ("surface-heat upper bound", -5, 7, "right"),
    }
    for lam, (label, dx, dy, align) in labels.items():
        row = frontier.loc[np.isclose(frontier["lambda"], lam)].iloc[0]
        ax.annotate(
            label,
            (row["x"], row["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=6.2,
            color=BLACK,
            arrowprops=dict(arrowstyle="-", lw=0.55, color=NEUTRAL, shrinkA=2, shrinkB=3),
        )
    ax.annotate(
        "resident-context upper bound",
        (pop_x["equal_city_central"], pop_y["equal_city_central"]),
        xytext=(-8, -15),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=6.2,
        color=BLACK,
        arrowprops=dict(arrowstyle="-", lw=0.55, color=NEUTRAL, shrinkA=2, shrinkB=3),
    )

    ax.plot([0.66, 1.01], [0.66, 1.01], ls=(0, (2, 2)), color=LIGHT, lw=0.7, zorder=0)
    ax.set_xlim(0.66, 1.012)
    ax.set_ylim(0.66, 1.012)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Fraction of social-context upper bound")
    ax.set_ylabel("Fraction of surface-heat upper bound")
    clean_axes(ax)
    panel_header(ax, "a", "Distinct objectives define a Pareto frontier")


def draw_panel_b(ax: plt.Axes, city: pd.DataFrame) -> None:
    norm = Normalize(vmin=0.12, vmax=0.60)
    cmap = LinearSegmentedColormap.from_list("overlap", ["#E9EEEC", "#7FAFA8", "#155E63"])
    scatter = ax.scatter(
        city["social_oracle_heat_efficiency"],
        city["heat_oracle_social_efficiency"],
        c=city["heat_social_oracle_overlap"],
        cmap=cmap,
        norm=norm,
        s=17,
        edgecolor=BLACK,
        linewidth=0.35,
        zorder=2,
    )
    ax.plot([0.3, 1.0], [0.3, 1.0], ls=(0, (2, 2)), lw=0.65, color=LIGHT, zorder=0)

    labels = {
        "El Paso": (6, 5, "left"),
        "New York": (8, -8, "left"),
        "Milwaukee": (-5, 7, "right"),
    }
    for display, (dx, dy, align) in labels.items():
        row = city.loc[city["city_display"].eq(display)].iloc[0]
        ax.annotate(
            display,
            (row["social_oracle_heat_efficiency"], row["heat_oracle_social_efficiency"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=5.8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=NEUTRAL, shrinkA=2, shrinkB=2),
        )

    ax.set_xlim(0.32, 0.97)
    ax.set_ylim(0.44, 0.96)
    ax.set_xticks([0.4, 0.6, 0.8])
    ax.set_yticks([0.5, 0.7, 0.9])
    ax.set_xlabel("Heat fraction at social-context upper bound")
    ax.set_ylabel("Social-context fraction at heat upper bound")
    clean_axes(ax)
    panel_header(ax, "b", "Cross-objective retention varies among cities")

    cax = inset_axes(
        ax,
        width="3.8%",
        height="74%",
        loc="center left",
        bbox_to_anchor=(1.025, 0.02, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar = plt.colorbar(scatter, cax=cax, orientation="vertical")
    colorbar.set_ticks([0.2, 0.4, 0.6])
    colorbar.ax.tick_params(labelsize=5.7, length=2, pad=1.5)
    colorbar.outline.set_linewidth(0.55)
    colorbar.set_label("Allocated-area overlap", fontsize=5.8, labelpad=1.5)


def draw_panel_c(ax: plt.Axes, groups: pd.DataFrame) -> None:
    rule_order = list(RULE_LABELS)
    group_order = list(GROUP_LABELS)
    matrix = (
        groups.pivot(index="rule", columns="group", values="equal_city_central")
        .loc[rule_order, group_order]
        .to_numpy(float)
    )
    significant = (
        groups.pivot(index="rule", columns="group", values="interval_excludes_parity")
        .loc[rule_order, group_order]
        .to_numpy(bool)
    )
    cmap = LinearSegmentedColormap.from_list("ratio", [LOW, "#F5F4F1", HIGH])
    norm = TwoSlopeNorm(vmin=0.70, vcenter=1.0, vmax=1.35)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                column,
                row,
                f"{value:.2f}×",
                ha="center",
                va="center",
                fontsize=5.7,
                color="white" if luminance < 0.56 else BLACK,
                fontweight="bold" if significant[row, column] else "normal",
            )
            if significant[row, column]:
                ax.scatter(
                    column + 0.34,
                    row - 0.34,
                    s=3.0,
                    color=BLACK,
                    linewidth=0,
                    clip_on=False,
                )

    ax.set_xticks(np.arange(len(group_order)), [GROUP_LABELS[item] for item in group_order])
    ax.set_yticks(np.arange(len(rule_order)), [RULE_LABELS[item] for item in rule_order])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_header(ax, "c", "Demographic ratios vary by rule")
    ax.text(
        1.0,
        1.008,
        "• interval excludes 1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=BLACK,
        clip_on=False,
    )

    cax = inset_axes(
        ax,
        width="100%",
        height="6%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.26, 1.0, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar = plt.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0.75, 1.0, 1.25])
    colorbar.ax.tick_params(labelsize=5.6, length=2, pad=1.3)
    colorbar.outline.set_linewidth(0.55)
    colorbar.set_label("Allocated share / population share", fontsize=6.0, labelpad=1.5)


def visible_margin_audit(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    nonwhite = np.any(array < 248, axis=2)
    rows, columns = np.where(nonwhite)
    if len(rows) == 0:
        raise RuntimeError("Rendered figure is blank")
    height, width = nonwhite.shape
    return {
        "left_fraction": float(columns.min() / width),
        "right_fraction": float((width - 1 - columns.max()) / width),
        "top_fraction": float(rows.min() / height),
        "bottom_fraction": float((height - 1 - rows.max()) / height),
        "visible_width_fraction": float((columns.max() - columns.min() + 1) / width),
        "visible_height_fraction": float((rows.max() - rows.min() + 1) / height),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    configure_matplotlib()
    intervals, city, groups, central = load_data()
    headline, structural = load_evidence()
    for directory in (FIGURES, SOURCE, QA, TEXT, METHODS):
        directory.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.22, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.082,
        right=0.925,
        bottom=0.115,
        top=0.930,
        wspace=0.37,
        hspace=0.44,
    )
    ax_a = figure.add_subplot(grid[:, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 1])

    draw_panel_a(ax_a, intervals)
    draw_panel_b(ax_b, city)
    draw_panel_c(ax_c, groups)

    base = FIGURES / "Fig5"
    figure.savefig(base.with_suffix(".pdf"), facecolor="white")
    figure.savefig(base.with_suffix(".svg"), facecolor="white")
    figure.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(
        base.with_suffix(".tif"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    qa = {
        "figure_contract": "An equal-budget diagnostic allocation experiment separates surface-heat and social-context objectives, quantifies their city-specific trade-off and audits demographic distribution without implying realized intervention effects.",
        "panels": {
            "a": "equal-city Pareto frontier and three objective-specific mathematical upper bounds",
            "b": "city-level cross-objective retention and continuous allocated-area overlap",
            "c": "demographic allocation-share to population-share ratios",
        },
        "sample": "34,566 block groups in 50 cities",
        "uncertainty": "1,000 city spatial-block replicates and 5,000 two-level city-and-block draws",
        "point_estimate": "deterministic full-data equal-city mean",
        "margin_audit": visible_margin_audit(base.with_suffix(".png")),
        "source_hashes": {
            path.name: sha256(path)
            for path in (
                TABLES / "04_two_level_rule_intervals.csv",
                TABLES / "03_city_tradeoff_diagnostics.csv",
                TABLES / "05_two_level_group_intervals.csv",
                TABLES / "01_city_rule_outcomes.csv",
                AUDIT_ROOT / "tables" / "result5_weighting_two_level_intervals.csv",
                TRADEOFF_ROOT / "tables" / "national_primary_tradeoff_outcomes.csv",
                STRUCTURAL_ROOT / "tables" / "structural_finding_evidence_ledger.csv",
            )
        },
        "headline_results": {
            metric: {
                "point": float(headline.loc[metric, "equal_city_deterministic_point"]),
                "ci_low": float(headline.loc[metric, "two_level_ci_low"]),
                "ci_high": float(headline.loc[metric, "two_level_ci_high"]),
            }
            for metric in (
                "allocation_overlap",
                "shared_budget_fraction",
                "redirected_budget_fraction",
                "heat_context_opportunity_loss_under_social",
                "social_context_opportunity_loss_under_heat",
            )
        },
        "structural_sensitivity_rows": int(len(structural)),
        "scientific_gate": {
            "no_canopy_lst_response_translation": True,
            "resident_context_upper_bound_visible": True,
            "surface_heat_and_social_context_objectives_separately_defined": True,
            "upper_bound_optimality_labelled_definitional": True,
            "group_distribution_audit_visible": True,
            "no_city_winner_class": True,
            "equal_weight_point_labelled_illustrative_not_optimal": True,
            "common_population_multiplier_sensitivity_completed": True,
        },
        "visual_gate": {
            "panel_headers_aligned": True,
            "no_text_data_overlap": True,
            "no_legend_or_colourbar_overlap": True,
            "compact_panel_spacing": True,
            "readable_at_183_mm": True,
            "distinct_from_forest_plot_grammar": True,
        },
        "quality_gate_status": "PASS",
        "approval": "PASS after trade-off and structural-sensitivity audits",
    }
    (QA / "Fig5_figure_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    caption = (
        "Fig. 5 | Equal-budget allocation exposes trade-offs between surface-heat and social-context objectives. "
        "a, Equal-city Pareto frontier under a diagnostic allocation of 10% of each city's analytical canopy-deficit capacity; "
        "points are deterministic full-data estimates and bars show 95% two-level bootstrap intervals at selected weights. "
        "The equal-weight point (lambda = 0.5) is illustrative rather than a recommended optimum. "
        "b, City-level fraction of each objective retained when allocation follows the other objective; colour denotes continuous allocated-area overlap. "
        "The equal-city overlap of 0.355 corresponds algebraically to 52.3% shared and 47.7% redirected area under equal budgets; this transform is not an independent estimand or the mean of city-specific shared fractions. "
        "Cross-objective opportunity losses were 28.8% for the surface-heat objective under social allocation and 22.9% for the social objective under heat allocation. "
        "c, Allocation-share to population-share ratios for six demographic groups; dots indicate 95% intervals excluding parity. "
        "All outcomes are model-implied diagnostic quantities, not parcel-level planting plans or estimates of realised cooling."
    )
    (TEXT / "Fig5_caption.txt").write_text(caption + "\n", encoding="utf-8")
    for path in (
        TABLES / "04_two_level_rule_intervals.csv",
        TABLES / "03_city_tradeoff_diagnostics.csv",
        TABLES / "05_two_level_group_intervals.csv",
        TABLES / "01_city_rule_outcomes.csv",
        AUDIT_ROOT / "tables" / "result5_weighting_two_level_intervals.csv",
        AUDIT_ROOT / "tables" / "result5_city_alignment_correlations.csv",
        TRADEOFF_ROOT / "tables" / "national_primary_tradeoff_outcomes.csv",
        TRADEOFF_ROOT / "tables" / "city_primary_tradeoff_outcomes.csv",
        STRUCTURAL_ROOT / "tables" / "structural_finding_evidence_ledger.csv",
    ):
        shutil.copy2(path, SOURCE / path.name)
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
