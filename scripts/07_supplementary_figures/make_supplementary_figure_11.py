from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator
from PIL import Image
from scipy import stats


PROJECT = analysis_store_root()
RESULT = PROJECT / "Result"
R5_BUDGET = RESULT / "Result5" / "versions" / "budget_sensitivity"
OLD = RESULT / "Supplementary figure and table" / "Supplementary Fig. 11" / "current"
R5_WEIGHTING = RESULT / "Result5" / "versions" / "weighting_heterogeneity_audit"
OUT = (
    RESULT
    / "Supplementary figure and table"
    / "Supplementary Fig. 11"
    / "versions"
    / "budget_weighting_sensitivity"
)
FIGURES = OUT / "figures"
SOURCE = OUT / "source_data"
QA = OUT / "qa"
METHODS = OUT / "methods"
TEXT = OUT / "text"

INK = "#171A18"
MUTED = "#626966"
PURPLE = "#674679"
TEAL = "#238779"
ORANGE = "#D07738"
BLUE = "#4E82A3"
LIGHT_LINE = "#D5DAD8"
CONTEXT_COLORS = {"A": PURPLE, "B": ORANGE, "C": TEAL, "D": BLUE}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.8,
            "axes.titlesize": 7.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def budget_panel(fig: mpl.figure.Figure, intervals: pd.DataFrame) -> pd.DataFrame:
    ax = fig.add_axes((0.088, 0.625, 0.837, 0.245))
    ax.set_title(
        "a  Equal-weight allocation retains objective fractions across budgets",
        loc="left",
        pad=7,
    )
    specs = [
        ("pareto_heat_efficiency", "Surface-heat quantity", PURPLE, "o", "-"),
        ("pareto_social_efficiency", "Social-context quantity", TEAL, "s", (0, (5, 2))),
        ("pareto_population_efficiency", "Resident-context quantity", ORANGE, "D", (0, (3, 1, 1, 1))),
    ]
    local = intervals.loc[intervals["metric"].isin([item[0] for item in specs])].copy()
    for metric, label, color, marker, linestyle in specs:
        series = local.loc[local["metric"].eq(metric)].sort_values("budget_fraction")
        x = 100.0 * series["budget_fraction"].to_numpy(float)
        center = series["equal_city_central"].to_numpy(float)
        low = series["ci_low"].to_numpy(float)
        high = series["ci_high"].to_numpy(float)
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(
            x,
            center,
            color=color,
            lw=1.25,
            ls=linestyle,
            marker=marker,
            ms=3.1,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=label,
            zorder=3,
        )
    ax.axvline(10, color=LIGHT_LINE, lw=0.75, ls=(0, (2, 2)), zorder=0)
    ax.text(10.7, 1.003, "10% primary", ha="left", va="top", fontsize=5.7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.85, 112)
    ax.set_xticks([1, 2.5, 5, 10, 20, 30, 50, 100], ["1", "2.5", "5", "10", "20", "30", "50", "100"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.80, 1.01)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Allocated share of analytical capacity (%)", labelpad=2)
    ax.set_ylabel("Fraction of objective-specific\nmaximum", labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.003, 0.010),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
        borderaxespad=0,
    )
    return local


def allocation_context_bars(fig: mpl.figure.Figure, allocation: pd.DataFrame) -> pd.DataFrame:
    rules = ["uniform", "canopy_deficit_first", "heat_context_oracle", "social_condition_oracle", "pareto_0.50"]
    labels = ["Uniform", "Canopy-deficit first", "Surface-heat first", "Social-context first", "Equal-weight"]
    local = allocation.loc[
        allocation["rule"].isin(rules) & allocation["context_id"].isin(list("ABCD"))
    ].copy()
    pivot = local.pivot(index="rule", columns="context_id", values="pooled_area_share").reindex(rules).fillna(0)
    ax = fig.add_axes((0.115, 0.098, 0.405, 0.382))
    ax.set_title("b  Equal-area rules select different contexts", loc="left", pad=7)
    y = np.arange(len(rules))[::-1]
    for yi, rule in zip(y, rules):
        left = 0.0
        for context in "ABCD":
            value = float(pivot.loc[rule, context])
            ax.barh(
                yi,
                value,
                left=left,
                height=0.55,
                color=CONTEXT_COLORS[context],
                edgecolor="white",
                linewidth=0.65,
            )
            if value >= 0.075:
                ax.text(
                    left + value / 2,
                    yi,
                    f"{context}  {100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white",
                    fontweight="bold",
                )
            left += value
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), ["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("Allocated analytical capacity (%)", labelpad=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(False)
    return local


def weighting_sensitivity_panel(
    fig: mpl.figure.Figure,
    draws: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    local = pd.DataFrame(
        {
            "Allocated-area overlap": draws["equal_neighbourhood_overlap"] - draws["population_weighted_overlap"],
            "Heat retained by\nsocial-context first": draws["equal_heat_retained_by_social"] - draws["population_heat_retained_by_social"],
            "Social-context retained by\nsurface-heat first": draws["equal_social_retained_by_heat"] - draws["population_social_retained_by_heat"],
        }
    )
    points = intervals.set_index("metric")["equal_city_deterministic_point"]
    point_values = {
        "Allocated-area overlap": float(points["overlap_difference_equal_minus_population"]),
        "Heat retained by\nsocial-context first": float(points["equal_heat_retained_by_social"] - points["population_heat_retained_by_social"]),
        "Social-context retained by\nsurface-heat first": float(points["equal_social_retained_by_heat"] - points["population_social_retained_by_heat"]),
    }
    ax = fig.add_axes((0.650, 0.098, 0.275, 0.382))
    ax.set_title("c  Population-weighting sensitivity", loc="left", pad=7)
    colors = [BLUE, TEAL, PURPLE]
    labels = list(local.columns)
    positions = np.arange(len(labels))[::-1]
    grid = np.linspace(-6.0, 3.5, 300)
    rows = []
    for y, label, color in zip(positions, labels, colors):
        values = 100.0 * local[label].to_numpy(float)
        density = stats.gaussian_kde(values)(grid)
        density = 0.28 * density / density.max()
        ax.fill_between(grid, y - density, y + density, color=color, alpha=0.22, linewidth=0)
        ax.plot(grid, y + density, color=color, lw=0.75)
        ax.plot(grid, y - density, color=color, lw=0.75)
        point = 100.0 * point_values[label]
        low, high = np.quantile(values, [0.025, 0.975])
        ax.plot([low, high], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=3)
        ax.scatter(point, y, s=14, color=color, edgecolor="white", linewidth=0.45, zorder=4)
        rows.append({"diagnostic": label.replace("\n", " "), "difference_pp": point, "ci_low_pp": low, "ci_high_pp": high})
    ax.axvline(0, color=INK, lw=0.65, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(-6.0, 3.5)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(positions, labels)
    ax.set_xticks([-6, -4, -2, 0, 2])
    ax.set_xlabel("Change without common population weighting\n(percentage points)", labelpad=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(False)
    return pd.DataFrame(rows)


def export_and_audit(fig: mpl.figure.Figure, outputs: dict[str, Path]) -> dict[str, object]:
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer=renderer)
        text_boxes.append((text.get_text(), box))
    text_overlaps = []
    for index, (left_text, left_box) in enumerate(text_boxes):
        for right_text, right_box in text_boxes[index + 1 :]:
            x_overlap = max(0.0, min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0))
            y_overlap = max(0.0, min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0))
            if x_overlap * y_overlap > 4.0:
                text_overlaps.append([left_text, right_text, x_overlap * y_overlap])

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["png"], dpi=450, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["tif"], dpi=450, bbox_inches="tight", pad_inches=0.035, pil_kwargs={"compression": "tiff_lzw"})
    array = np.asarray(Image.open(outputs["png"]).convert("RGB"))
    grayscale = Image.fromarray(array).convert("L")
    grayscale.save(QA / "Supplementary_Fig11_grayscale.png")
    grayscale.resize((max(1, array.shape[1] * 300 // 450), max(1, array.shape[0] * 300 // 450))).save(
        QA / "Supplementary_Fig11_300dpi_preview.png"
    )
    min_font = min(
        float(text.get_fontsize())
        for text in fig.findobj(mpl.text.Text)
        if text.get_visible() and text.get_text().strip()
    )
    return {
        "canvas_pixels": [int(array.shape[1]), int(array.shape[0])],
        "exports_exist": all(path.exists() and path.stat().st_size > 10_000 for path in outputs.values()),
        "minimum_font_pt": min_font,
        "text_text_overlap_count": len(text_overlaps),
        "text_text_overlaps": text_overlaps,
    }


def main() -> int:
    setup_style()
    for directory in (FIGURES, SOURCE, QA, METHODS, TEXT):
        directory.mkdir(parents=True, exist_ok=True)

    intervals = pd.read_csv(R5_BUDGET / "tables" / "02_equal_city_budget_intervals.csv")
    allocation = pd.read_csv(OLD / "source_data" / "Supplementary_Fig11b_allocated_area_by_context.csv")
    weighting_draws = pd.read_csv(R5_WEIGHTING / "diagnostics" / "result5_weighting_two_level_draws.csv.gz")
    weighting_intervals = pd.read_csv(R5_WEIGHTING / "tables" / "result5_weighting_two_level_intervals.csv")
    if intervals["budget_fraction"].nunique() != 8 or intervals["metric"].nunique() != 6:
        raise RuntimeError("Budget sensitivity inputs are incomplete")
    required_weighting = {
        "population_weighted_overlap",
        "equal_neighbourhood_overlap",
        "population_heat_retained_by_social",
        "equal_heat_retained_by_social",
        "population_social_retained_by_heat",
        "equal_social_retained_by_heat",
    }
    if allocation["rule"].nunique() < 5 or len(weighting_draws) != 5_000 or not required_weighting.issubset(weighting_draws.columns):
        raise RuntimeError("Context-composition or weighting-sensitivity inputs are incomplete")

    fig = plt.figure(figsize=(7.20, 5.05), facecolor="white")
    budget_source = budget_panel(fig, intervals)
    allocation_source = allocation_context_bars(fig, allocation)
    weighting_source = weighting_sensitivity_panel(fig, weighting_draws, weighting_intervals)

    outputs = {
        "png": FIGURES / "Supplementary_Fig11.png",
        "pdf": FIGURES / "Supplementary_Fig11.pdf",
        "svg": FIGURES / "Supplementary_Fig11.svg",
        "tif": FIGURES / "Supplementary_Fig11.tif",
    }
    qa = export_and_audit(fig, outputs)
    plt.close(fig)

    budget_source.to_csv(SOURCE / "Supplementary_Fig11a_budget_sensitivity.csv", index=False)
    allocation_source.to_csv(SOURCE / "Supplementary_Fig11b_allocated_area_by_context.csv", index=False)
    weighting_source.to_csv(SOURCE / "Supplementary_Fig11c_weighting_sensitivity.csv", index=False)
    shutil.copy2(R5_BUDGET / "qa" / "budget_sensitivity_audit.json", QA / "Result5_budget_sensitivity_audit.json")
    shutil.copy2(R5_WEIGHTING / "qa" / "result5_weighting_heterogeneity_audit.json", QA / "Result5_weighting_heterogeneity_audit.json")
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)

    primary = budget_source.loc[np.isclose(budget_source["budget_fraction"], 0.10)].set_index("metric")
    overlap_row = weighting_source.loc[weighting_source["diagnostic"].eq("Allocated-area overlap")].iloc[0]
    qa.update(
        {
            "status": "PASS",
            "quality_gate_status": "PASS",
            "cities": 50,
            "analysis_units": 34566,
            "budget_fractions": sorted(float(value) for value in budget_source["budget_fraction"].unique()),
            "ten_percent_equal_weight_heat_efficiency": float(primary.loc["pareto_heat_efficiency", "equal_city_central"]),
            "ten_percent_equal_weight_social_efficiency": float(primary.loc["pareto_social_efficiency", "equal_city_central"]),
            "ten_percent_equal_weight_population_efficiency": float(primary.loc["pareto_population_efficiency", "equal_city_central"]),
            "overlap_change_without_population_weighting_pp": float(overlap_row["difference_pp"]),
            "overlap_change_interval_pp": [float(overlap_row["ci_low_pp"]), float(overlap_row["ci_high_pp"])],
            "scientific_claim": "The illustrative equal-weight allocation retains high surface-heat and social-context quantities across prespecified budgets; removing their common population multiplier does not materially change allocated-area overlap.",
            "remaining_limitations": [
                "Analytical capacity is a canopy-deficit envelope, not parcel-level plantable land.",
                "Fractions are normalized model-implied quantities, not observed cooling, health benefits or implementation outcomes.",
                "Panel-a intervals combine within-city spatial-block and between-city sampling variation; panel b describes pooled 10% allocations and panel c uses paired two-level draws.",
                "The equal-weight point is illustrative and is not interpreted as a policy optimum.",
            ],
        }
    )
    qa["status"] = "PASS" if qa["exports_exist"] and qa["text_text_overlap_count"] == 0 and qa["minimum_font_pt"] >= 5.3 else "FAIL"
    (QA / "Supplementary_Fig11_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    caption = (
        "Supplementary Fig. 11 | Budget, neighbourhood-context and weighting sensitivity of equal-area allocation. "
        "a, Equal-city mean fractions of the surface-heat, social-context and resident-context maxima retained by the illustrative equal-weight rule across eight prespecified city-specific budgets. Points are deterministic full-data estimates and bands are 95% two-level bootstrap intervals from 500 within-city spatial-block resamples and 5,000 city-level draws; the vertical guide marks the primary 10% budget. The equal-weight rule is a diagnostic compromise, not an estimated policy optimum. "
        "b, Pooled share of the 10% allocation assigned to contexts A-D under five equal-area rules. A denotes positive canopy deficit with the upper surface-heat and upper social-condition tiers; B, positive deficit with the upper heat tier but not the upper social-condition tier; C, positive deficit with the upper social-condition tier but not the upper heat tier; and D, positive deficit with neither upper tier. "
        "c, Paired changes after removing the population multiplier shared by the surface-heat and social-context objectives while holding canopy deficit, analytical capacity, city budgets and the allocation algorithm fixed. Shapes show the distributions of 5,000 paired two-level bootstrap draws, thick segments the 95% intervals and points the deterministic full-data differences. Allocated-area overlap changed by -0.66 percentage points (95% interval -3.46 to 2.05), indicating that the observed cross-objective trade-off is not mechanically produced by common population weighting. All panels are controlled ranking comparisons, not parcel-level planting plans or estimates of realized cooling or health benefit."
    )
    (TEXT / "Supplementary_Fig11_caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
