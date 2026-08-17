from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT = analysis_store_root()
RESULT = PROJECT / "Result"
ANALYSIS = RESULT / "Result5" / "versions" / "group_budget_sensitivity"
DIFFERENCE_PATH = ANALYSIS / "tables" / "03_paired_differences_from_uniform.csv"
ABSOLUTE_PATH = ANALYSIS / "tables" / "02_equal_city_group_ratio_intervals.csv"
CENTRAL_PATH = ANALYSIS / "tables" / "01_city_group_ratios_by_budget.csv"
ANALYSIS_QA_PATH = ANALYSIS / "qa" / "group_budget_sensitivity_audit.json"
OUT = (
    RESULT
    / "Supplementary figure and table"
    / "Supplementary Fig. 14"
    / "versions"
    / "group_budget_paired_differences"
)
FIGURES = OUT / "figures"
SOURCE = OUT / "source_data"
QA = OUT / "qa"
METHODS = OUT / "methods"
TEXT = OUT / "text"

BUDGETS = np.asarray([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00])
RULES = ("heat_context_oracle", "social_condition_oracle", "pareto_0.50")
RULE_TITLES = ("Surface-heat first", "Social-context first", r"Equal-weight ($\lambda$ = 0.5)")
GROUPS = (
    "people_of_color_pct",
    "hispanic_or_latino_pct",
    "black_alone_pct",
    "asian_alone_pct",
    "white_alone_pct",
    "age_65_plus_pct",
)
GROUP_LABELS = ("Not White alone", "Hispanic/Latino", "Black", "Asian", "White", "Age 65+")
INK = "#171A18"
LIMIT = 0.40
CMAP = LinearSegmentedColormap.from_list(
    "paired_difference", ["#3D6F8D", "#B8CCD6", "#F4F1EC", "#E8B49D", "#D17652"], N=256
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.4,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.6,
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


def text_overlap_audit(fig: mpl.figure.Figure) -> list[list[object]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for artist in fig.findobj(mpl.text.Text):
        if artist.get_visible() and artist.get_text().strip():
            texts.append((artist.get_text(), artist.get_window_extent(renderer=renderer)))
    overlaps = []
    for index, (left_text, left_box) in enumerate(texts):
        for right_text, right_box in texts[index + 1 :]:
            x_overlap = max(0.0, min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0))
            y_overlap = max(0.0, min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0))
            if x_overlap * y_overlap > 4.0:
                overlaps.append([left_text, right_text, x_overlap * y_overlap])
    return overlaps


def main() -> int:
    setup_style()
    for directory in (FIGURES, SOURCE, QA, METHODS, TEXT):
        directory.mkdir(parents=True, exist_ok=True)

    differences = pd.read_csv(DIFFERENCE_PATH)
    absolute = pd.read_csv(ABSOLUTE_PATH)
    central = pd.read_csv(CENTRAL_PATH)
    analysis_qa = json.loads(ANALYSIS_QA_PATH.read_text(encoding="utf-8"))
    norm = TwoSlopeNorm(vmin=-LIMIT, vcenter=0.0, vmax=LIMIT)

    fig = plt.figure(figsize=(7.20, 3.05), facecolor="white")
    lefts = (0.115, 0.405, 0.695)
    width = 0.255
    bottom = 0.255
    height = 0.605
    matrices: dict[str, np.ndarray] = {}
    support_matrices: dict[str, np.ndarray] = {}
    for panel_index, (left, rule, title) in enumerate(zip(lefts, RULES, RULE_TITLES)):
        local = differences.loc[differences["rule"].eq(rule)]
        matrix = (
            local.pivot(
                index="group",
                columns="budget_fraction",
                values="equal_city_central_difference_from_uniform",
            )
            .reindex(index=GROUPS, columns=BUDGETS)
            .to_numpy(float)
        )
        support = (
            local.pivot(index="group", columns="budget_fraction", values="paired_interval_excludes_zero")
            .reindex(index=GROUPS, columns=BUDGETS)
            .to_numpy(bool)
        )
        matrices[rule] = matrix
        support_matrices[rule] = support
        ax = fig.add_axes((left, bottom, width, height))
        ax.imshow(matrix, cmap=CMAP, norm=norm, interpolation="nearest", aspect="auto")
        ax.set_xticks(np.arange(len(BUDGETS)), labels=["1", "2.5", "5", "10", "20", "30", "50", "100"])
        ax.set_yticks(np.arange(len(GROUPS)), labels=GROUP_LABELS if panel_index == 0 else [""] * len(GROUPS))
        ax.tick_params(length=0, pad=2.0)
        ax.set_xlabel("Allocated share of analytical capacity (%)", labelpad=2.2)
        ax.set_xticks(np.arange(-0.5, len(BUDGETS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(GROUPS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row, column in np.argwhere(support):
            ax.plot(
                column,
                row,
                marker="o",
                ms=3.2,
                mfc="white",
                mec=INK,
                mew=0.55,
                linestyle="none",
                zorder=4,
            )
        ax.add_patch(Rectangle((2.5, -0.5), 1.0, len(GROUPS), fill=False, edgecolor=INK, linewidth=0.75, zorder=5))
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(
            left,
            0.935,
            f"{'abc'[panel_index]}  {title}",
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
        )

    cax = fig.add_axes((0.205, 0.105, 0.455, 0.030))
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=CMAP),
        cax=cax,
        orientation="horizontal",
        ticks=[-0.4, -0.2, 0, 0.2, 0.4],
    )
    colorbar.ax.tick_params(labelsize=5.2, length=2.0, pad=1.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_label("Difference from uniform in allocation/population ratio", fontsize=5.8, labelpad=1.6)
    support_handle = Line2D(
        [0],
        [0],
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor=INK,
        markeredgewidth=0.6,
        linestyle="none",
        label="Paired 95% interval excludes 0",
    )
    fig.legend(
        handles=[support_handle],
        loc="center left",
        bbox_to_anchor=(0.700, 0.120),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
    )

    overlaps = text_overlap_audit(fig)
    outputs = {
        "png": FIGURES / "Supplementary_Fig14.png",
        "pdf": FIGURES / "Supplementary_Fig14.pdf",
        "svg": FIGURES / "Supplementary_Fig14.svg",
        "tif": FIGURES / "Supplementary_Fig14.tif",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["png"], dpi=450, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["tif"], dpi=450, bbox_inches="tight", pad_inches=0.035, pil_kwargs={"compression": "tiff_lzw"})
    array = np.asarray(Image.open(outputs["png"]).convert("RGB"))
    Image.fromarray(array).convert("L").save(QA / "Supplementary_Fig14_grayscale.png")
    Image.fromarray(array).resize(
        (max(1, array.shape[1] * 300 // 450), max(1, array.shape[0] * 300 // 450)),
        Image.Resampling.LANCZOS,
    ).save(QA / "Supplementary_Fig14_300dpi_preview.png")
    minimum_font = min(
        float(text.get_fontsize())
        for text in fig.findobj(mpl.text.Text)
        if text.get_visible() and text.get_text().strip()
    )
    plt.close(fig)

    differences.to_csv(SOURCE / "Supplementary_Fig14_paired_differences_from_uniform.csv", index=False)
    absolute.to_csv(SOURCE / "Supplementary_Fig14_absolute_group_ratios.csv", index=False)
    central.to_csv(
        SOURCE / "Supplementary_Fig14_city_group_ratios_by_budget.csv.gz",
        index=False,
        compression="gzip",
    )
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)

    values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    supports = np.concatenate([matrix.ravel() for matrix in support_matrices.values()])
    qa = {
        "status": "PASS",
        "quality_gate_status": "PASS",
        "analysis_units": int(analysis_qa["rows"]),
        "cities": int(analysis_qa["cities"]),
        "budgets": len(BUDGETS),
        "rules_compared_with_uniform": len(RULES),
        "groups": len(GROUPS),
        "city_spatial_block_replicates": int(analysis_qa["city_spatial_block_replicates"]),
        "two_level_draws": int(analysis_qa["two_level_draws"]),
        "matrix_cells": int(values.size),
        "paired_interval_supported_cells": int(supports.sum()),
        "maximum_absolute_central_difference": float(np.max(np.abs(values))),
        "colour_limit": LIMIT,
        "full_budget_max_absolute_difference": float(
            np.max(
                np.abs(
                    differences.loc[
                        np.isclose(differences["budget_fraction"], 1.0),
                        "equal_city_central_difference_from_uniform",
                    ]
                )
            )
        ),
        "minimum_font_pt": minimum_font,
        "text_text_overlap_count": len(overlaps),
        "text_text_overlaps": overlaps,
        "exports_exist": all(path.exists() and path.stat().st_size > 10_000 for path in outputs.values()),
        "scientific_claim": (
            "Relative to uniform allocation, demographic distribution shifts are largest at small analytical budgets, differ by objective and converge to zero when all positive capacity is allocated."
        ),
        "principal_limitation": (
            "Allocation/population ratios describe modelled distribution of canopy increments, not equity, thermal benefit or health outcome; the caption states this boundary and uses paired differences only."
        ),
        "remaining_limitations": [
            "Race and ethnicity groups overlap and are interpreted separately.",
            "Race and ethnicity are not ranking inputs, whereas age 65+ contributes to the social-condition ordering.",
            "Open circles reflect spatial-block and city resampling uncertainty in allocation ratios, not uncertainty in future planting feasibility or canopy growth.",
        ],
    }
    qa["status"] = "PASS" if all(
        [
            qa["analysis_units"] == 34566,
            qa["cities"] == 50,
            qa["matrix_cells"] == 3 * 6 * 8,
            qa["city_spatial_block_replicates"] == 500,
            qa["two_level_draws"] == 5000,
            qa["maximum_absolute_central_difference"] <= LIMIT,
            qa["full_budget_max_absolute_difference"] < 1e-8,
            qa["minimum_font_pt"] >= 5.2,
            qa["text_text_overlap_count"] == 0,
            qa["exports_exist"],
        ]
    ) else "FAIL"
    (QA / "Supplementary_Fig14_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ten = differences.loc[np.isclose(differences["budget_fraction"], 0.10)].set_index(["rule", "group"])
    caption = (
        "Supplementary Fig. 14 | Objective-specific demographic allocation differences are largest at small budgets and disappear at full allocation. "
        "a-c, Equal-city difference between the surface-heat-first, social-context-first and illustrative equal-weight rules and uniform allocation in each group's allocation-share-to-population-share ratio across eight city-specific budgets, 50 cities and the fixed 34,566-block-group frame. Colours show deterministic full-data paired differences; open circles mark cells whose paired 95% two-level bootstrap interval excludes zero after 500 within-city spatial-block replicates and 5,000 city-level draws. The outlined 10% column corresponds to the primary budget in Fig. 5. At 10%, the surface-heat-first rule shifted the ratio by "
        f"{ten.loc[('heat_context_oracle', 'hispanic_or_latino_pct'), 'equal_city_central_difference_from_uniform']:+.2f} for Hispanic/Latino residents "
        f"(95% interval {ten.loc[('heat_context_oracle', 'hispanic_or_latino_pct'), 'ci_low']:+.2f} to {ten.loc[('heat_context_oracle', 'hispanic_or_latino_pct'), 'ci_high']:+.2f}) and "
        f"{ten.loc[('heat_context_oracle', 'age_65_plus_pct'), 'equal_city_central_difference_from_uniform']:+.2f} for residents aged 65+ "
        f"({ten.loc[('heat_context_oracle', 'age_65_plus_pct'), 'ci_low']:+.2f} to {ten.loc[('heat_context_oracle', 'age_65_plus_pct'), 'ci_high']:+.2f}); the social-context-first rule shifted the Black-resident ratio by "
        f"{ten.loc[('social_condition_oracle', 'black_alone_pct'), 'equal_city_central_difference_from_uniform']:+.2f} "
        f"({ten.loc[('social_condition_oracle', 'black_alone_pct'), 'ci_low']:+.2f} to {ten.loc[('social_condition_oracle', 'black_alone_pct'), 'ci_high']:+.2f}). Race and ethnicity were not ranking inputs, whereas age 65+ contributes to the social-condition ordering; the groups are not mutually exclusive. Ratios describe the distribution of model-implied allocation within analytical capacity, not equity achieved, observed cooling, personal heat exposure or health benefit."
    )
    (TEXT / "Supplementary_Fig14_caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
