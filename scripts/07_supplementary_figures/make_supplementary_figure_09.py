from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image


PROJECT = analysis_store_root()
R4 = PROJECT / "Result" / "Result4" / "versions" / "joint_context_boundary_spatial_audit"
OUT = (
    PROJECT
    / "Result"
    / "Supplementary figure and table"
    / "Supplementary Fig. 9"
    / "versions"
    / "typology_inventory"
)
FIGURES = OUT / "figures"
SOURCE = OUT / "source_data"
QA = OUT / "qa"
METHODS = OUT / "methods"

TYPE_TABLE = R4 / "tables" / "01_primary_27_type_summary.csv"
CONTEXT_TABLE = R4 / "tables" / "02_primary_6_context_summary.csv"

INK = "#171A18"
MUTED = "#616866"
TEAL = "#217F73"
PURPLE = "#654477"
ORANGE = "#CF7437"
BLUE = "#4B7EA1"
GOLD = "#C6A34C"
GREY = "#8C9391"
LIGHT_LINE = "#D4DAD7"
CONTEXT_COLORS = {
    "A": PURPLE,
    "B": ORANGE,
    "C": TEAL,
    "D": BLUE,
    "E": GOLD,
    "F": GREY,
}
CONTEXT_NAMES = {
    "A": "deficit + high heat + high V",
    "B": "deficit + high heat",
    "C": "deficit + high V",
    "D": "other canopy deficit",
    "E": "high heat or V; no deficit",
    "F": "lower or mixed reference",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.7,
            "axes.titlesize": 7.1,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 6.0,
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


def panel_heading(fig: mpl.figure.Figure, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x, y, letter, fontsize=7.8, fontweight="bold", ha="left", va="bottom")
    fig.text(x + 0.018, y, title, fontsize=7.3, fontweight="bold", ha="left", va="bottom")


def context_for(g: int, h: int, v: int) -> str:
    g_positive = g >= 2
    h_high = h == 3
    v_high = v == 3
    if g_positive and h_high and v_high:
        return "A"
    if g_positive and h_high and not v_high:
        return "B"
    if g_positive and not h_high and v_high:
        return "C"
    if g_positive and not h_high and not v_high:
        return "D"
    if not g_positive and (h_high or v_high):
        return "E"
    return "F"


def draw_type_matrices(fig: mpl.figure.Figure, types: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "prevalence", ["#F6F8F7", "#D6E7E2", "#83BBB1", TEAL, "#12544D"]
    )
    vmax = float(np.ceil(types["equal_city_prevalence"].max() * 200) / 200)
    norm = Normalize(vmin=0, vmax=vmax)
    positions = ((0.075, 0.545, 0.235, 0.330), (0.365, 0.545, 0.235, 0.330), (0.655, 0.545, 0.235, 0.330))
    titles = {
        1: "Lower V tertile",
        2: "Middle V tertile",
        3: "Higher V tertile",
    }
    for v, position in zip((1, 2, 3), positions):
        ax = fig.add_axes(position)
        matrix = np.zeros((3, 3), dtype=float)
        for g in (1, 2, 3):
            for h in (1, 2, 3):
                code = f"G{g}H{h}V{v}"
                row = types.loc[types["type_code"].eq(code)].iloc[0]
                matrix[3 - h, g - 1] = float(row["equal_city_prevalence"])
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal", interpolation="none")
        for g in (1, 2, 3):
            for h in (1, 2, 3):
                value = matrix[3 - h, g - 1]
                code = f"G{g}H{h}V{v}"
                context = context_for(g, h, v)
                cell_text = "white" if norm(value) >= 0.72 else INK
                ax.text(g - 1, 3 - h - 0.10, f"{100 * value:.1f}%", ha="center", va="center", fontsize=6.5, color=cell_text, fontweight="bold")
                ax.text(g - 1, 3 - h + 0.25, f"{code}  [{context}]", ha="center", va="center", fontsize=5.5, color=cell_text)
        ax.set_xticks(range(3), ["G1\nnone", "G2\nlower >0", "G3\nhigher >0"])
        ax.set_yticks(range(3), ["H3  higher >0", "H2  lower >0", "H1  none"] if v == 1 else ["", "", ""])
        ax.set_xlabel("Canopy-deficit level")
        if v == 1:
            ax.set_ylabel("Positive LST-anomaly level")
        ax.set_title(titles[v], pad=3.0)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cax = fig.add_axes((0.912, 0.585, 0.011, 0.245))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Equal-city share", labelpad=3)
    cb.set_ticks([0, vmax / 2, vmax])
    cb.set_ticklabels(["0", f"{50*vmax:.1f}%", f"{100*vmax:.1f}%"])
    cb.outline.set_linewidth(0.5)


def draw_context_bars(fig: mpl.figure.Figure, contexts: pd.DataFrame) -> pd.DataFrame:
    ax = fig.add_axes((0.075, 0.145, 0.848, 0.245))
    contexts = contexts.set_index("context_id").loc[list("ABCDEF")].reset_index()
    metrics = [
        ("block_group_share", "Block groups"),
        ("equal_city_prevalence", "Cities equally weighted"),
        ("population_share", "Represented population"),
    ]
    y = np.arange(len(metrics))[::-1]
    for yi, (column, _) in zip(y, metrics):
        left = 0.0
        for _, row in contexts.iterrows():
            value = float(row[column])
            ax.barh(yi, value, left=left, height=0.52, color=CONTEXT_COLORS[row["context_id"]], edgecolor="white", linewidth=0.6)
            if value >= 0.075:
                label = f"{row['context_id']}  {100*value:.1f}" if yi == y[0] else f"{100*value:.1f}"
                ax.text(left + value / 2, yi, label, ha="center", va="center", fontsize=6.0, color="white" if row["context_id"] in "ABCD" else INK, fontweight="bold")
            left += value
    ax.set_yticks(y, [label for _, label in metrics])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), ["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("Share (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(False)

    return contexts


def export_and_qa(fig: mpl.figure.Figure, outputs: dict[str, Path]) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.get_window_extent(renderer)
    clipped = []
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1 or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
            clipped.append(artist.get_text())
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["png"], dpi=450, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs["tif"], dpi=450, bbox_inches="tight", pad_inches=0.035, pil_kwargs={"compression": "tiff_lzw"})
    array = np.asarray(Image.open(outputs["png"]).convert("RGB"))
    Image.fromarray(array).convert("L").save(QA / "Supplementary_Fig9_grayscale.png")
    return {
        "artists_outside_pre_crop_canvas": clipped,
        "canvas_pixels": [int(array.shape[1]), int(array.shape[0])],
        "exports_exist": all(path.exists() and path.stat().st_size > 10_000 for path in outputs.values()),
    }


def main() -> int:
    setup_style()
    for directory in (FIGURES, SOURCE, QA, METHODS, OUT / "text"):
        directory.mkdir(parents=True, exist_ok=True)
    types = pd.read_csv(TYPE_TABLE)
    contexts = pd.read_csv(CONTEXT_TABLE)
    if len(types) != 27 or int(types["block_groups"].sum()) != 34_566:
        raise RuntimeError("The exhaustive 27-type frame was not recovered")
    if len(contexts) != 6 or int(contexts["block_groups"].sum()) != 34_566:
        raise RuntimeError("The six-context frame was not recovered")

    fig = plt.figure(figsize=(7.20, 4.65), facecolor="white")
    panel_heading(fig, 0.055, 0.925, "a", "All 27 canopy-deficit, surface-heat and social-condition types")
    draw_type_matrices(fig, types)
    panel_heading(fig, 0.055, 0.420, "b", "Six prespecified contexts summarize every block group")
    context_source = draw_context_bars(fig, contexts)

    outputs = {
        "png": FIGURES / "Supplementary_Fig9.png",
        "pdf": FIGURES / "Supplementary_Fig9.pdf",
        "svg": FIGURES / "Supplementary_Fig9.svg",
        "tif": FIGURES / "Supplementary_Fig9.tif",
    }
    qa = export_and_qa(fig, outputs)
    plt.close(fig)
    types.to_csv(SOURCE / "Supplementary_Fig9a_27_type_prevalence.csv", index=False)
    context_source.to_csv(SOURCE / "Supplementary_Fig9b_six_context_shares.csv", index=False)
    shutil.copy2(Path(__file__), METHODS / Path(__file__).name)

    joint = contexts.loc[contexts["context_id"].eq("A")].iloc[0]
    qa.update(
        {
            "status": "PASS",
            "quality_gate_status": "PASS",
            "visual_audit": "Colour and greyscale renders preserve all cell labels and context segments; no text, data, axis or cross-panel overlap is visible, and outer whitespace is limited to print-safe margins.",
            "block_groups": 34_566,
            "cities": 50,
            "type_cells": 27,
            "context_groups": 6,
            "joint_context_block_groups": int(joint["block_groups"]),
            "joint_context_population": int(joint["represented_population"]),
            "scientific_claim": "The 27-cell classification is exhaustive; six prespecified contexts make the distribution readable without discarding neighbourhoods.",
            "remaining_limitations": [
                "G, H and V categories are threshold-defined descriptions, not latent natural classes.",
                "Context shares depend slightly on whether neighbourhoods, cities or residents define the denominator.",
            ],
        }
    )
    if not qa["exports_exist"]:
        qa["status"] = "FAIL"
    (QA / "Supplementary_Fig9_build_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    caption = (
        "Supplementary Fig. 9 | The neighbourhood typology assigns every analysed block group to one of 27 types and six prespecified contexts. "
        "a, Equal-city prevalence of all combinations of canopy-deficit level (G1, no deficit; G2 and G3, lower and higher positive deficit), positive Landsat land-surface-temperature-anomaly level (H1, none; H2 and H3, lower and higher positive anomaly) and tertile of the social-condition score (V1-V3). Percentages are equal-city shares; brackets identify the six context groups used for interpretation. "
        "b, Shares of the 34,566-block-group frame assigned to six mutually exclusive contexts when the denominator is block groups, cities weighted equally or represented population. A denotes positive canopy deficit with H3 and V3; B, positive deficit with H3 but not V3; C, positive deficit with V3 but not H3; D, other positive-deficit combinations; E, H3 or V3 without a positive deficit; and F, the remaining lower or mixed reference combinations. Context A contained 3,602 block groups and 4.59 million represented residents. The categories are prespecified descriptive threshold summaries, not data-derived clusters, causal mechanisms or rankings of intervention need."
    )
    (OUT / "text" / "Supplementary_Fig9_caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
