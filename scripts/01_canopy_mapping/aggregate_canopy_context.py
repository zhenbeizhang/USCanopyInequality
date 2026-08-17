from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import math
import re
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from rasterio.windows import Window
from scipy import ndimage as ndi
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import find_boundaries
from PIL import Image, ImageOps


PROJECT_ROOT = analysis_store_root()
ORIG_DIR = PROJECT_ROOT / "\u539f\u59cbtif"
MASK_DIR = PROJECT_ROOT / "predict_mosaic_clip2"
SHP_DIR = PROJECT_ROOT / "2024\u5e74\u7f8e\u56fd\u4eba\u53e3top50\u57ce\u5e02" / "city_shp"
RESULT2_CITY_TABLE = (
    PROJECT_ROOT
    / "Result"
    / "Result2"
    / "Result2_NatureCities_final"
    / "tables"
    / "result2_naturecities_final_city_equity_matrix.csv"
)
TRAIN_LOG = model_store_root() / "save_weights" / "UNet_20260610_1605" / "results20260610_1605.txt"

OUT_ROOT = PROJECT_ROOT / "Result" / "Result1" / "Result1_NatureCities_manuscript_Fig1"
FIG_DIR = OUT_ROOT / "figures"
SRC_DIR = OUT_ROOT / "source_data"
CROP_DIR = OUT_ROOT / "display_crops"
QA_DIR = OUT_ROOT / "qa"
BASEMAP_DIR = OUT_ROOT / "basemap"
for folder in [OUT_ROOT, FIG_DIR, SRC_DIR, CROP_DIR, QA_DIR, BASEMAP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


SCENE_CITIES = [
    ("New York", "NewYorkcity"),
    ("Las Vegas", "LasVegascity"),
    ("Houston", "Houstoncity"),
    ("Raleigh", "Raleighcity"),
    ("Seattle", "Seattlecity"),
    ("Miami", "Miamicity"),
]

PALETTE = {
    "black": "#252525",
    "grey": "#747474",
    "light_grey": "#D8DEE2",
    "pale": "#F3F5F4",
    "green": "#2E8B57",
    "green_dark": "#125A35",
    "green_light": "#8BC7A2",
    "blue": "#2D6FA3",
    "red": "#B64A45",
    "orange": "#D98B3A",
}

CANOPY_CMAP = LinearSegmentedColormap.from_list(
    "canopy_green",
    ["#EAF5E7", "#C7E7BC", "#7CCB89", "#2B9B5B", "#005C35"],
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_pub(fig: plt.Figure, stem: Path, dpi: int = 600) -> list[str]:
    saved = []
    for ext in ["svg", "pdf", "png", "tif"]:
        path = stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 300
        if ext == "tif":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(str(path))
    return saved


def panel_label(ax: plt.Axes, label: str, x: float = -0.05, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["black"],
        clip_on=False,
    )


def city_key_to_display(city_key: str) -> str:
    replacements = {
        "NewYorkcity": "New York",
        "LosAngelescity": "Los Angeles",
        "Albuquerquecity": "Albuquerque",
        "ColoradoSpringscity": "Colorado Springs",
        "FortWorthcity": "Fort Worth",
        "KansasCitycity": "Kansas City",
        "LongBeachcity": "Long Beach",
        "LasVegascity": "Las Vegas",
        "OklahomaCitycity": "Oklahoma City",
        "SanAntoniocity": "San Antonio",
        "SanDiegocity": "San Diego",
        "SanFranciscocity": "San Francisco",
        "SanJosecity": "San Jose",
        "VirginiaBeachcity": "Virginia Beach",
    }
    return replacements.get(city_key, city_key.replace("city", ""))


def is_tree(arr: np.ndarray) -> np.ndarray:
    return arr == 255


def is_valid(arr: np.ndarray) -> np.ndarray:
    return arr != 254


def clean_mask(mask: np.ndarray, min_obj_px: int = 90, hole_px: int = 45) -> np.ndarray:
    clean = remove_small_objects(mask.astype(bool), min_size=min_obj_px, connectivity=2)
    clean = remove_small_holes(clean, area_threshold=hole_px, connectivity=2)
    return clean.astype(bool)


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(3):
        band = arr[:, :, i]
        lo, hi = np.percentile(band, [1.5, 98.5])
        if hi <= lo:
            hi = lo + 1
        out[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def overlay_canopy(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.43) -> np.ndarray:
    img = stretch_rgb(rgb)
    color = np.array([46, 139, 87], dtype=np.float32) / 255.0
    img[mask] = (1 - alpha) * img[mask] + alpha * color
    boundary = find_boundaries(mask, mode="outer")
    img[boundary] = np.array([0.03, 0.26, 0.12], dtype=np.float32)
    return np.clip(img, 0, 1)


def crop_to_aspect(img: np.ndarray, target_aspect: float) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or target_aspect <= 0:
        return img
    current = w / h
    if current > target_aspect:
        new_w = max(1, int(round(h * target_aspect)))
        x0 = max(0, (w - new_w) // 2)
        return img[:, x0 : x0 + new_w]
    new_h = max(1, int(round(w / target_aspect)))
    y0 = max(0, (h - new_h) // 2)
    return img[y0 : y0 + new_h, :]


def window_score(mask_arr: np.ndarray) -> dict[str, float] | None:
    valid = is_valid(mask_arr)
    valid_frac = float(valid.mean())
    if valid_frac < 0.96:
        return None
    tree_raw = is_tree(mask_arr) & valid
    canopy_frac = float(tree_raw.sum() / max(valid.sum(), 1))
    if not (0.035 <= canopy_frac <= 0.30):
        return None
    clean = clean_mask(tree_raw)
    cleaned_frac = float(clean.sum() / max(valid.sum(), 1))
    if not (0.025 <= cleaned_frac <= 0.27):
        return None
    labels, ncomp = ndi.label(clean)
    if ncomp < 8:
        return None
    comp_sizes = np.bincount(labels.ravel())[1:]
    max_comp_frac = float(comp_sizes.max() / max(clean.sum(), 1)) if comp_sizes.size else 1.0
    if max_comp_frac > 0.36:
        return None
    small_removed = max(canopy_frac - cleaned_frac, 0)
    target = 0.14
    score = (
        2.2 * (1 - min(abs(cleaned_frac - target) / target, 1))
        + 1.4 * min(ncomp / 65, 1)
        + 1.2 * (1 - max_comp_frac)
        + 0.7 * min(small_removed / 0.02, 1)
    )
    return {
        "score": float(score),
        "valid_fraction": valid_frac,
        "canopy_fraction_raw": canopy_frac,
        "canopy_fraction_clean": cleaned_frac,
        "components": float(ncomp),
        "max_component_fraction": max_comp_frac,
        "small_removed_fraction": float(small_removed),
    }


def choose_sparse_window(city_key: str, win: int = 896, n_random: int = 360) -> dict[str, object]:
    mask_path = MASK_DIR / f"{city_key}_mosaic_clip2.tif"
    rgb_path = ORIG_DIR / f"{city_key}.tif"
    if not mask_path.exists() or not rgb_path.exists():
        raise FileNotFoundError(f"Missing source data for {city_key}")

    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(city_key)) % (2**32)
    rng = np.random.default_rng(seed)
    best: dict[str, object] | None = None
    with rasterio.open(mask_path) as mds, rasterio.open(rgb_path) as rds:
        max_h = min(mds.height, rds.height)
        max_w = min(mds.width, rds.width)
        candidates: list[tuple[int, int]] = []
        margin = max(win // 2, 500)
        if max_w <= win or max_h <= win:
            raise ValueError(f"Raster smaller than display window for {city_key}")
        for _ in range(n_random):
            row = int(rng.integers(margin, max_h - margin - win))
            col = int(rng.integers(margin, max_w - margin - win))
            candidates.append((row, col))
        # A loose grid improves robustness when random samples fall outside narrow city shapes.
        for row in np.linspace(margin, max_h - margin - win, 9).astype(int):
            for col in np.linspace(margin, max_w - margin - win, 9).astype(int):
                candidates.append((int(row), int(col)))
        for row, col in candidates:
            w = Window(col, row, win, win)
            arr = mds.read(1, window=w)
            metrics = window_score(arr)
            if metrics is None:
                continue
            if best is None or metrics["score"] > best["score"]:
                rgb = np.moveaxis(rds.read([1, 2, 3], window=w), 0, -1)
                if rgb.shape[:2] != arr.shape:
                    continue
                # Avoid very dark/water-dominated samples.
                if float(np.mean(rgb)) < 35 or float(np.std(rgb)) < 18:
                    continue
                best = {
                    "city_display": city_key_to_display(city_key),
                    "city_key": city_key,
                    "row": int(row),
                    "col": int(col),
                    "window_size_px": int(win),
                    "score": metrics["score"],
                    "metrics": metrics,
                    "rgb": rgb,
                    "mask_raw": is_tree(arr),
                    "mask_clean": clean_mask(is_tree(arr)),
                }
    if best is None:
        raise RuntimeError(f"No sparse urban canopy window found for {city_key}")
    return best


def write_crop_products(samples: list[dict[str, object]]) -> pd.DataFrame:
    for old_png in CROP_DIR.glob("*.png"):
        old_png.unlink()
    rows = []
    for sample in samples:
        city_key = str(sample["city_key"])
        rgb = sample["rgb"]
        raw = sample["mask_raw"]
        clean = sample["mask_clean"]
        overlay = overlay_canopy(rgb, clean)
        Image.fromarray((stretch_rgb(rgb) * 255).astype(np.uint8)).save(CROP_DIR / f"{city_key}_rgb.png")
        Image.fromarray((overlay * 255).astype(np.uint8)).save(CROP_DIR / f"{city_key}_canopy_overlay_cleaned.png")
        Image.fromarray((raw.astype(np.uint8) * 255)).save(CROP_DIR / f"{city_key}_raw_mask_window.png")
        Image.fromarray((clean.astype(np.uint8) * 255)).save(CROP_DIR / f"{city_key}_cleaned_mask_window.png")
        metrics = sample["metrics"]
        rows.append(
            {
                "city_key": city_key,
                "city_display": sample["city_display"],
                "row": sample["row"],
                "col": sample["col"],
                "window_size_px": sample["window_size_px"],
                **metrics,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(SRC_DIR / "Fig1a_sparse_scene_window_selection.csv", index=False, encoding="utf-8-sig")
    return df


def parse_training_log() -> pd.DataFrame:
    pattern = re.compile(
        r"\[epoch:\s*(?P<epoch>\d+)\].*?Precision:\s*(?P<Precision>[\d.]+)\s+"
        r"Recall:\s*(?P<Recall>[\d.]+)\s+F1:\s*(?P<F1>[\d.]+)\s+OA:\s*(?P<OA>[\d.]+)\s+"
        r"IoU:\s*(?P<IoU>[\d.]+)\s+Dice:\s*(?P<Dice>[\d.]+)\s+Kappa:\s*(?P<Kappa>[\d.]+)"
    )
    rows = []
    if TRAIN_LOG.exists():
        for line in TRAIN_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = pattern.search(line)
            if m:
                row = {k: float(v) for k, v in m.groupdict().items()}
                row["epoch"] = int(row["epoch"])
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(SRC_DIR / "Supplementary_validation_training_log_parsed.csv", index=False, encoding="utf-8-sig")
    return df


def city_centroids(city_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shp in SHP_DIR.glob("*.shp"):
        try:
            gdf = gpd.read_file(shp)
            if gdf.crs is None:
                gdf = gdf.set_crs(4326)
            gdf = gdf.to_crs(4326)
            geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
            centroid = geom.centroid
            display = shp.stem.replace(" city", "")
            key = re.sub(r"[^A-Za-z0-9]", "", display) + "city"
            rows.append({"city_key_from_shp": key, "city_display_shp": display, "lon": centroid.x, "lat": centroid.y})
        except Exception:
            continue
    cent = pd.DataFrame(rows)
    if cent.empty:
        return cent
    stats = city_stats[["city_key", "city_display", "area_weighted_canopy_pct", "population"]].copy()
    # Prefer exact display-name matches; they are more stable than heavily normalised city keys.
    out = cent.merge(stats, left_on="city_display_shp", right_on="city_display", how="left")
    out["city_display"] = out["city_display"].fillna(out["city_display_shp"])
    out.to_csv(SRC_DIR / "Fig1a_city_centroids_source_data.csv", index=False, encoding="utf-8-sig")
    return out


def load_us_states_basemap() -> gpd.GeoDataFrame:
    """Load a lightweight CONUS state basemap for display only."""
    zip_path = BASEMAP_DIR / "ne_50m_admin_1_states_provinces.zip"
    shp_path = BASEMAP_DIR / "ne_50m_admin_1_states_provinces.shp"
    url = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_1_states_provinces.zip"
    if not shp_path.exists():
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(BASEMAP_DIR)
        except Exception:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        states = gpd.read_file(shp_path)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if states.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if states.crs is None:
        states = states.set_crs("EPSG:4326")
    states = states.to_crs("EPSG:4326")
    if "admin" in states.columns:
        states = states[states["admin"].eq("United States of America")]
    elif "adm0_a3" in states.columns:
        states = states[states["adm0_a3"].eq("USA")]
    name_col = "name" if "name" in states.columns else None
    if name_col:
        states = states[~states[name_col].isin(["Alaska", "Hawaii", "Puerto Rico"])]
    states = states.cx[-126:-66, 24:50].copy()
    return states


def add_scale_bar(ax: plt.Axes, image_shape: tuple[int, int], meters: int = 100) -> None:
    px_per_meter = 1 / 0.6
    length_px = meters * px_per_meter
    h, w = image_shape
    x0 = w * 0.08
    y0 = h * 0.89
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", lw=2.4, solid_capstyle="butt", zorder=8)
    ax.plot([x0, x0 + length_px], [y0, y0], color=PALETTE["black"], lw=0.72, solid_capstyle="butt", zorder=9)
    tick_half = h * 0.010
    for xpos in [x0, x0 + length_px]:
        ax.plot([xpos, xpos], [y0 - tick_half, y0 + tick_half], color="white", lw=2.1, zorder=8)
        ax.plot([xpos, xpos], [y0 - tick_half, y0 + tick_half], color=PALETTE["black"], lw=0.65, zorder=9)
    ax.text(
        x0 + length_px / 2,
        y0 - h * 0.035,
        f"{meters} m",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color="white",
        bbox={"facecolor": (0, 0, 0, 0.35), "edgecolor": "none", "pad": 0.7},
    )


def plot_map_and_workflow(ax: plt.Axes, centroids: pd.DataFrame):
    if centroids.empty:
        ax.axis("off")
        return
    ax.set_facecolor("#F9FBF8")
    states = load_us_states_basemap()
    valid = centroids.dropna(subset=["lon", "lat"]).copy()
    valid = valid[(valid["lon"].between(-126, -66)) & (valid["lat"].between(24, 50))]
    vals = pd.to_numeric(valid["area_weighted_canopy_pct"], errors="coerce").astype(float)
    norm = Normalize(vmin=float(np.nanpercentile(vals, 2)), vmax=float(np.nanpercentile(vals, 98)))
    scaled = np.clip(norm(vals.to_numpy()), 0, 1)
    point_size = 13 + 68 * np.power(scaled, 1.35)

    if not states.empty:
        states_proj = states.to_crs("EPSG:5070")
        points = gpd.GeoDataFrame(
            valid,
            geometry=gpd.points_from_xy(valid["lon"], valid["lat"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:5070")
        states_proj.plot(ax=ax, color="#EAF0EA", edgecolor="white", linewidth=0.42, zorder=0)
        states_proj.boundary.plot(ax=ax, color="#C4D0C6", linewidth=0.28, zorder=1)
        x = points.geometry.x.to_numpy()
        y = points.geometry.y.to_numpy()
        bounds = states_proj.total_bounds
        ax.set_xlim(bounds[0] - 130000, bounds[2] + 130000)
        ax.set_ylim(bounds[1] - 85000, bounds[3] + 120000)
    else:
        conus_outline = np.array(
            [
                [-124.8, 48.8],
                [-122.4, 37.0],
                [-117.0, 32.5],
                [-106.0, 31.6],
                [-97.0, 25.8],
                [-82.0, 25.0],
                [-80.0, 31.0],
                [-74.0, 40.0],
                [-68.0, 45.0],
                [-83.0, 47.8],
                [-96.0, 49.0],
                [-111.0, 49.0],
                [-124.8, 48.8],
            ]
        )
        ax.fill(conus_outline[:, 0], conus_outline[:, 1], color="#E8EEE8", ec="#C7D0C7", lw=0.45, zorder=0)
        x = valid["lon"].to_numpy()
        y = valid["lat"].to_numpy()
        ax.set_xlim(-126, -66)
        ax.set_ylim(24, 50)
    sc = ax.scatter(
        x,
        y,
        c=vals,
        cmap=CANOPY_CMAP,
        norm=norm,
        s=point_size,
        edgecolor="white",
        linewidth=0.62,
        alpha=0.98,
        zorder=3,
    )
    ax.scatter(x, y, s=point_size + 8, color="none", edgecolor="#365943", linewidth=0.28, alpha=0.50, zorder=2.9)
    selected_names = {name for name, _ in SCENE_CITIES}
    selected_mask = valid["city_display"].isin(selected_names).to_numpy()
    ax.scatter(
        np.asarray(x)[selected_mask],
        np.asarray(y)[selected_mask],
        s=point_size[selected_mask] + 62,
        color="none",
        edgecolor=PALETTE["green_dark"],
        linewidth=0.95,
        alpha=0.90,
        zorder=3.4,
    )
    label_offsets = {
        "Seattle": (7, 8, "left"),
        "Las Vegas": (7, -9, "left"),
        "Houston": (7, -9, "left"),
        "Miami": (7, 9, "left"),
        "Raleigh": (7, -8, "left"),
        "New York": (-8, 8, "right"),
    }
    coord_lookup = {name: (xx, yy) for name, xx, yy in zip(valid["city_display"], x, y)}
    for name, (dx, dy, ha) in label_offsets.items():
        if name not in coord_lookup:
            continue
        xx, yy = coord_lookup[name]
        ax.annotate(
            name,
            xy=(xx, yy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.2,
            ha=ha,
            va="center",
            color=PALETTE["green_dark"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.18},
            arrowprops={"arrowstyle": "-", "color": PALETTE["green_dark"], "lw": 0.32, "shrinkA": 1, "shrinkB": 2},
            zorder=4,
            clip_on=False,
        )
    ax.set_aspect(0.88, adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#D7DED7")
    cax = ax.inset_axes([0.092, 0.072, 0.205, 0.030])
    cax.set_facecolor((1, 1, 1, 0.86))
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([10, 20, 30])
    cb.ax.tick_params(labelsize=5.9, length=1.5, pad=0.8, colors=PALETTE["black"])
    cax.text(
        0.0,
        1.72,
        "Canopy cover (%)",
        transform=cax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=PALETTE["black"],
    )
    return sc


def plot_scene_panel(fig: plt.Figure, spec, samples: list[dict[str, object]]) -> None:
    ncols = 2
    nrows = int(math.ceil(len(samples) / ncols))
    sub = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=spec, wspace=0.035, hspace=0.07)
    axes = []
    for i, sample in enumerate(samples):
        ax = fig.add_subplot(sub[i // ncols, i % ncols])
        axes.append(ax)
        overlay = overlay_canopy(sample["rgb"], sample["mask_clean"])
        ax.imshow(overlay)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.6)
        ax.text(
            0.025,
            0.045,
            str(sample["city_display"]),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.2,
            color=PALETTE["black"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.1},
        )
        add_scale_bar(ax, overlay.shape[:2], meters=100)
    panel_label(axes[0], "b", x=-0.08, y=1.05)


def plot_combined_atlas_panel(fig: plt.Figure, spec, centroids: pd.DataFrame, samples: list[dict[str, object]]) -> None:
    holder = fig.add_subplot(spec)
    holder.set_axis_off()
    panel_label(holder, "a", x=-0.016, y=1.008)

    gap = 0.005
    left = 0.000
    bottom = 0.026
    total_w = 1.0 - 2.0 * left
    bottom_h = 0.305
    map_x = left
    map_y = bottom + bottom_h + gap
    map_w = 0.600
    map_h = 1.0 - map_y - 0.014
    map_ax = holder.inset_axes([map_x, map_y, map_w, map_h])
    plot_map_and_workflow(map_ax, centroids)

    right_x = map_x + map_w + gap
    right_w = left + total_w - right_x
    right_h = (map_h - gap) / 2.0
    bottom_w = (total_w - 3.0 * gap) / 4.0
    positions = {
        "New York": [right_x, map_y + right_h + gap, right_w, right_h],
        "Las Vegas": [right_x, map_y, right_w, right_h],
        "Houston": [left, bottom, bottom_w, bottom_h],
        "Raleigh": [left + bottom_w + gap, bottom, bottom_w, bottom_h],
        "Seattle": [left + 2.0 * (bottom_w + gap), bottom, bottom_w, bottom_h],
        "Miami": [left + 3.0 * (bottom_w + gap), bottom, bottom_w, bottom_h],
    }
    holder_box = holder.get_position()
    holder_aspect = (holder_box.width * fig.get_figwidth()) / max(holder_box.height * fig.get_figheight(), 1e-9)
    for sample in samples:
        pos = positions.get(str(sample["city_display"]))
        if pos is None:
            continue
        chip = holder.inset_axes(pos)
        overlay = overlay_canopy(sample["rgb"], sample["mask_clean"])
        target_aspect = (pos[2] / max(pos[3], 1e-9)) * holder_aspect
        display = crop_to_aspect(overlay, target_aspect)
        chip.imshow(display, aspect="auto")
        chip.set_xticks([])
        chip.set_yticks([])
        for spine in chip.spines.values():
            spine.set_visible(True)
            spine.set_color("#D4DED5")
            spine.set_linewidth(0.65)
        chip.text(
            0.025,
            0.040,
            str(sample["city_display"]),
            transform=chip.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.1,
            color=PALETTE["black"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.95},
        )
        add_scale_bar(chip, display.shape[:2], meters=100)


def mask_to_rgb(mask: np.ndarray, bg: tuple[float, float, float] = (0.96, 0.96, 0.94)) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.float32)
    out[:, :] = bg
    out[mask] = np.array([46, 139, 87], dtype=np.float32) / 255.0
    return out


def canopy_patch_areas_m2(mask: np.ndarray) -> np.ndarray:
    labels, ncomp = ndi.label(mask.astype(bool))
    if ncomp == 0:
        return np.array([], dtype=float)
    sizes_px = np.bincount(labels.ravel())[1:]
    return sizes_px.astype(float) * 0.36


def plot_morphology_fingerprints(ax: plt.Axes, samples: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    colors = ["#184E35", "#5A8F36", "#7AA85E", "#3B7F70", "#8AAE5A", "#2E8B57"]
    y_positions = np.arange(len(samples))[::-1]
    for y, sample, color in zip(y_positions, samples, colors):
        areas = canopy_patch_areas_m2(sample["mask_clean"])
        areas = areas[areas > 0]
        if areas.size == 0:
            q10 = q50 = q90 = np.nan
        else:
            q10, q50, q90 = np.percentile(areas, [10, 50, 90])
        cover = float(sample["metrics"]["canopy_fraction_clean"] * 100)
        n_patch = int(areas.size)
        rows.append(
            {
                "city_key": sample["city_key"],
                "city_display": sample["city_display"],
                "patch_count": n_patch,
                "canopy_cover_pct_in_window": cover,
                "patch_area_p10_m2": q10,
                "patch_area_median_m2": q50,
                "patch_area_p90_m2": q90,
            }
        )
        if areas.size:
            sample_rng = np.random.default_rng(sum(ord(c) for c in str(sample["city_key"])))
            draw = np.sort(sample_rng.choice(areas, size=min(80, areas.size), replace=False))
            yy = y + sample_rng.normal(0, 0.035, size=draw.size)
            ax.scatter(draw, yy, s=3.0, color=color, alpha=0.16, linewidth=0, zorder=1)
            ax.hlines(y, q10, q90, color=color, lw=2.0, alpha=0.78, zorder=2)
            ax.scatter([q50], [y], s=25, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.text(
            0.64,
            y,
            f"{cover:.1f}% / {n_patch} patches",
            ha="left",
            va="center",
            fontsize=6.2,
            color=PALETTE["black"],
            transform=ax.get_yaxis_transform(),
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.15},
        )
    ax.set_xscale("log")
    ax.set_xlim(20, 6500)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(s["city_display"]) for s in samples], fontsize=6.5, color=PALETTE["black"])
    ax.set_xlabel("Connected canopy-patch area (m2, log scale)", color=PALETTE["black"])
    ax.grid(axis="x", which="major", color="#E3E8E4", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["black"])
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", colors=PALETTE["black"], labelcolor=PALETTE["black"], width=0.75, pad=2.2)
    ax.tick_params(axis="y", length=0, pad=2, colors=PALETTE["black"], labelcolor=PALETTE["black"])
    ax.text(
        0.64,
        1.04,
        "cover / count",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["black"],
    )
    df = pd.DataFrame(rows)
    df.to_csv(SRC_DIR / "Fig1b_canopy_morphology_fingerprints.csv", index=False, encoding="utf-8-sig")
    panel_label(ax, "b", x=-0.045, y=1.005)
    return df


def plot_postprocess(ax1: plt.Axes, ax2: plt.Axes, sample: dict[str, object]) -> None:
    raw = sample["mask_raw"]
    clean = sample["mask_clean"]
    for ax, mask, title in [(ax1, raw, "Raw mask"), (ax2, clean, "Cleaned mask")]:
        ax.imshow(mask_to_rgb(mask), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            1.025,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.6,
            color=PALETTE["black"],
            clip_on=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color(PALETTE["light_grey"])
    removed = float(sample["metrics"]["small_removed_fraction"] * 100)
    ax1.text(
        0.02,
        -0.13,
        f"small speckles removed: {removed:.1f} pp",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color=PALETTE["grey"],
        clip_on=False,
    )
    ax1.text(
        0.0,
        1.20,
        "Post-processing for display",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=PALETTE["black"],
        clip_on=False,
    )
    panel_label(ax1, "c", x=-0.24, y=1.20)


def plot_validation(ax: plt.Axes, log_df: pd.DataFrame) -> None:
    if log_df.empty:
        ax.text(0.5, 0.5, "Validation log not found", transform=ax.transAxes, ha="center", va="center", fontsize=6)
        ax.axis("off")
        return
    best = log_df.loc[log_df["F1"].idxmax()].copy()
    metrics = ["Precision", "Recall", "F1", "IoU", "OA", "Kappa"]
    vals = best[metrics].astype(float)
    y = np.arange(len(metrics))[::-1]
    colors = [PALETTE["green"] if m in {"F1", "IoU"} else "#8C8C8C" for m in metrics]
    ax.hlines(y, 0, vals.values, color=colors, lw=2.1, alpha=0.88)
    ax.scatter(vals.values, y, s=18, color=colors, zorder=3, edgecolor="white", linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=5.3)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.grid(axis="x", color="#E7E7E7", lw=0.45)
    ax.set_axisbelow(True)
    for value, yy in zip(vals.values, y):
        ax.text(value + 0.025, yy, f"{value:.2f}", ha="left", va="center", fontsize=5.0, color=PALETTE["black"])
    pd.DataFrame([best]).to_csv(SRC_DIR / "Fig1d_best_validation_metrics.csv", index=False, encoding="utf-8-sig")
    panel_label(ax, "d", x=-0.18, y=1.08)


def plot_city_ranking(ax: plt.Axes, city_stats: pd.DataFrame) -> None:
    df = city_stats.sort_values("area_weighted_canopy_pct", ascending=True).reset_index(drop=True)
    x = np.arange(len(df))
    selected_names = {name for name, _ in SCENE_CITIES}
    vals = df["area_weighted_canopy_pct"].astype(float)
    norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    colors = CANOPY_CMAP(norm(vals))
    colors = np.array(colors)
    colors[~df["city_display"].isin(selected_names), 3] = 0.78
    ax.bar(x, vals, width=0.82, color=colors, edgecolor="none", zorder=2)
    selected = df[df["city_display"].isin(selected_names)].copy()
    ax.scatter(
        selected.index,
        selected["area_weighted_canopy_pct"],
        s=18,
        color=PALETTE["green_dark"],
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    median = float(vals.median())
    ax.axhline(median, color=PALETTE["black"], lw=0.65, ls="--", alpha=0.55, zorder=1)
    ax.text(len(df) - 0.3, median + 0.65, "median", ha="right", va="bottom", fontsize=6.0, color=PALETTE["black"])
    ax.set_xlim(-0.8, len(df) + 0.9)
    ax.set_ylim(0, max(42, float(vals.max()) + 4.5))
    ax.set_xticks([0, 24, 49])
    ax.set_xticklabels(["lowest", "ranked cities", "highest"], color=PALETTE["black"], fontsize=6.2)
    ax.set_ylabel("Area-weighted canopy cover (%)", color=PALETTE["black"])
    ax.grid(axis="y", color="#E3E8E4", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(PALETTE["black"])
    ax.spines["bottom"].set_color(PALETTE["black"])
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", colors=PALETTE["black"], labelcolor=PALETTE["black"], width=0.75, pad=2.2)
    labels = selected_names
    label_offsets = {
        "Las Vegas": (7, 14, "left"),
        "Houston": (0, 18, "center"),
        "Miami": (0, 18, "center"),
        "New York": (4, 20, "left"),
        "Raleigh": (8, -20, "left"),
        "Seattle": (-32, 30, "right"),
    }
    for idx, row in df.iterrows():
        if row["city_display"] in labels:
            dx, dy, ha = label_offsets.get(row["city_display"], (6, 16, "left"))
            ax.annotate(
                row["city_display"],
                xy=(idx, row["area_weighted_canopy_pct"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.1,
                ha=ha,
                va="center",
                color=PALETTE["black"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.12},
                arrowprops={"arrowstyle": "-", "color": PALETTE["black"], "lw": 0.35, "shrinkA": 0, "shrinkB": 2},
                clip_on=True,
            )
    df[["city_key", "city_display", "area_weighted_canopy_pct", "population_weighted_canopy_pct", "population"]].to_csv(
        SRC_DIR / "Fig1c_city_canopy_supply_barcode.csv", index=False, encoding="utf-8-sig"
    )
    panel_label(ax, "c", x=-0.045, y=1.005)


def make_figure() -> dict[str, object]:
    apply_style()
    for old_csv in list(SRC_DIR.glob("Fig1*.csv")) + list(SRC_DIR.glob("Supplementary*.csv")):
        old_csv.unlink()
    city_stats = pd.read_csv(RESULT2_CITY_TABLE)
    cent = city_centroids(city_stats)
    samples = [choose_sparse_window(city_key) for _, city_key in SCENE_CITIES]
    selection = write_crop_products(samples)
    log_df = parse_training_log()
    if not log_df.empty:
        best = log_df.loc[log_df["F1"].idxmax()].copy()
        pd.DataFrame([best]).to_csv(SRC_DIR / "Supplementary_best_validation_metrics.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(8.70, 7.55))
    gs = fig.add_gridspec(
        12,
        12,
        height_ratios=[1, 1, 1, 1, 1, 1, 0.90, 0.90, 0.90, 0.90, 0.90, 0.06],
        width_ratios=[1] * 12,
    )

    plot_combined_atlas_panel(fig, gs[0:6, 0:12], cent, samples)

    ax_b = fig.add_subplot(gs[6:11, 0:6])
    plot_morphology_fingerprints(ax_b, samples)

    ax_c = fig.add_subplot(gs[6:11, 6:12])
    plot_city_ranking(ax_c, city_stats)

    fig.subplots_adjust(left=0.050, right=0.990, top=0.982, bottom=0.075, wspace=0.58, hspace=0.20)
    stem = FIG_DIR / "Fig1_Result1_canopy_mapping_manuscript"
    saved = save_pub(fig, stem)
    plt.close(fig)

    png = stem.with_suffix(".png")
    if png.exists():
        img = Image.open(png)
        ImageOps.grayscale(img).save(QA_DIR / "Fig1_Result1_canopy_mapping_manuscript_grayscale.png")

    note = {
        "core_conclusion": "A city-clipped 0.6 m tree-canopy atlas resolves sparse urban tree crowns across diverse U.S. residential and street settings.",
        "archetype": "asymmetric mixed-modality figure",
        "postprocessing": "Display windows use remove_small_objects(min_size=90 px) and remove_small_holes(area_threshold=45 px); original city rasters are not modified. Post-processing comparison is intentionally omitted from the main figure.",
        "selected_scene_cities": [s["city_display"] for s in samples],
        "selection_table": str(SRC_DIR / "Fig1a_sparse_scene_window_selection.csv"),
        "saved": saved,
    }
    (OUT_ROOT / "Fig1_Result1_canopy_mapping_design_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    return {
        "saved": saved,
        "selection_rows": int(len(selection)),
        "mean_clean_canopy_fraction": float(selection["canopy_fraction_clean"].mean()),
        "validation_rows": int(len(log_df)),
        "out_root": str(OUT_ROOT),
    }


if __name__ == "__main__":
    print(json.dumps(make_figure(), ensure_ascii=False, indent=2))
