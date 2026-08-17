from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


INPUT_ROOT = analysis_store_root() / "inputs" / "canopy_tiles"
PREDICTION_ROOT = Path(str(analysis_store_root().joinpath('predict')))
OUT = Path(str(analysis_store_root().joinpath('Result', 'Result1', 'versions', 'step3_canopy_recovery_validation')))

WINDOW_SIZE = 256
TILE_SIZE = 5120
TILES_PER_CITY = 4
SEED = 20260801
BOOTSTRAP_REPLICATES = 20_000


def approximate_canopy_fraction(path: Path) -> float:
    with rasterio.open(path) as src:
        array = src.read(1, out_shape=(1, 80, 80), masked=False)
    return float((array > 127).mean())


def select_tiles() -> pd.DataFrame:
    rows: list[dict] = []
    for city_dir in sorted(PREDICTION_ROOT.iterdir()):
        if not city_dir.is_dir():
            continue
        candidates = sorted(city_dir.glob("*_predict-unet-high.tif"))
        candidate_rows = []
        for path in candidates:
            fraction = approximate_canopy_fraction(path)
            if 0.005 <= fraction <= 0.95:
                candidate_rows.append((path, fraction))
        if not candidate_rows:
            continue
        candidate_rows.sort(key=lambda item: (item[1], item[0].name))
        quantiles = np.linspace(0.2, 0.8, TILES_PER_CITY)
        indices = sorted(
            set(int(round(q * (len(candidate_rows) - 1))) for q in quantiles)
        )
        while len(indices) < min(TILES_PER_CITY, len(candidate_rows)):
            for index in range(len(candidate_rows)):
                if index not in indices:
                    indices.append(index)
                    if len(indices) == min(TILES_PER_CITY, len(candidate_rows)):
                        break
        for index in sorted(indices):
            prediction_path, fraction = candidate_rows[index]
            input_name = prediction_path.name.replace("_predict-unet-high", "")
            input_path = INPUT_ROOT / city_dir.name / input_name
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            rows.append(
                {
                    "city_folder": city_dir.name,
                    "prediction_path": str(prediction_path),
                    "input_path": str(input_path),
                    "approximate_canopy_fraction": fraction,
                }
            )
    return pd.DataFrame(rows)


def valid_rgb_pair(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    return np.any(rgb_a > 0, axis=0) & np.any(rgb_b > 0, axis=0)


def read_vertical_pair(src, x: int) -> np.ndarray:
    return src.read(window=Window(x - 1, 0, 2, src.height))


def read_horizontal_pair(src, y: int) -> np.ndarray:
    return src.read(window=Window(0, y - 1, src.width, 2))


def pair_counts(prediction_pair: np.ndarray, rgb_pair: np.ndarray, direction: str) -> dict[str, int]:
    if direction == "vertical":
        pred_a = prediction_pair[0, :, 0] > 127
        pred_b = prediction_pair[0, :, 1] > 127
        valid = valid_rgb_pair(rgb_pair[:, :, 0], rgb_pair[:, :, 1])
    elif direction == "horizontal":
        pred_a = prediction_pair[0, 0, :] > 127
        pred_b = prediction_pair[0, 1, :] > 127
        valid = valid_rgb_pair(rgb_pair[:, 0, :], rgb_pair[:, 1, :])
    else:
        raise ValueError(direction)
    return {
        "valid_pairs": int(valid.sum()),
        "transitions": int(np.logical_xor(pred_a, pred_b)[valid].sum()),
        "canopy_pixels": int(pred_a[valid].sum() + pred_b[valid].sum()),
        "valid_pixels": int(2 * valid.sum()),
    }


def audit_tile(row: pd.Series) -> list[dict]:
    boundary_positions = list(range(WINDOW_SIZE, TILE_SIZE, WINDOW_SIZE))
    control_positions = [WINDOW_SIZE // 2 + i * WINDOW_SIZE for i in range(len(boundary_positions))]
    records: list[dict] = []
    with rasterio.open(row["prediction_path"]) as pred_src, rasterio.open(row["input_path"]) as rgb_src:
        if pred_src.width != TILE_SIZE or pred_src.height != TILE_SIZE:
            raise ValueError(f"Unexpected prediction tile size: {pred_src.name} {pred_src.shape}")
        prediction = pred_src.read(1)
        rgb = rgb_src.read()
        for line_type, positions in (("boundary", boundary_positions), ("control", control_positions)):
            for direction in ("vertical", "horizontal"):
                total = {"valid_pairs": 0, "transitions": 0, "canopy_pixels": 0, "valid_pixels": 0}
                for position in positions:
                    if direction == "vertical":
                        prediction_pair = prediction[np.newaxis, :, position - 1 : position + 1]
                        rgb_pair = rgb[:, :, position - 1 : position + 1]
                    else:
                        prediction_pair = prediction[np.newaxis, position - 1 : position + 1, :]
                        rgb_pair = rgb[:, position - 1 : position + 1, :]
                    counts = pair_counts(prediction_pair, rgb_pair, direction)
                    for key, value in counts.items():
                        total[key] += value
                records.append(
                    {
                        "city_folder": row["city_folder"],
                        "prediction_path": row["prediction_path"],
                        "approximate_canopy_fraction": row["approximate_canopy_fraction"],
                        "line_type": line_type,
                        "direction": direction,
                        **total,
                    }
                )
    return records


def aggregate_rates(frame: pd.DataFrame) -> dict[str, float]:
    totals = frame[["valid_pairs", "transitions", "canopy_pixels", "valid_pixels"]].sum()
    return {
        "transition_rate": float(totals["transitions"] / totals["valid_pairs"]),
        "canopy_fraction": float(totals["canopy_pixels"] / totals["valid_pixels"]),
        "valid_pairs": int(totals["valid_pairs"]),
    }


def contrast(frame: pd.DataFrame) -> dict[str, float]:
    boundary = aggregate_rates(frame.loc[frame["line_type"] == "boundary"])
    control = aggregate_rates(frame.loc[frame["line_type"] == "control"])
    return {
        "boundary_transition_rate": boundary["transition_rate"],
        "control_transition_rate": control["transition_rate"],
        "transition_rate_difference_pp": 100 * (
            boundary["transition_rate"] - control["transition_rate"]
        ),
        "transition_rate_ratio": boundary["transition_rate"] / control["transition_rate"],
        "boundary_canopy_fraction": boundary["canopy_fraction"],
        "control_canopy_fraction": control["canopy_fraction"],
        "canopy_fraction_difference_pp": 100 * (
            boundary["canopy_fraction"] - control["canopy_fraction"]
        ),
        "boundary_valid_pairs": boundary["valid_pairs"],
        "control_valid_pairs": control["valid_pairs"],
    }


def city_bootstrap(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    cities = sorted(frame["city_folder"].unique())
    rng = np.random.default_rng(SEED)
    names = [
        "transition_rate_difference_pp",
        "transition_rate_ratio",
        "canopy_fraction_difference_pp",
    ]
    draws = {name: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for name in names}
    count_columns = ["valid_pairs", "transitions", "canopy_pixels", "valid_pixels"]
    city_counts = np.zeros((len(cities), 2, len(count_columns)), dtype=np.int64)
    for city_index, city in enumerate(cities):
        city_frame = frame.loc[frame["city_folder"] == city]
        for line_index, line_type in enumerate(("boundary", "control")):
            city_counts[city_index, line_index] = (
                city_frame.loc[city_frame["line_type"] == line_type, count_columns]
                .sum()
                .to_numpy(dtype=np.int64)
            )
    position = 0
    chunk_size = 1_000
    while position < BOOTSTRAP_REPLICATES:
        size = min(chunk_size, BOOTSTRAP_REPLICATES - position)
        indices = rng.integers(0, len(cities), size=(size, len(cities)))
        sampled = city_counts[indices].sum(axis=1)
        boundary = sampled[:, 0, :]
        control = sampled[:, 1, :]
        boundary_transition = boundary[:, 1] / boundary[:, 0]
        control_transition = control[:, 1] / control[:, 0]
        values = {
            "transition_rate_difference_pp": 100 * (boundary_transition - control_transition),
            "transition_rate_ratio": boundary_transition / control_transition,
            "canopy_fraction_difference_pp": 100
            * (boundary[:, 2] / boundary[:, 3] - control[:, 2] / control[:, 3]),
        }
        for name in names:
            draws[name][position : position + size] = values[name]
        position += size
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in draws.items()
    }


def main() -> None:
    for folder in ("source_data", "qa", "tables", "scripts"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    selection_path = OUT / "source_data" / "seam_audit_selected_tiles.csv"
    if selection_path.exists():
        selected = pd.read_csv(selection_path)
    else:
        selected = select_tiles()
        selected.to_csv(selection_path, index=False)
    records: list[dict] = []
    for _, row in selected.iterrows():
        records.extend(audit_tile(row))
    line_frame = pd.DataFrame(records)
    line_frame.to_csv(OUT / "source_data" / "seam_line_counts.csv", index=False)

    city_rows = []
    for city, frame in line_frame.groupby("city_folder"):
        city_rows.append({"city_folder": city, **contrast(frame)})
    city_frame = pd.DataFrame(city_rows).sort_values("city_folder")
    city_frame.to_csv(OUT / "tables" / "seam_audit_by_city.csv", index=False)

    national = contrast(line_frame)
    intervals = city_bootstrap(line_frame)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cities": int(selected["city_folder"].nunique()),
        "tiles": int(len(selected)),
        "window_size_px": WINDOW_SIZE,
        "stride_px": WINDOW_SIZE,
        "comparison": "internal inference boundaries versus window-midpoint controls",
        **national,
        **{
            f"{name}_{bound}": value
            for name, bounds in intervals.items()
            for bound, value in bounds.items()
        },
    }
    with (OUT / "qa" / "step3_seam_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    Path(OUT / "scripts" / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
