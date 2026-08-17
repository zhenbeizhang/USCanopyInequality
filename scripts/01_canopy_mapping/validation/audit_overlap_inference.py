from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy import ndimage


PROJECT = Path(str(model_store_root()))
CHECKPOINT = PROJECT / "save_weights" / "UNet_20260610_1605" / "model_best.pth"
OUT = Path(str(analysis_store_root().joinpath('Result', 'Result1', 'versions', 'step3_canopy_recovery_validation')))
SELECTION = OUT / "source_data" / "seam_audit_selected_tiles.csv"

WINDOW_SIZE = 256
ARCHIVED_STRIDE = 256
OVERLAP_STRIDE = 128
ARCHIVED_REPRODUCTION_AGREEMENT_THRESHOLD = 0.99995
BATCH_SIZE = 32
THRESHOLD = 0.5
SEED = 20260801
BOOTSTRAP_REPLICATES = 20_000


def normalise_patches(patches: np.ndarray) -> np.ndarray:
    patches = patches.astype(np.float32, copy=False)
    minima = patches.min(axis=(2, 3), keepdims=True)
    maxima = patches.max(axis=(2, 3), keepdims=True)
    ranges = maxima - minima
    return np.divide(
        patches - minima,
        ranges,
        out=np.zeros_like(patches, dtype=np.float32),
        where=ranges != 0,
    )


def infer_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    _, height, width = image.shape
    probability_sum = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint16)
    positions = [(y, x) for y in range(0, height, stride) for x in range(0, width, stride)]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), BATCH_SIZE):
            batch_positions = positions[start : start + BATCH_SIZE]
            batch = np.zeros(
                (len(batch_positions), image.shape[0], WINDOW_SIZE, WINDOW_SIZE),
                dtype=np.float32,
            )
            shapes: list[tuple[int, int]] = []
            for index, (y, x) in enumerate(batch_positions):
                y_end = min(y + WINDOW_SIZE, height)
                x_end = min(x + WINDOW_SIZE, width)
                patch = image[:, y:y_end, x:x_end]
                batch[index, :, : patch.shape[1], : patch.shape[2]] = patch
                shapes.append((patch.shape[1], patch.shape[2]))
            batch = normalise_patches(batch)
            probabilities = model(torch.from_numpy(batch).to(device)).squeeze(1).cpu().numpy()
            for index, ((y, x), (patch_height, patch_width)) in enumerate(
                zip(batch_positions, shapes)
            ):
                probability_sum[y : y + patch_height, x : x + patch_width] += probabilities[
                    index, :patch_height, :patch_width
                ]
                counts[y : y + patch_height, x : x + patch_width] += 1
    return np.divide(
        probability_sum,
        counts,
        out=np.zeros_like(probability_sum),
        where=counts > 0,
    )


def transition_rates(mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    boundary_positions = list(range(WINDOW_SIZE, mask.shape[1], WINDOW_SIZE))
    control_positions = [WINDOW_SIZE // 2 + i * WINDOW_SIZE for i in range(len(boundary_positions))]
    counts = {}
    for label, positions in (("boundary", boundary_positions), ("control", control_positions)):
        transitions = 0
        pairs = 0
        for position in positions:
            pair_valid = valid[:, position - 1] & valid[:, position]
            transitions += int(
                np.logical_xor(mask[:, position - 1], mask[:, position])[pair_valid].sum()
            )
            pairs += int(pair_valid.sum())
            pair_valid = valid[position - 1, :] & valid[position, :]
            transitions += int(
                np.logical_xor(mask[position - 1, :], mask[position, :])[pair_valid].sum()
            )
            pairs += int(pair_valid.sum())
        counts[label] = (transitions, pairs)
    boundary_rate = counts["boundary"][0] / counts["boundary"][1]
    control_rate = counts["control"][0] / counts["control"][1]
    return {
        "boundary_transition_rate": boundary_rate,
        "control_transition_rate": control_rate,
        "transition_rate_ratio": boundary_rate / control_rate,
        "transition_rate_difference_pp": 100 * (boundary_rate - control_rate),
    }


def component_density(mask: np.ndarray, valid: np.ndarray) -> float:
    # A 2-pixel minimum suppresses isolated compression/noise pixels while retaining
    # the narrow crown fragments whose inflation was targeted by the fragment-inflation audit.
    labelled, count = ndimage.label(mask & valid, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0.0
    sizes = np.bincount(labelled.ravel())[1:]
    retained = int((sizes >= 2).sum())
    return retained / valid.sum() * 1_000_000


def select_one_tile_per_city() -> pd.DataFrame:
    selected = pd.read_csv(SELECTION)
    rows = []
    for city, frame in selected.groupby("city_folder"):
        city_median = frame["approximate_canopy_fraction"].median()
        chosen = frame.iloc[
            np.argmin(np.abs(frame["approximate_canopy_fraction"].to_numpy() - city_median))
        ]
        rows.append(chosen)
    return pd.DataFrame(rows).sort_values("city_folder").reset_index(drop=True)


def bootstrap_city_means(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    values = frame[columns].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    draws = np.empty((BOOTSTRAP_REPLICATES, len(columns)), dtype=float)
    for start in range(0, BOOTSTRAP_REPLICATES, 1_000):
        size = min(1_000, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        draws[start : start + size] = values[indices].mean(axis=1)
    return {
        column: {
            "mean": float(frame[column].mean()),
            "lower_95": float(np.quantile(draws[:, index], 0.025)),
            "upper_95": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, column in enumerate(columns)
    }


def main() -> None:
    for folder in ("source_data", "qa", "tables", "predictions", "scripts"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from uscanopy.segmentation import build_model

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(n_channels=3)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    selected = select_one_tile_per_city()
    selected.to_csv(OUT / "source_data" / "overlap_inference_selected_tiles.csv", index=False)
    partial_path = OUT / "source_data" / "overlap_inference_partial_results.csv"
    if partial_path.exists():
        rows = pd.read_csv(partial_path).to_dict("records")
        completed_cities = {row["city_folder"] for row in rows}
    else:
        rows = []
        completed_cities = set()
    for index, row in selected.iterrows():
        if row["city_folder"] in completed_cities:
            continue
        with rasterio.open(row["input_path"]) as src:
            image = src.read()
        with rasterio.open(row["prediction_path"]) as src:
            archived = src.read(1) > 127
        valid = np.any(image > 0, axis=0)
        probability_archived_stride = infer_probability(
            model, image, ARCHIVED_STRIDE, device
        )
        reproduced = probability_archived_stride > THRESHOLD
        probability_overlap = infer_probability(model, image, OVERLAP_STRIDE, device)
        overlap = probability_overlap > THRESHOLD

        archived_valid = archived[valid]
        overlap_valid = overlap[valid]
        agreement = float((archived_valid == reproduced[valid]).mean())
        union = np.logical_or(archived_valid, overlap_valid).sum()
        intersection = np.logical_and(archived_valid, overlap_valid).sum()
        row_result = {
            "city_folder": row["city_folder"],
            "prediction_path": row["prediction_path"],
            "valid_pixels": int(valid.sum()),
            "archived_reproduction_agreement": agreement,
            "archived_canopy_fraction": float(archived_valid.mean()),
            "overlap_canopy_fraction": float(overlap_valid.mean()),
            "overlap_minus_archived_canopy_pp": float(
                100 * (overlap_valid.mean() - archived_valid.mean())
            ),
            "archived_overlap_pixel_iou": float(intersection / union) if union else 1.0,
            "archived_component_density_per_million_valid_px": component_density(archived, valid),
            "overlap_component_density_per_million_valid_px": component_density(overlap, valid),
            **{
                f"archived_{key}": value
                for key, value in transition_rates(archived, valid).items()
            },
            **{
                f"overlap_{key}": value
                for key, value in transition_rates(overlap, valid).items()
            },
        }
        row_result["overlap_minus_archived_component_density_percent"] = 100 * (
            row_result["overlap_component_density_per_million_valid_px"]
            / row_result["archived_component_density_per_million_valid_px"]
            - 1
        )
        rows.append(row_result)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print(
            f"[{index + 1:02d}/{len(selected)}] {row['city_folder']}: "
            f"cover delta={row_result['overlap_minus_archived_canopy_pp']:+.3f} pp, "
            f"seam ratio {row_result['archived_transition_rate_ratio']:.2f}->"
            f"{row_result['overlap_transition_rate_ratio']:.2f}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tables" / "overlap_inference_audit_by_city.csv", index=False)
    columns = [
        "overlap_minus_archived_canopy_pp",
        "archived_overlap_pixel_iou",
        "archived_transition_rate_ratio",
        "overlap_transition_rate_ratio",
        "overlap_minus_archived_component_density_percent",
    ]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cities": int(len(result)),
        "tiles": int(len(result)),
        "archived_stride_px": ARCHIVED_STRIDE,
        "overlap_stride_px": OVERLAP_STRIDE,
        "archived_reproduction_agreement_threshold": (
            ARCHIVED_REPRODUCTION_AGREEMENT_THRESHOLD
        ),
        "all_archived_predictions_reproduced_pass": bool(
            result["archived_reproduction_agreement"].min()
            >= ARCHIVED_REPRODUCTION_AGREEMENT_THRESHOLD
        ),
        "minimum_archived_reproduction_agreement": float(
            result["archived_reproduction_agreement"].min()
        ),
        "city_equal_bootstrap": bootstrap_city_means(result, columns),
        "city_medians": {column: float(result[column].median()) for column in columns},
    }
    with (OUT / "qa" / "step3_overlap_inference_audit.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    (OUT / "scripts" / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
