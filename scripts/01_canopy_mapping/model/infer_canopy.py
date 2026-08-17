"""Run checkpoint-compatible sliding-window canopy inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from uscanopy.segmentation import build_model


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch = patch.astype(np.float32, copy=True)
    for band_index in range(patch.shape[0]):
        band = patch[band_index]
        minimum = float(np.nanmin(band))
        maximum = float(np.nanmax(band))
        patch[band_index] = (band - minimum) / (maximum - minimum) if maximum > minimum else 0.0
    return patch


def predict_array(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    window_size: int = 256,
    stride: int = 256,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("Input must be a band-first raster with at least three bands")
    image = image[:3]
    height, width = image.shape[1:]
    probability_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.uint16)

    model.eval()
    with torch.inference_mode():
        for row in range(0, height, stride):
            for column in range(0, width, stride):
                row_end = min(row + window_size, height)
                column_end = min(column + window_size, width)
                patch = np.zeros((3, window_size, window_size), dtype=np.float32)
                valid_height = row_end - row
                valid_width = column_end - column
                patch[:, :valid_height, :valid_width] = image[:, row:row_end, column:column_end]
                patch = normalize_patch(patch)
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                probability = model(tensor).squeeze().detach().cpu().numpy()
                probability_sum[row:row_end, column:column_end] += probability[:valid_height, :valid_width]
                count[row:row_end, column:column_end] += 1

    probabilities = np.divide(
        probability_sum,
        count,
        out=np.zeros_like(probability_sum),
        where=count > 0,
    )
    mask = (probabilities > threshold).astype(np.uint8) * 255
    return mask, probabilities


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = build_model(n_channels=3)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Three-band input GeoTIFF")
    parser.add_argument("output", type=Path, help="Output binary canopy GeoTIFF")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--probability-output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    with rasterio.open(args.input) as source:
        image = source.read(indexes=(1, 2, 3))
        profile = source.profile.copy()
    mask, probabilities = predict_array(
        model,
        image,
        device,
        window_size=args.window_size,
        stride=args.stride,
        threshold=args.threshold,
    )
    profile.update(count=1, dtype="uint8", nodata=254, compress="lzw")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output, "w", **profile) as destination:
        destination.write(mask, 1)
    if args.probability_output:
        probability_profile = profile.copy()
        probability_profile.update(dtype="float32", nodata=np.nan)
        args.probability_output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.probability_output, "w", **probability_profile) as destination:
            destination.write(probabilities.astype(np.float32), 1)


if __name__ == "__main__":
    main()
