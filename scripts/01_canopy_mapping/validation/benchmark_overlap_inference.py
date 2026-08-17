"""Benchmark exact float32 overlap inference for a representative city tile."""

from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch


PROJECT = Path(str(model_store_root()))
CHECKPOINT = PROJECT / "save_weights" / "UNet_20260610_1605" / "model_best.pth"
OUT = Path(
    str(analysis_store_root().joinpath('Result', 'Result1', 'versions', 'step3_canopy_recovery_validation'))
)
SELECTION = OUT / "source_data" / "seam_audit_selected_tiles.csv"
WINDOW_SIZE = 256
STRIDE = 128
THRESHOLD = 0.5


def normalise(patches: np.ndarray) -> np.ndarray:
    patches = patches.astype(np.float32, copy=False)
    minima = patches.min(axis=(2, 3), keepdims=True)
    maxima = patches.max(axis=(2, 3), keepdims=True)
    ranges = maxima - minima
    return np.divide(
        patches - minima,
        ranges,
        out=np.zeros_like(patches),
        where=ranges != 0,
    )


def infer(model, image, batch_size, device, amp=False, channels_last=False):
    _, height, width = image.shape
    total = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint16)
    positions = [
        (y, x)
        for y in range(0, height, STRIDE)
        for x in range(0, width, STRIDE)
    ]
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            current = positions[start : start + batch_size]
            batch = np.zeros(
                (len(current), image.shape[0], WINDOW_SIZE, WINDOW_SIZE),
                dtype=np.float32,
            )
            shapes = []
            for index, (y, x) in enumerate(current):
                patch = image[:, y : y + WINDOW_SIZE, x : x + WINDOW_SIZE]
                batch[index, :, : patch.shape[1], : patch.shape[2]] = patch
                shapes.append(patch.shape[1:])
            tensor = torch.from_numpy(normalise(batch)).to(device)
            if channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp
            ):
                probability = model(tensor).squeeze(1).float().cpu().numpy()
            for index, ((y, x), (patch_height, patch_width)) in enumerate(
                zip(current, shapes)
            ):
                total[y : y + patch_height, x : x + patch_width] += probability[
                    index, :patch_height, :patch_width
                ]
                counts[y : y + patch_height, x : x + patch_width] += 1
    return total / counts


def main() -> None:
    from uscanopy.segmentation import build_model

    selected = pd.read_csv(SELECTION)
    row = selected.loc[selected["city_folder"].eq("Albuquerquecity")].iloc[0]
    with rasterio.open(row["input_path"]) as source:
        image = source.read()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(n_channels=3)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    records = []
    reference = None
    variants = (
        (32, False, False, False),
        (64, False, False, False),
        (96, False, False, False),
        (64, True, False, False),
        (128, True, False, False),
        (64, False, True, True),
        (128, False, True, True),
    )
    for batch_size, amp, channels_last, fast_cudnn in variants:
        torch.backends.cudnn.benchmark = fast_cudnn
        torch.backends.cudnn.deterministic = not fast_cudnn
        torch.backends.cudnn.allow_tf32 = fast_cudnn
        torch.backends.cuda.matmul.allow_tf32 = fast_cudnn
        model = model.to(
            memory_format=(
                torch.channels_last if channels_last else torch.contiguous_format
            )
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.perf_counter()
        probability = infer(
            model,
            image,
            batch_size,
            device,
            amp=amp,
            channels_last=channels_last,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = probability
            maximum_difference = 0.0
            mask_agreement = 1.0
        else:
            maximum_difference = float(np.max(np.abs(probability - reference)))
            mask_agreement = float(
                ((probability > THRESHOLD) == (reference > THRESHOLD)).mean()
            )
        records.append(
            {
                "batch_size": batch_size,
                "float16_autocast": amp,
                "channels_last": channels_last,
                "fast_cudnn": fast_cudnn,
                "elapsed_seconds": elapsed,
                "max_probability_difference_from_batch32": maximum_difference,
                "binary_agreement_with_batch32": mask_agreement,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
        torch.cuda.reset_peak_memory_stats()
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "tables" / "overlap_inference_gpu_benchmark.csv", index=False)
    with (OUT / "qa" / "step3_overlap_benchmark.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(records, stream, indent=2)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
