from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import hashlib
import json
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(str(model_store_root()))
CHECKPOINT = PROJECT / "save_weights" / "UNet_20260610_1605" / "model_best.pth"
SAMPLE_ROOT = analysis_store_root() / "inputs" / "annotations"
OUT = Path(str(analysis_store_root().joinpath('Result', 'Result1', 'versions', 'step3_canopy_recovery_validation')))

SPLITS = {
    "train": (SAMPLE_ROOT / "train" / "Train-Image", SAMPLE_ROOT / "train" / "Train-Mask"),
    "checkpoint_selection": (SAMPLE_ROOT / "test" / "Test-Image", SAMPLE_ROOT / "test" / "Test-Mask"),
    "model_selection_independent": (
        SAMPLE_ROOT / "predict" / "Predict-Image",
        SAMPLE_ROOT / "predict" / "Predict-Mask",
    ),
}

SEED = 20260801
THRESHOLD = 0.5
BOOTSTRAP_REPLICATES = 20_000
BATCH_SIZE = 16


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode("ascii"))
    h.update(str(array.dtype).encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()


def base_id(name: str) -> str:
    return name.split("_x", 1)[0]


def patch_offsets(name: str) -> tuple[int, int]:
    match = re.search(r"_x(\d+)_y(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse patch offsets from {name}")
    return int(match.group(1)), int(match.group(2))


def read_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(image_path) as src:
        image = src.read()
        metadata = {
            "crs": str(src.crs),
            "resolution_x": abs(float(src.transform.a)),
            "resolution_y": abs(float(src.transform.e)),
            "left": float(src.bounds.left),
            "bottom": float(src.bounds.bottom),
            "right": float(src.bounds.right),
            "top": float(src.bounds.top),
            "bands": int(src.count),
            "width": int(src.width),
            "height": int(src.height),
            "dtype": str(src.dtypes[0]),
        }
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return image, mask, metadata


def build_inventory() -> pd.DataFrame:
    rows: list[dict] = []
    for split, (image_dir, mask_dir) in SPLITS.items():
        image_paths = sorted(image_dir.glob("*.tif"))
        for image_path in image_paths:
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            image, mask, meta = read_pair(image_path, mask_path)
            x_offset, y_offset = patch_offsets(image_path.name)
            image_hash = sha256_array(image)
            mask_binary = (mask > 127).astype(np.uint8)
            mask_hash = sha256_array(mask_binary)
            pair_hash = hashlib.sha256(f"{image_hash}:{mask_hash}".encode("ascii")).hexdigest()
            rows.append(
                {
                    "split": split,
                    "filename": image_path.name,
                    "base_id": base_id(image_path.name),
                    "x_offset_px": x_offset,
                    "y_offset_px": y_offset,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image_hash": image_hash,
                    "mask_hash": mask_hash,
                    "image_mask_pair_hash": pair_hash,
                    "labelled_canopy_fraction": float(mask_binary.mean()),
                    "image_modified_utc": datetime.fromtimestamp(
                        image_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    **meta,
                }
            )
    inventory = pd.DataFrame(rows)

    train = inventory.loc[inventory["split"] == "train"]
    selected = inventory.loc[inventory["split"] == "checkpoint_selection"]
    train_ids = set(train["base_id"])
    selected_ids = set(selected["base_id"])
    train_image_hashes = set(train["image_hash"])
    selected_image_hashes = set(selected["image_hash"])
    train_pair_hashes = set(train["image_mask_pair_hash"])
    selected_pair_hashes = set(selected["image_mask_pair_hash"])

    inventory["base_id_seen_in_train"] = inventory["base_id"].isin(train_ids)
    inventory["base_id_seen_in_checkpoint_selection"] = inventory["base_id"].isin(selected_ids)
    inventory["image_duplicate_of_train"] = inventory["image_hash"].isin(train_image_hashes)
    inventory["image_duplicate_of_checkpoint_selection"] = inventory["image_hash"].isin(
        selected_image_hashes
    )
    inventory["pair_duplicate_of_train"] = inventory["image_mask_pair_hash"].isin(train_pair_hashes)
    inventory["pair_duplicate_of_checkpoint_selection"] = inventory[
        "image_mask_pair_hash"
    ].isin(selected_pair_hashes)

    # A base tile contains four 256-pixel windows at offsets 0 and 154; windows from
    # different splits therefore overlap spatially whenever the base identifier recurs.
    inventory["training_clean"] = ~(
        inventory["base_id_seen_in_train"]
        | inventory["image_duplicate_of_train"]
        | inventory["pair_duplicate_of_train"]
    )
    inventory["fully_untouched"] = ~(
        inventory["base_id_seen_in_train"]
        | inventory["base_id_seen_in_checkpoint_selection"]
        | inventory["image_duplicate_of_train"]
        | inventory["image_duplicate_of_checkpoint_selection"]
        | inventory["pair_duplicate_of_train"]
        | inventory["pair_duplicate_of_checkpoint_selection"]
    )
    return inventory


class PatchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with rasterio.open(row["image_path"]) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(row["mask_path"]) as src:
            mask = src.read(1)

        # This exactly reproduces torchvision ToTensor followed by the archived
        # per-patch, per-band min-max normalisation. The initial uint8 / 255 scaling
        # cancels algebraically in the subsequent min-max operation.
        for band_index in range(image.shape[0]):
            band = image[band_index]
            minimum = float(band.min())
            maximum = float(band.max())
            if maximum != minimum:
                image[band_index] = (band - minimum) / (maximum - minimum)
            else:
                image[band_index] = band - minimum

        target = (mask > 127).astype(np.uint8)
        return (
            torch.from_numpy(image),
            torch.from_numpy(target),
            row["filename"],
            row["base_id"],
        )


def confusion_counts(target: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int, int]:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    tp = int(np.logical_and(target, prediction).sum())
    tn = int(np.logical_and(~target, ~prediction).sum())
    fp = int(np.logical_and(~target, prediction).sum())
    fn = int(np.logical_and(target, ~prediction).sum())
    return tn, fp, fn, tp


def metrics_from_counts(counts: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = np.asarray(counts, dtype=np.float64)
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    oa = (tp + tn) / total if total else np.nan
    expected = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / (total * total)
    kappa = (oa - expected) / (1 - expected) if expected < 1 else np.nan
    observed_fraction = (tp + fn) / total if total else np.nan
    predicted_fraction = (tp + fp) / total if total else np.nan
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "overall_accuracy": float(oa),
        "kappa": float(kappa),
        "observed_canopy_fraction": float(observed_fraction),
        "predicted_canopy_fraction": float(predicted_fraction),
        "canopy_fraction_bias_pp": float(100 * (predicted_fraction - observed_fraction)),
    }


def evaluate_frame(model: torch.nn.Module, frame: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(
        PatchDataset(frame),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict] = []
    model.eval()
    with torch.inference_mode():
        for images, targets, filenames, base_ids in loader:
            images = images.to(device, non_blocking=True)
            probabilities = model(images).squeeze(1).cpu().numpy()
            predictions = probabilities > THRESHOLD
            targets_np = targets.numpy().astype(bool)
            for i, filename in enumerate(filenames):
                counts = confusion_counts(targets_np[i], predictions[i])
                metrics = metrics_from_counts(np.asarray(counts))
                rows.append(
                    {
                        "filename": filename,
                        "base_id": base_ids[i],
                        "tn": counts[0],
                        "fp": counts[1],
                        "fn": counts[2],
                        "tp": counts[3],
                        "mean_probability": float(probabilities[i].mean()),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def clustered_bootstrap(
    frame: pd.DataFrame, replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED
) -> dict[str, dict[str, float]]:
    grouped = frame.groupby("base_id", sort=True)[["tn", "fp", "fn", "tp"]].sum()
    base_counts = grouped.to_numpy(dtype=np.int64)
    n_clusters = len(base_counts)
    rng = np.random.default_rng(seed)
    metric_names = list(metrics_from_counts(base_counts.sum(axis=0)))
    draws = {name: np.empty(replicates, dtype=np.float64) for name in metric_names}
    position = 0
    chunk_size = 1_000
    while position < replicates:
        size = min(chunk_size, replicates - position)
        indices = rng.integers(0, n_clusters, size=(size, n_clusters))
        sampled_counts = base_counts[indices].sum(axis=1)
        for j, counts in enumerate(sampled_counts):
            metrics = metrics_from_counts(counts)
            for name in metric_names:
                draws[name][position + j] = metrics[name]
        position += size
    return {
        name: {
            "lower_95": float(np.nanquantile(values, 0.025)),
            "upper_95": float(np.nanquantile(values, 0.975)),
        }
        for name, values in draws.items()
    }


def summarise_subset(
    name: str, patch_results: pd.DataFrame, membership: pd.Series
) -> tuple[dict, pd.DataFrame]:
    subset = patch_results.loc[membership].copy()
    if subset.empty:
        raise RuntimeError(f"Validation subset {name} is empty")
    counts = subset[["tn", "fp", "fn", "tp"]].sum().to_numpy()
    estimates = metrics_from_counts(counts)
    intervals = clustered_bootstrap(subset)
    summary = {
        "analysis_set": name,
        "patches": int(len(subset)),
        "base_tile_clusters": int(subset["base_id"].nunique()),
        "pixels": int(counts.sum()),
        **{f"{key}_estimate": value for key, value in estimates.items()},
        **{
            f"{key}_{bound}": value
            for key, bounds in intervals.items()
            for bound, value in bounds.items()
        },
    }
    return summary, subset


def source_hash_table() -> pd.DataFrame:
    paths = [
        CHECKPOINT,
        repo_root() / "src" / "uscanopy" / "segmentation.py",
        repo_root() / "scripts" / "01_canopy_mapping" / "model" / "train_model.py",
        repo_root() / "scripts" / "01_canopy_mapping" / "model" / "infer_canopy.py",
        PROJECT / "save_weights" / "UNet_20260610_1605" / "UNet.txt",
        PROJECT / "save_weights" / "UNet_20260610_1605" / "results20260610_1605.txt",
    ]
    return pd.DataFrame(
        [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
    )


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for folder in ("source_data", "qa", "tables", "figures", "predictions", "text", "scripts"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    source_hashes = source_hash_table()
    source_hashes.to_csv(OUT / "source_data" / "model_and_code_hashes.csv", index=False)

    inventory = build_inventory()
    inventory.to_csv(OUT / "source_data" / "annotation_split_inventory.csv", index=False)

    split_summary = (
        inventory.groupby("split")
        .agg(
            patches=("filename", "size"),
            base_tile_clusters=("base_id", "nunique"),
            labelled_canopy_fraction_mean=("labelled_canopy_fraction", "mean"),
            labelled_canopy_fraction_median=("labelled_canopy_fraction", "median"),
            base_id_seen_in_train=("base_id_seen_in_train", "sum"),
            base_id_seen_in_checkpoint_selection=("base_id_seen_in_checkpoint_selection", "sum"),
            image_duplicate_of_train=("image_duplicate_of_train", "sum"),
            pair_duplicate_of_train=("pair_duplicate_of_train", "sum"),
            training_clean=("training_clean", "sum"),
            fully_untouched=("fully_untouched", "sum"),
        )
        .reset_index()
    )
    split_summary.to_csv(OUT / "tables" / "split_overlap_audit.csv", index=False)

    from uscanopy.segmentation import build_model

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(n_channels=3)
    load_result = model.load_state_dict(checkpoint["model"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(str(load_result))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_results: list[pd.DataFrame] = []
    for split in ("checkpoint_selection", "model_selection_independent"):
        split_inventory = inventory.loc[inventory["split"] == split].copy()
        result = evaluate_frame(model, split_inventory, device)
        result["split"] = split
        result = result.merge(
            split_inventory[
                [
                    "filename",
                    "training_clean",
                    "fully_untouched",
                    "labelled_canopy_fraction",
                    "base_id_seen_in_train",
                    "base_id_seen_in_checkpoint_selection",
                    "image_duplicate_of_train",
                    "image_duplicate_of_checkpoint_selection",
                ]
            ],
            on="filename",
            how="left",
            validate="one_to_one",
        )
        all_results.append(result)
    patch_results = pd.concat(all_results, ignore_index=True)
    patch_results.to_csv(OUT / "source_data" / "patch_level_validation_results.csv", index=False)

    summaries: list[dict] = []
    selection = patch_results["split"].eq("checkpoint_selection")
    independent = patch_results["split"].eq("model_selection_independent")
    requested_subsets = [
        ("checkpoint-selection split (not independent)", selection),
        ("all model-selection-independent patches", independent),
        ("training-base-tile-clean independent patches", independent & patch_results["training_clean"]),
        (
            "fully untouched base tiles (primary independent test)",
            independent & patch_results["fully_untouched"],
        ),
    ]
    subset_exports: dict[str, pd.DataFrame] = {}
    for name, membership in requested_subsets:
        summary, subset = summarise_subset(name, patch_results, membership)
        summaries.append(summary)
        subset_exports[name] = subset

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUT / "tables" / "independent_validation_metrics.csv", index=False)

    primary = subset_exports["fully untouched base tiles (primary independent test)"].copy()
    primary["canopy_density_group"] = pd.qcut(
        primary["labelled_canopy_fraction"],
        q=3,
        labels=["lower third", "middle third", "upper third"],
        duplicates="drop",
    )
    density_summaries: list[dict] = []
    for group, frame in primary.groupby("canopy_density_group", observed=True):
        counts = frame[["tn", "fp", "fn", "tp"]].sum().to_numpy()
        density_summaries.append(
            {
                "canopy_density_group": str(group),
                "patches": len(frame),
                "base_tile_clusters": frame["base_id"].nunique(),
                "labelled_canopy_fraction_min": frame["labelled_canopy_fraction"].min(),
                "labelled_canopy_fraction_median": frame["labelled_canopy_fraction"].median(),
                "labelled_canopy_fraction_max": frame["labelled_canopy_fraction"].max(),
                **metrics_from_counts(counts),
            }
        )
    pd.DataFrame(density_summaries).to_csv(
        OUT / "tables" / "primary_test_metrics_by_canopy_density.csv", index=False
    )

    selection_row = summary_frame.iloc[0]
    reproduction_differences = {
        "precision_abs_difference": abs(selection_row["precision_estimate"] - 0.846),
        "recall_abs_difference": abs(selection_row["recall_estimate"] - 0.855),
        "f1_abs_difference": abs(selection_row["f1_estimate"] - 0.851),
        "iou_abs_difference": abs(selection_row["iou_estimate"] - 0.740),
        "overall_accuracy_abs_difference": abs(
            selection_row["overall_accuracy_estimate"] - 0.930
        ),
    }
    qa = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": source_hashes.loc[
            source_hashes["path"] == str(CHECKPOINT), "sha256"
        ].iloc[0],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_args": vars(checkpoint["args"]),
        "model_class": "src.UNET_nogate.UNet",
        "threshold": THRESHOLD,
        "device": str(device),
        "bootstrap_unit": "base tile identifier",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "selection_metric_reproduction": reproduction_differences,
        "selection_metric_reproduction_pass": bool(
            max(reproduction_differences.values()) <= 0.0015
        ),
        "all_expected_split_counts_pass": bool(
            dict(zip(split_summary["split"], split_summary["patches"]))
            == {"checkpoint_selection": 1211, "model_selection_independent": 1213, "train": 9693}
        ),
        "strict_state_dict_load_pass": True,
        "primary_test_has_at_least_50_base_tiles_pass": bool(primary["base_id"].nunique() >= 50),
    }
    qa["overall_pass"] = all(
        qa[key]
        for key in (
            "selection_metric_reproduction_pass",
            "all_expected_split_counts_pass",
            "strict_state_dict_load_pass",
            "primary_test_has_at_least_50_base_tiles_pass",
        )
    )
    with (OUT / "qa" / "step3_independent_validation_qa.json").open("w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2, ensure_ascii=False)

    shutil.copy2(Path(__file__), OUT / "scripts" / Path(__file__).name)
    print(summary_frame.to_string(index=False))
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
