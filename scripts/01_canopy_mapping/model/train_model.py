"""Reproduce the recovered canopy-model training configuration.

The model and loss intentionally retain the recovered implementation,
including probability output from the model before the BCE-Tversky objective.
This preserves checkpoint compatibility rather than redefining the estimator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter

from uscanopy.segmentation import build_model


class PairedPatchDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, augment: bool) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.names = sorted(path.name for path in image_dir.glob("*.tif") if (mask_dir / path.name).exists())
        if not self.names:
            raise FileNotFoundError(f"No paired GeoTIFF patches found in {image_dir} and {mask_dir}")

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _normalize(image: torch.Tensor) -> torch.Tensor:
        for index in range(image.shape[0]):
            band = image[index]
            minimum, maximum = band.min(), band.max()
            image[index] = (band - minimum) / (maximum - minimum) if maximum > minimum else band - minimum
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        with rasterio.open(self.image_dir / name) as source:
            image = torch.from_numpy(source.read(indexes=(1, 2, 3)).astype(np.float32))
        with rasterio.open(self.mask_dir / name) as source:
            mask = torch.from_numpy((source.read(1) > 0).astype(np.float32)).unsqueeze(0)
        image = TF.resize(image, [256, 256], antialias=True)
        mask = TF.resize(mask, [256, 256], interpolation=TF.InterpolationMode.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                scale = random.uniform(0.8, 1.2)
                size = [max(1, int(256 * scale)), max(1, int(256 * scale))]
                image = TF.resize(image, size, antialias=True)
                mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)
            if random.random() < 0.5:
                angle = random.uniform(-45, 45)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = ColorJitter(0.2, 0.2, 0.2, 0.1)(image)
            if random.random() < 0.3:
                sigma = random.uniform(0.1, 1.0)
                image = TF.gaussian_blur(image, 5, [sigma, sigma])
        return self._normalize(image), mask

    @staticmethod
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        images, masks = zip(*batch)
        max_height = max(item.shape[-2] for item in images)
        max_width = max(item.shape[-1] for item in images)
        image_batch = images[0].new_zeros((len(images), 3, max_height, max_width))
        mask_batch = masks[0].new_zeros((len(masks), 1, max_height, max_width))
        for index, (image, mask) in enumerate(zip(images, masks, strict=True)):
            image_batch[index, :, : image.shape[-2], : image.shape[-1]] = image
            mask_batch[index, :, : mask.shape[-2], : mask.shape[-1]] = mask
        return image_batch, mask_batch


def recovered_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    transformed = torch.sigmoid(probabilities)
    true_positive = (transformed * targets).sum(dim=(1, 2, 3))
    false_positive = ((1 - targets) * transformed).sum(dim=(1, 2, 3))
    false_negative = (targets * (1 - transformed)).sum(dim=(1, 2, 3))
    tversky = (true_positive + 1e-6) / (
        true_positive + 0.3 * false_positive + 0.7 * false_negative + 1e-6
    )
    tversky_loss = torch.mean(1 - tversky)
    bce_loss = F.binary_cross_entropy_with_logits(probabilities, targets)
    return 0.7 * tversky_loss + 0.3 * bce_loss


def confusion_counts(probabilities: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    prediction = probabilities >= 0.5
    truth = targets >= 0.5
    tp = torch.sum(prediction & truth).item()
    fp = torch.sum(prediction & ~truth).item()
    fn = torch.sum(~prediction & truth).item()
    tn = torch.sum(~prediction & ~truth).item()
    return np.asarray([tp, fp, fn, tn], dtype=np.float64)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    counts = np.zeros(4, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for images, masks in loader:
            counts += confusion_counts(model(images.to(device)).cpu(), masks)
    tp, fp, fn, tn = counts
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "iou": tp / max(tp + fp + fn, 1),
        "oa": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = [
        {"params": [], "weight_decay": 0.0},
        {"params": [], "weight_decay": weight_decay},
    ]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = 0 if len(parameter.shape) == 1 or name.endswith(".bias") else 1
        groups[target]["params"].append(parameter)
    return groups


def recovered_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = 3
    warmup_factor = 0.1
    end_factor = 1e-4

    def multiplier(step: int) -> float:
        warmup_steps = warmup_epochs * steps_per_epoch
        if step <= warmup_steps:
            alpha = step / max(warmup_steps, 1)
            return warmup_factor * (1 - alpha) + alpha
        current = step - warmup_steps
        cosine_steps = max((epochs - warmup_epochs) * steps_per_epoch, 1)
        return ((1 + math.cos(current * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-masks", type=Path, required=True)
    parser.add_argument("--validation-images", type=Path, required=True)
    parser.add_argument("--validation-masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_data = PairedPatchDataset(args.train_images, args.train_masks, augment=True)
    validation_data = PairedPatchDataset(args.validation_images, args.validation_masks, augment=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PairedPatchDataset.collate,
    )
    validation_loader = DataLoader(validation_data, batch_size=1, collate_fn=PairedPatchDataset.collate)
    model = build_model(n_channels=3).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, args.weight_decay),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = recovered_scheduler(optimizer, len(train_loader), args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    best_iou = -np.inf
    log_rows = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = recovered_loss(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, validation_loader, device)
            row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
            log_rows.append(row)
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "args": vars(args),
                    },
                    args.output / "model_best.pth",
                )
            (args.output / "training_log.json").write_text(json.dumps(log_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
