"""Verify that the public model definition strictly loads the frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from uscanopy.paths import model_store_root
from uscanopy.segmentation import build_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state(payload: object) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")
    metadata = {key: value for key, value in payload.items() if key not in {"model", "state_dict"}}
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not state:
        raise TypeError("Checkpoint does not contain a model state dictionary.")
    if all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=model_store_root() / "save_weights" / "UNet_20260610_1605" / "model_best.pth",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    with torch.serialization.safe_globals([argparse.Namespace]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state, metadata = extract_state(payload)

    model = build_model()
    incompatibility = model.load_state_dict(state, strict=True)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": metadata.get("epoch"),
        "strict_load": True,
        "missing_keys": list(incompatibility.missing_keys),
        "unexpected_keys": list(incompatibility.unexpected_keys),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "public_model": "uscanopy.segmentation.UNet",
    }
    destination = args.output
    if destination is None:
        destination = checkpoint.parent / "checkpoint_compatibility.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
