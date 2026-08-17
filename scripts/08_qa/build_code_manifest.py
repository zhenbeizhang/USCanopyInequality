"""Build a file-level SHA-256 manifest for the code release."""

from __future__ import annotations

import csv
import hashlib
import ast
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "manifests" / "code_manifest_sha256.csv"
SCRIPT_OUTPUT = ROOT / "manifests" / "script_inventory.csv"
AUDIT_OUTPUT = ROOT / "manifests" / "code_release_audit.json"
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "runtime",
    "work",
    "model_work",
    "manuscript",
    "external",
    "outputs",
}

RELEASE_VERSION = "0.3.0"


def is_excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return bool(EXCLUDED_PARTS.intersection(relative.parts))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> list[Path]:
    """Return files Git would include, including files not committed yet."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    files = []
    for relative_text in result.stdout.splitlines():
        path = ROOT / relative_text
        if path.is_file() and path != OUTPUT and not is_excluded(path):
            files.append(path)
    return sorted(files)


def main() -> None:
    script_rows = []
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        relative = path.relative_to(ROOT)
        source = path.read_text(encoding="utf-8")
        try:
            description = (ast.get_docstring(ast.parse(source)) or "").split("\n\n", 1)[0]
        except SyntaxError:
            description = "SYNTAX ERROR"
        if not description:
            description = path.stem.replace("_", " ").capitalize() + "."
        module = relative.parts[1] if len(relative.parts) > 1 else ""
        if module == "06_main_figures" and path.name.startswith("make_figure_"):
            role = "main figure"
        elif module == "06_main_figures":
            role = "figure support"
        elif module == "07_supplementary_figures":
            role = "supplementary figure"
        elif module == "08_qa":
            role = "quality assurance"
        elif "validation" in relative.parts:
            role = "model validation"
        else:
            role = "analysis"
        script_rows.append((relative.as_posix(), module, role, " ".join(description.split())))

    SCRIPT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SCRIPT_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("relative_path", "module", "role", "description"))
        writer.writerows(script_rows)

    python_files = [path for path in ROOT.rglob("*.py") if not is_excluded(path)]
    forbidden_suffixes = {".pth", ".pt", ".onnx", ".tif", ".tiff", ".gpkg", ".docx"}
    forbidden_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and not is_excluded(path) and path.suffix.lower() in forbidden_suffixes
    ]
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_version": RELEASE_VERSION,
        "python_files": len(python_files),
        "analysis_and_figure_scripts": len(script_rows),
        "main_figure_scripts": len(list((ROOT / "scripts" / "06_main_figures").glob("make_figure_[1-5].py"))),
        "supplementary_figure_scripts": len(
            list((ROOT / "scripts" / "07_supplementary_figures").glob("make_supplementary_figure_*.py"))
        ),
        "forbidden_binary_artifacts": forbidden_files,
        "full_analysis_frames_included": False,
        "full_resolution_rasters_included": False,
        "lightweight_atlas_layers_included": True,
        "checkpoint_included": False,
    }
    with AUDIT_OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")

    rows = []
    for path in release_files():
        relative = path.relative_to(ROOT)
        rows.append((relative.as_posix(), path.stat().st_size, sha256(path)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("relative_path", "bytes", "sha256"))
        writer.writerows(rows)
    print(f"Wrote {len(rows)} manifest rows to {OUTPUT}")


if __name__ == "__main__":
    main()
