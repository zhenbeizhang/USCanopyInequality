from __future__ import annotations

import hashlib
import json
from pathlib import Path


COG_ROOT = Path("/srv/who-has-shade/data/cogs")
MANIFEST_PATH = COG_ROOT / "cog_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []
    for record in manifest:
        path = COG_ROOT / record["output_file"]
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = sha256(path) if exists else None
        checks.append(
            {
                "file": path.name,
                "exists": exists,
                "bytesMatch": actual_bytes == record["bytes"],
                "sha256Match": actual_sha256 == record["sha256"],
            }
        )

    extras = sorted(
        path.name
        for path in COG_ROOT.glob("*.tif")
        if path.name not in {record["output_file"] for record in manifest}
    )
    failed = [check for check in checks if not all((check["exists"], check["bytesMatch"], check["sha256Match"]))]
    report = {
        "status": "PASS" if not failed and not extras and len(checks) == 50 else "FAIL",
        "manifestEntries": len(checks),
        "filesPresent": sum(bool(check["exists"]) for check in checks),
        "byteMatches": sum(bool(check["bytesMatch"]) for check in checks),
        "sha256Matches": sum(bool(check["sha256Match"]) for check in checks),
        "failed": failed,
        "extras": extras,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
