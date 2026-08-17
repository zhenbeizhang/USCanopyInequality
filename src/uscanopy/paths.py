"""Portable path configuration for the archived analysis scripts.

The original analysis was developed across several local Windows volumes.
Public scripts resolve those locations through environment variables so the
analysis can be run on any operating system without editing source files.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolved_env(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else fallback.resolve()


def repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[2]


def analysis_store_root() -> Path:
    """Root of the staged analysis archive supplied by the data release."""

    return _resolved_env("USCANOPY_ANALYSIS_ROOT", repo_root() / "work")


def model_store_root() -> Path:
    """Root containing model-development inputs and checkpoints."""

    return _resolved_env("USCANOPY_MODEL_ROOT", repo_root() / "model_work")


def manuscript_root() -> Path:
    """Optional location of manuscript files used by reporting audits."""

    return _resolved_env("USCANOPY_MANUSCRIPT_ROOT", repo_root() / "manuscript")


def external_input_root() -> Path:
    """Optional location for third-party inputs not distributed with the code."""

    return _resolved_env("USCANOPY_EXTERNAL_ROOT", repo_root() / "external")
