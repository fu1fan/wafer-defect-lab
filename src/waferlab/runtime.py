"""Runtime helpers shared by local and container workflows."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _resolve_root(var_name: str, default_root: Path) -> Path:
    value = os.getenv(var_name)
    if value:
        return Path(value).expanduser().resolve()
    return default_root.resolve()


def resolve_data_root(project_root: Path) -> Path:
    return _resolve_root("WAFERLAB_DATA_ROOT", project_root / "data")


def resolve_raw_root(project_root: Path) -> Path:
    return resolve_data_root(project_root) / "raw"


def resolve_interim_root(project_root: Path) -> Path:
    return resolve_data_root(project_root) / "interim"


def resolve_processed_root(project_root: Path) -> Path:
    return resolve_data_root(project_root) / "processed"


def resolve_output_root(project_root: Path) -> Path:
    return _resolve_root("WAFERLAB_OUTPUT_ROOT", project_root / "outputs")


def resolve_device(requested: str | None = None) -> str:
    """Resolve a preferred execution device with CUDA-first fallback."""
    requested = (requested or "auto").strip().lower()
    has_cuda = torch.cuda.is_available()

    if requested == "auto":
        return "cuda" if has_cuda else "cpu"
    if requested == "cuda" and not has_cuda:
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested
