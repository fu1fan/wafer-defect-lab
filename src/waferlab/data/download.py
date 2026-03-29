"""Kaggle dataset download utilities for the wafer defect project."""

from __future__ import annotations

import shutil
from pathlib import Path


DATASETS: dict[str, dict[str, str]] = {
    "WM-811K": {
        "handle": "qingyi/wm811k-wafer-map",
        "target_dir": "wm811k",
    },
    "MixedWM38": {
        "handle": "co1d7era/mixedtype-wafer-defect-datasets",
        "target_dir": "mixedwm38",
    },
}

AUTO_ALIASES: set[str] = {"auto", "all"}


def resolve_dataset_names(requested: list[str]) -> list[str]:
    """Resolve requested dataset names, expanding auto/all aliases."""
    lowered = {name.lower() for name in requested}
    if lowered & AUTO_ALIASES:
        return list(DATASETS.keys())

    invalid = [name for name in requested if name not in DATASETS]
    if invalid:
        available = ", ".join(list(DATASETS.keys()) + ["auto", "all"])
        raise ValueError(
            f"Unsupported dataset(s): {', '.join(invalid)}. Available options: {available}"
        )

    seen: set[str] = set()
    ordered: list[str] = []
    for name in requested:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _prepare_target_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset_name: str, output_root: Path, force: bool = False) -> Path:
    """Download a single dataset by name into *output_root*.

    Skips if the target directory already exists and is non-empty (unless *force*).
    Returns the path to the downloaded dataset directory.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "`kagglehub` is not installed. Please run `pip install -r requirements.txt`."
        ) from exc

    dataset_meta = DATASETS[dataset_name]
    target_dir = output_root / dataset_meta["target_dir"]

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        print(f"[skip] {dataset_name} already exists: {target_dir}")
        return target_dir

    _prepare_target_dir(target_dir, force=force)

    print(f"[download] {dataset_name} -> {target_dir}")
    try:
        cache_path = Path(kagglehub.dataset_download(dataset_meta["handle"]))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {dataset_name}. "
            "Check Kaggle credentials, network connectivity, and dataset access permissions."
        ) from exc

    if not cache_path.exists():
        raise RuntimeError(
            f"Cache path not found after downloading {dataset_name}: {cache_path}"
        )

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(cache_path, target_dir)
    print(f"[done] {dataset_name} saved to: {target_dir}")
    return target_dir
