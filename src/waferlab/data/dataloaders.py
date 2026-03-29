"""Dataloader builders for processed wafer-map datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from .datasets import WM811KProcessedDataset


def load_dataloader_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML dataloader config file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Dataloader config must define a mapping: {path}")
    return config


def build_dataloaders(
    config: dict[str, Any],
    *,
    processed_root: str | Path = "data/processed",
) -> dict[str, DataLoader]:
    """Build one or more dataloaders from config."""
    dataset_name = str(config.get("dataset_name", "")).strip().lower()
    if dataset_name != "wm811k":
        raise NotImplementedError(
            f"Dataloader builder for dataset '{dataset_name}' is not implemented yet."
        )

    loader_configs = config.get("loaders", {})
    if not isinstance(loader_configs, dict) or not loader_configs:
        raise ValueError("`loaders` must be a non-empty mapping in the dataloader config.")

    dataloaders: dict[str, DataLoader] = {}
    for loader_name, loader_config in loader_configs.items():
        if not isinstance(loader_config, dict):
            raise ValueError(f"Loader config for '{loader_name}' must be a mapping.")
        dataloaders[loader_name] = build_wm811k_dataloader(
            loader_config,
            processed_root=processed_root,
        )
    return dataloaders


def build_wm811k_dataloader(
    loader_config: dict[str, Any],
    *,
    processed_root: str | Path = "data/processed",
) -> DataLoader:
    """Build a WM-811K dataloader from a single loader config."""
    subset = str(loader_config.get("subset", "labeled"))
    filters = loader_config.get("filters", {})
    if filters is None:
        filters = {}
    if not isinstance(filters, dict):
        raise ValueError("`filters` must be a mapping.")

    dataset = WM811KProcessedDataset(
        processed_root,
        subset=subset,
        include_metadata=bool(loader_config.get("include_metadata", False)),
        return_masks=bool(loader_config.get("return_masks", True)),
        return_float=bool(loader_config.get("return_float", True)),
        filters=filters,
    )

    batch_size = int(loader_config.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(loader_config.get("shuffle", False)),
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
        drop_last=bool(loader_config.get("drop_last", False)),
    )
