"""Dataloader builders for processed wafer-map datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Subset

from ..config import load_yaml_config
from ..registry import DATASET_REGISTRY
from .transforms import WaferAugmentation, InjectFailureTypeIdx, compose


def load_dataloader_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML dataloader config file."""
    return load_yaml_config(config_path)


# ── Classification builders (train / eval) ───────────────────────────


def build_classification_dataloaders(
    config: dict[str, Any],
    *,
    processed_root: str | Path = "data/processed",
    smoke_test: bool = False,
) -> dict[str, DataLoader]:
    """Build train & val dataloaders for classification from a unified config.

    Parameters
    ----------
    config : dict
        Full training config containing ``task_mode``, ``data`` (with
        ``batch_size``, ``num_workers``, ``pin_memory``, ``augment``,
        and optionally ``dataset_name``).
    processed_root : str or Path
        Root of the processed data directory.
    smoke_test : bool
        If True, subset to 256 samples per split for quick testing.

    Returns
    -------
    dict with ``"train"`` and ``"val"`` DataLoaders.
    """
    task_mode = str(config.get("task_mode", "binary"))
    data_cfg = config.get("data", {})
    aug_cfg = data_cfg.get("augment", {})
    dataset_name = str(data_cfg.get("dataset_name", "wm811k")).strip().lower()
    dataset_config = data_cfg.get("dataset_config")

    include_meta = task_mode == "multiclass"

    # Assemble per-split transforms.
    train_transforms: list = []
    if aug_cfg.get("random_flip", True) or aug_cfg.get("random_rotate90", True):
        train_transforms.append(
            WaferAugmentation(
                random_flip=aug_cfg.get("random_flip", True),
                random_rotate90=aug_cfg.get("random_rotate90", True),
            )
        )
    if task_mode == "multiclass":
        train_transforms.append(InjectFailureTypeIdx())

    val_transforms: list = []
    if task_mode == "multiclass":
        val_transforms.append(InjectFailureTypeIdx())

    # Build datasets via registry.
    train_ds = DATASET_REGISTRY.build(dataset_name, {
        "processed_root": processed_root,
        "subset": "labeled",
        "transform": compose(train_transforms),
        "include_metadata": include_meta,
        "return_masks": False,
        "return_float": True,
        "filters": {"split_label": "Training"},
        "dataset_config": dataset_config,
    })

    val_ds = DATASET_REGISTRY.build(dataset_name, {
        "processed_root": processed_root,
        "subset": "labeled",
        "transform": compose(val_transforms),
        "include_metadata": include_meta,
        "return_masks": False,
        "return_float": True,
        "filters": {"split_label": "Test"},
        "dataset_config": dataset_config,
    })

    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(256, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(256, len(val_ds)))))

    bs = int(data_cfg.get("batch_size", 64))
    nw = int(data_cfg.get("num_workers", 4))
    pin = bool(data_cfg.get("pin_memory", True))

    return {
        "train": DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pin, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=pin, drop_last=False,
        ),
    }


def build_eval_dataloader(
    config: dict[str, Any],
    *,
    processed_root: str | Path = "data/processed",
    split: str = "Test",
) -> DataLoader:
    """Build a single evaluation dataloader for classification.

    Parameters
    ----------
    config : dict
        Full training/eval config containing ``task_mode`` and ``data``.
    processed_root : str or Path
        Root of the processed data directory.
    split : str
        ``"Training"``, ``"Test"``, or ``"all"`` (no split filter).

    Returns
    -------
    DataLoader for the requested split.
    """
    task_mode = str(config.get("task_mode", "binary"))
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset_name", "wm811k")).strip().lower()
    dataset_config = data_cfg.get("dataset_config")

    include_meta = task_mode == "multiclass"
    transforms: list = []
    if task_mode == "multiclass":
        transforms.append(InjectFailureTypeIdx())

    filters: dict[str, Any] = {}
    if split != "all":
        filters["split_label"] = split

    eval_ds = DATASET_REGISTRY.build(dataset_name, {
        "processed_root": processed_root,
        "subset": "labeled",
        "transform": compose(transforms),
        "include_metadata": include_meta,
        "return_masks": False,
        "return_float": True,
        "filters": filters,
        "dataset_config": dataset_config,
    })

    return DataLoader(
        eval_ds,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )


# ── Generic config-driven builder (anomaly pipelines, etc.) ─────────


def build_dataloaders(
    config: dict[str, Any],
    *,
    processed_root: str | Path = "data/processed",
) -> dict[str, DataLoader]:
    """Build one or more dataloaders from a config with ``dataset_name`` + ``loaders``."""
    dataset_name = str(config.get("dataset_name", "")).strip().lower()
    dataset_config = config.get("dataset_config")

    loader_configs = config.get("loaders", {})
    if not isinstance(loader_configs, dict) or not loader_configs:
        raise ValueError("`loaders` must be a non-empty mapping in the dataloader config.")

    dataloaders: dict[str, DataLoader] = {}
    for loader_name, loader_config in loader_configs.items():
        if not isinstance(loader_config, dict):
            raise ValueError(f"Loader config for '{loader_name}' must be a mapping.")

        filters = loader_config.get("filters", {})
        if filters is None:
            filters = {}
        if not isinstance(filters, dict):
            raise ValueError(f"`filters` in '{loader_name}' must be a mapping.")

        ds = DATASET_REGISTRY.build(dataset_name, {
            "processed_root": processed_root,
            "subset": str(loader_config.get("subset", "labeled")),
            "include_metadata": bool(loader_config.get("include_metadata", False)),
            "return_masks": bool(loader_config.get("return_masks", True)),
            "return_float": bool(loader_config.get("return_float", True)),
            "filters": filters,
            "dataset_config": loader_config.get("dataset_config", dataset_config),
        })

        batch_size = int(loader_config.get("batch_size", 64))
        if batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")

        dataloaders[loader_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=bool(loader_config.get("shuffle", False)),
            num_workers=int(loader_config.get("num_workers", 0)),
            pin_memory=bool(loader_config.get("pin_memory", False)),
            drop_last=bool(loader_config.get("drop_last", False)),
        )
    return dataloaders


def build_wm811k_dataloader(
    loader_config: dict[str, Any],
    *,
    processed_root: str | Path = "data/processed",
) -> DataLoader:
    """Build a WM-811K dataloader from a single loader config.

    Kept for backward compatibility; delegates to :data:`DATASET_REGISTRY`.
    """
    filters = loader_config.get("filters", {})
    if filters is None:
        filters = {}
    if not isinstance(filters, dict):
        raise ValueError("`filters` must be a mapping.")

    ds = DATASET_REGISTRY.build("wm811k", {
        "processed_root": processed_root,
        "subset": str(loader_config.get("subset", "labeled")),
        "include_metadata": bool(loader_config.get("include_metadata", False)),
        "return_masks": bool(loader_config.get("return_masks", True)),
        "return_float": bool(loader_config.get("return_float", True)),
        "filters": filters,
        "dataset_config": loader_config.get("dataset_config"),
    })

    batch_size = int(loader_config.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(loader_config.get("shuffle", False)),
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
        drop_last=bool(loader_config.get("drop_last", False)),
    )
