"""Dataloader builders for processed wafer-map datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from ..config import load_yaml_config
from ..registry import DATASET_REGISTRY
from .transforms import (
    ClassAwareAugmentation,
    InjectFailureTypeIdx,
    WaferAugmentation,
    WaferRandomErasing,
    compose,
)


def load_dataloader_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML dataloader config file."""
    return load_yaml_config(config_path)


# ── Classification builders (train / eval) ───────────────────────────


def _build_class_balanced_sampler(
    dataset: Any,
    task_mode: str,
    alpha: float = 1.0,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that rebalances class frequencies.

    Per-sample weight is proportional to ``class_count ** (-alpha)``.

    * ``alpha=1.0`` – pure inverse-frequency (original behaviour, fully flat).
    * ``alpha=0.5`` – square-root smoothing (partial rebalancing).
    * ``alpha=0.0`` – no rebalancing (original class distribution).

    Values in ``(0, 1)`` give a smooth trade-off between under-representing
    the majority class and over-repeating rare classes.
    """
    from ..models.resnet import FAILURE_TYPE_TO_IDX

    # Unwrap Subset if present.
    inner = dataset.dataset if isinstance(dataset, Subset) else dataset
    index_df = inner.index_df

    if task_mode == "multiclass":
        labels = index_df["failure_type"].map(
            lambda ft: FAILURE_TYPE_TO_IDX.get(str(ft), 0),
        ).to_numpy(dtype=np.int64)
    else:
        labels = (~index_df["is_normal"]).astype(int).to_numpy(dtype=np.int64)

    # If wrapped in Subset, restrict to the subset indices.
    if isinstance(dataset, Subset):
        labels = labels[dataset.indices]

    class_counts = np.bincount(labels)
    class_weights = np.power(
        np.maximum(class_counts, 1).astype(np.float64), -alpha,
    )
    # Normalise so weights sum to 1 (avoids numerical issues).
    class_weights = class_weights / class_weights.sum()
    sample_weights = class_weights[labels]
    sample_weights_t = torch.from_numpy(sample_weights).double()

    num_samples = len(labels)
    effective_ratio = class_weights.max() / max(class_weights.min(), 1e-12)
    print(
        f"[class_balanced_sampler] {num_samples} samples, "
        f"{len(class_counts)} classes, alpha={alpha:.2f}, "
        f"max/min weight ratio={effective_ratio:.1f}x"
    )
    return WeightedRandomSampler(
        weights=sample_weights_t,
        num_samples=num_samples,
        replacement=True,
    )


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

    # ── Augmentation parameters ──────────────────────────────────────
    random_flip = bool(aug_cfg.get("random_flip", True))
    random_rotate90 = bool(aug_cfg.get("random_rotate90", True))
    random_translate_frac = float(aug_cfg.get("random_translate_frac", 0.0))
    random_scale_min = float(aug_cfg.get("random_scale_min", 1.0))
    random_scale_max = float(aug_cfg.get("random_scale_max", 1.0))
    has_spatial_aug = (
        random_flip
        or random_rotate90
        or random_translate_frac > 0
        or abs(random_scale_min - 1.0) > 1e-6
        or abs(random_scale_max - 1.0) > 1e-6
    )

    # Class-aware augmentation: stronger transforms for minority classes.
    use_class_aware = bool(aug_cfg.get("class_aware", False)) and include_meta

    # Global random erasing (applied to ALL classes, independent of
    # class-aware augmentation which has its own minority-only erasing).
    global_erasing_p = float(aug_cfg.get("random_erasing_p", 0.0))

    # Assemble per-split transforms.
    train_transforms: list = []

    if use_class_aware:
        minority_list = aug_cfg.get("minority_classes", None)
        minority_set = set(minority_list) if minority_list else None
        train_transforms.append(
            ClassAwareAugmentation(
                minority_classes=minority_set,
                random_flip=random_flip,
                random_rotate90=random_rotate90,
                random_translate_frac=random_translate_frac,
                random_scale_min=random_scale_min,
                random_scale_max=random_scale_max,
                minority_translate_frac=float(
                    aug_cfg.get("minority_translate_frac", 0.12)
                ),
                minority_scale_min=float(
                    aug_cfg.get("minority_scale_min", 0.90)
                ),
                minority_scale_max=float(
                    aug_cfg.get("minority_scale_max", 1.10)
                ),
                minority_erasing_p=float(
                    aug_cfg.get("minority_erasing_p", 0.3)
                ),
            )
        )
    elif has_spatial_aug:
        train_transforms.append(
            WaferAugmentation(
                random_flip=random_flip,
                random_rotate90=random_rotate90,
                random_translate_frac=random_translate_frac,
                random_scale_min=random_scale_min,
                random_scale_max=random_scale_max,
            )
        )

    if global_erasing_p > 0:
        train_transforms.append(WaferRandomErasing(p=global_erasing_p))

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

    # Optional class-balanced sampler for the training set.
    sampler_type = str(data_cfg.get("sampler", "")).strip().lower()
    sampler_alpha = float(data_cfg.get("sampler_alpha", 1.0))
    train_sampler = None
    shuffle_train = True
    if sampler_type == "class_balanced" and not smoke_test:
        train_sampler = _build_class_balanced_sampler(
            train_ds, task_mode, alpha=sampler_alpha,
        )
        shuffle_train = False  # mutually exclusive with sampler

    return {
        "train": DataLoader(
            train_ds, batch_size=bs, shuffle=shuffle_train,
            sampler=train_sampler,
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
