#!/usr/bin/env python
"""Train a wafer-level ResNet classifier on WM-811K labeled data.

Examples
--------
    # Binary normal/abnormal baseline (default)
    python scripts/train_classifier.py

    # Multi-class failure-type classification
    python scripts/train_classifier.py --config configs/train/wm811k_resnet_baseline.yaml \
        --task-mode multiclass

    # Quick smoke test (1 epoch, small subset)
    python scripts/train_classifier.py --epochs 1 --smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sized
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from waferlab.data.datasets import WM811KProcessedDataset
from waferlab.models.classifier import FAILURE_TYPE_TO_IDX, build_classifier
from waferlab.runtime import resolve_device, resolve_output_root, resolve_processed_root
from waferlab.engine.trainer import Trainer


# ── Augmentation transforms ─────────────────────────────────────────

class WaferAugmentation:
    """Simple spatial augmentations safe for wafer maps."""

    def __init__(self, *, random_flip: bool = True, random_rotate90: bool = True) -> None:
        self.random_flip = random_flip
        self.random_rotate90 = random_rotate90

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]  # [1, H, W]

        # Decide transforms once so image and masks stay consistent.
        do_hflip = self.random_flip and torch.rand(1).item() > 0.5
        do_vflip = self.random_flip and torch.rand(1).item() > 0.5
        k = int(torch.randint(0, 4, (1,)).item()) if self.random_rotate90 else 0

        img = self._apply(img, do_hflip, do_vflip, k)
        sample["image"] = img

        for mask_key in ("wafer_mask", "defect_mask"):
            if mask_key in sample:
                sample[mask_key] = self._apply(sample[mask_key], do_hflip, do_vflip, k)

        return sample

    @staticmethod
    def _apply(t: torch.Tensor, hflip: bool, vflip: bool, k: int) -> torch.Tensor:
        if hflip:
            t = t.flip(-1)
        if vflip:
            t = t.flip(-2)
        if k > 0:
            t = torch.rot90(t, k, dims=(-2, -1))
        return t


# ── Multiclass label injection ───────────────────────────────────────

class InjectFailureTypeIdx:
    """Add ``failure_type_idx`` field for multi-class mode."""

    def __call__(self, sample: dict) -> dict:
        meta = sample.get("metadata", {})
        ft = str(meta.get("failure_type", "none"))
        sample["failure_type_idx"] = FAILURE_TYPE_TO_IDX.get(ft, 0)
        return sample


# ── Helpers ──────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_datasets(
    config: dict,
    task_mode: str,
    processed_root: Path,
    smoke_test: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build train & val dataloaders from WM-811K labeled processed data."""
    aug_cfg = config.get("data", {}).get("augment", {})
    data_cfg = config.get("data", {})

    transforms_train = []
    if aug_cfg.get("random_flip", True) or aug_cfg.get("random_rotate90", True):
        transforms_train.append(
            WaferAugmentation(
                random_flip=aug_cfg.get("random_flip", True),
                random_rotate90=aug_cfg.get("random_rotate90", True),
            )
        )

    include_meta = task_mode == "multiclass"
    if task_mode == "multiclass":
        transforms_train.append(InjectFailureTypeIdx())

    def compose(fns):
        def _apply(sample):
            for fn in fns:
                sample = fn(sample)
            return sample
        return _apply if fns else None

    # Train: WM-811K labeled, split_label == "Training"
    train_ds = WM811KProcessedDataset(
        processed_root,
        subset="labeled",
        transform=compose(transforms_train),
        include_metadata=include_meta,
        return_masks=False,
        return_float=True,
        filters={"split_label": "Training"},
    )

    # Val: WM-811K labeled, split_label == "Test"
    val_transforms = [InjectFailureTypeIdx()] if task_mode == "multiclass" else []
    val_ds = WM811KProcessedDataset(
        processed_root,
        subset="labeled",
        transform=compose(val_transforms),
        include_metadata=include_meta,
        return_masks=False,
        return_float=True,
        filters={"split_label": "Test"},
    )

    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(256, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(256, len(val_ds)))))

    bs = int(data_cfg.get("batch_size", 64))
    nw = int(data_cfg.get("num_workers", 4))
    pin = bool(data_cfg.get("pin_memory", True))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin, drop_last=False,
    )
    return train_loader, val_loader


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train wafer-level classifier")
    p.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "wm811k_resnet_baseline.yaml",
    )
    p.add_argument("--task-mode", choices=["binary", "multiclass"], default=None,
                   help="Override task_mode in config")
    p.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    p.add_argument("--lr", type=float, default=None, help="Learning-rate override")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to save checkpoints (default: $WAFERLAB_OUTPUT_ROOT/...)")
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from a saved checkpoint (.pt)",
    )
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 1 epoch on a tiny subset for quick validation")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    task_mode = args.task_mode or config.get("task_mode", "binary")
    device = resolve_device(args.device)

    # CLI overrides.
    train_cfg = config.setdefault("train", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.smoke_test:
        train_cfg["epochs"] = 1
        train_cfg["log_interval"] = 10

    default_output_dir = resolve_output_root(PROJECT_ROOT) / "wm811k_resnet_baseline"
    output_dir = args.output_dir or default_output_dir

    # Determine num_classes from task mode.
    model_cfg = config.get("model", {})
    if task_mode == "binary":
        model_cfg["num_classes"] = 2
    else:
        model_cfg["num_classes"] = len(FAILURE_TYPE_TO_IDX)

    print(f"Task mode : {task_mode}")
    print(f"Model     : {model_cfg.get('arch', 'resnet18')}")
    print(f"Classes   : {model_cfg['num_classes']}")
    print(f"Epochs    : {train_cfg.get('epochs', 30)}")
    print(f"Device    : {device}")
    print(f"Output    : {output_dir}")
    print()

    # Build components.
    processed_root = resolve_processed_root(PROJECT_ROOT)
    train_loader, val_loader = _build_datasets(
        config, task_mode, processed_root, smoke_test=args.smoke_test,
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    print(f"Train samples: {len(train_dataset if isinstance(train_dataset, Sized) else [])}")
    print(f"Val samples  : {len(val_dataset if isinstance(val_dataset, Sized) else [])}")

    model = build_classifier(model_cfg)
    trainer = Trainer(
        model, train_loader, val_loader, train_cfg,
        device=device,
        output_dir=output_dir,
        task_mode=task_mode,
    )

    if args.resume_from is not None:
        ckpt_path = args.resume_from.resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)
        if trainer.start_epoch > trainer.epochs:
            print(
                f"Checkpoint already contains {len(trainer.history)} epochs. "
                f"Configured epochs={trainer.epochs}, nothing left to run."
            )
            history = trainer.history
            history_path = output_dir / "history.json"
            with history_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            print(f"\nTraining history saved to {history_path}")
            print(f"Best val accuracy: {trainer.best_val_acc:.4f}")
            return 0

        print(
            f"Resuming from {ckpt_path} at epoch "
            f"{trainer.start_epoch}/{trainer.epochs}"
        )

    history = trainer.fit()

    # Save training history as JSON for later analysis.
    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best val accuracy: {trainer.best_val_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
