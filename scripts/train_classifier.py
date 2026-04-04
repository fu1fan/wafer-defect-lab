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
from collections.abc import Sized
from datetime import datetime
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, Subset

from waferlab.data.dataloaders import build_classification_dataloaders
from waferlab.models.resnet import FAILURE_TYPE_TO_IDX
from waferlab.registry import MODEL_REGISTRY
from waferlab.runtime import resolve_device, resolve_output_root, resolve_processed_root
from waferlab.engine.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Helpers ──────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train wafer-level classifier")
    p.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "wm811k.yaml",
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

    # Determine num_classes from task mode.
    model_cfg = config.get("model", {})
    if task_mode == "binary":
        model_cfg["num_classes"] = 2
    else:
        model_cfg["num_classes"] = len(FAILURE_TYPE_TO_IDX)

    arch = model_cfg.get("arch", "resnet18")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = resolve_output_root(PROJECT_ROOT) / f"{arch}_{task_mode}_{timestamp}"
    output_dir = args.output_dir or default_output_dir

    print(f"Task mode : {task_mode}")
    print(f"Model     : {model_cfg.get('arch', 'resnet18')}")
    print(f"Classes   : {model_cfg['num_classes']}")
    print(f"Epochs    : {train_cfg.get('epochs', 30)}")
    print(f"Device    : {device}")
    print(f"Output    : {output_dir}")
    print()

    # Build components.
    processed_root = resolve_processed_root(PROJECT_ROOT)
    loaders = build_classification_dataloaders(
        config, processed_root=processed_root, smoke_test=args.smoke_test,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    print(f"Train samples: {len(train_dataset if isinstance(train_dataset, Sized) else [])}")
    print(f"Val samples  : {len(val_dataset if isinstance(val_dataset, Sized) else [])}")

    model = MODEL_REGISTRY.build(model_cfg.get("arch", "resnet18"), model_cfg)
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
