#!/usr/bin/env python
"""Evaluate a trained wafer-level classifier on WM-811K labeled data.

Examples
--------
    python scripts/eval_classifier.py --checkpoint outputs/wm811k_resnet_baseline/best.pt
    python scripts/eval_classifier.py --checkpoint outputs/wm811k_resnet_baseline/best.pt \
        --task-mode multiclass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import numpy as np
import torch
from torch.utils.data import DataLoader

from waferlab.data.datasets import WM811KProcessedDataset
from waferlab.models.classifier import (
    FAILURE_TYPE_NAMES,
    FAILURE_TYPE_TO_IDX,
    build_classifier,
)
from waferlab.engine.evaluator import evaluate
from waferlab.metrics.classification import compute_metrics, format_metrics


class InjectFailureTypeIdx:
    def __call__(self, sample: dict) -> dict:
        meta = sample.get("metadata", {})
        ft = str(meta.get("failure_type", "none"))
        sample["failure_type_idx"] = FAILURE_TYPE_TO_IDX.get(ft, 0)
        return sample


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate wafer-level classifier")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "wm811k_resnet_baseline.yaml",
    )
    p.add_argument("--task-mode", choices=["binary", "multiclass"], default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to save evaluation results (default: checkpoint dir)")
    p.add_argument("--split", choices=["Training", "Test", "all"], default="Test",
                   help="Which split_label to evaluate on")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    task_mode = args.task_mode or config.get("task_mode", "binary")

    model_cfg = config.get("model", {})
    if task_mode == "binary":
        model_cfg["num_classes"] = 2
        class_names = ["normal", "abnormal"]
    else:
        model_cfg["num_classes"] = len(FAILURE_TYPE_TO_IDX)
        class_names = FAILURE_TYPE_NAMES

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    model = build_classifier(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Task mode: {task_mode}  |  Classes: {model_cfg['num_classes']}")

    # Build eval dataset.
    processed_root = PROJECT_ROOT / "data" / "processed"
    include_meta = task_mode == "multiclass"
    transforms = [InjectFailureTypeIdx()] if task_mode == "multiclass" else None

    def compose(fns):
        if not fns:
            return None
        def _apply(s):
            for fn in fns:
                s = fn(s)
            return s
        return _apply

    filters: dict = {}
    if args.split != "all":
        filters["split_label"] = args.split

    eval_ds = WM811KProcessedDataset(
        processed_root,
        subset="labeled",
        transform=compose(transforms),
        include_metadata=include_meta,
        return_masks=False,
        return_float=True,
        filters=filters,
    )

    data_cfg = config.get("data", {})
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    print(f"Eval samples: {len(eval_ds)}  (split={args.split})")

    # Run evaluation.
    results = evaluate(model, eval_loader, device=args.device, task_mode=task_mode)
    metrics = compute_metrics(results["y_true"], results["y_pred"], class_names=class_names)

    print("\n" + format_metrics(metrics))
    print()

    # Save results.
    save_metrics = {
        k: v for k, v in metrics.items() if k != "confusion_matrix"
    }
    save_metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    metrics_path = output_dir / f"eval_metrics_{args.split}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(save_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
