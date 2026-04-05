#!/usr/bin/env python
"""Evaluate a trained wafer-level classifier on WM-811K labeled data.

Examples
--------
    # Preferred: drive entirely from a run summary produced by training
    python scripts/eval_classifier.py \
        --run-summary outputs/wm811k_resnet_baseline/run_summary.json

    # Legacy: manual arguments (still supported)
    python scripts/eval_classifier.py \
        --checkpoint outputs/wm811k_resnet_baseline/best.pt
    python scripts/eval_classifier.py \
        --checkpoint outputs/wm811k_resnet_baseline/best.pt --task-mode multiclass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from waferlab.config import load_yaml_config
from waferlab.data.dataloaders import build_eval_dataloader
from waferlab.models.resnet import FAILURE_TYPE_NAMES, FAILURE_TYPE_TO_IDX
from waferlab.registry import MODEL_REGISTRY
from waferlab.runtime import load_run_summary, resolve_device, resolve_processed_root
from waferlab.engine.evaluator import evaluate
from waferlab.metrics.classification import compute_metrics, format_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate wafer-level classifier")
    p.add_argument(
        "--run-summary", type=Path, default=None,
        help="Path to run_summary.json produced by train_classifier.py. "
             "When provided, checkpoint / config / task-mode / output-dir "
             "are read from it automatically (CLI flags still override).",
    )
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to .pt checkpoint (overrides run-summary)")
    p.add_argument("--config", type=Path, default=None,
                   help="Training config YAML (only needed without --run-summary)")
    p.add_argument("--task-mode", choices=["binary", "multiclass"], default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to save evaluation results (default: checkpoint dir)")
    p.add_argument("--split", choices=["Training", "Test", "all"], default="Test",
                   help="Which split_label to evaluate on")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Resolve parameters: run_summary provides defaults, CLI overrides ──
    summary: dict = {}
    if args.run_summary is not None:
        summary = load_run_summary(args.run_summary)

    # Checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpt_best = summary.get("checkpoints", {}).get("best")
        if ckpt_best is not None:
            checkpoint = Path(ckpt_best)
    if checkpoint is None:
        raise SystemExit(
            "Error: --checkpoint is required (or provide --run-summary)."
        )

    # Task mode
    task_mode = args.task_mode
    if task_mode is None:
        task_mode = summary.get("task_mode")

    # Model config — prefer the snapshot stored in run_summary so that
    # arch / num_classes / in_channels are always consistent with the checkpoint.
    model_cfg: dict
    config: dict  # full training config for build_eval_dataloader
    if "model" in summary:
        model_cfg = dict(summary["model"])
        # train_config may be a plain path string (old format) or
        # {"path": ..., "snapshot": ...} (new format).
        train_config_entry = summary.get("train_config", {})
        if isinstance(train_config_entry, dict):
            config = train_config_entry.get("snapshot") or {}
        else:
            # Old format: fall back to reading the yaml from the path.
            config_path = Path(train_config_entry) if train_config_entry else None
            if config_path and config_path.exists():
                config = load_yaml_config(config_path)
            else:
                config = {}
    else:
        config_path = args.config or PROJECT_ROOT / "configs" / "modal" / "experiments" / "wm811k_resnet18_baseline.yaml"
        config = load_yaml_config(config_path)
        model_cfg = config.get("model", {})
        task_mode = task_mode or config.get("task_mode", "binary")

    task_mode = task_mode or "binary"

    # Ensure config carries the resolved task_mode (needed by build_eval_dataloader).
    config["task_mode"] = task_mode

    if task_mode == "binary":
        model_cfg.setdefault("num_classes", 2)
        class_names = ["normal", "abnormal"]
    else:
        model_cfg.setdefault("num_classes", len(FAILURE_TYPE_TO_IDX))
        class_names = list(FAILURE_TYPE_NAMES)

    # Output directory: CLI > run_summary output_dir > checkpoint parent
    output_dir: Path
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif "output_dir" in summary:
        output_dir = Path(summary["output_dir"])
    else:
        output_dir = checkpoint.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    # Load model.
    model = MODEL_REGISTRY.build(model_cfg.get("arch", "resnet18"), model_cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Task mode: {task_mode}  |  Classes: {model_cfg['num_classes']}")

    # Build eval dataloader.
    processed_root = resolve_processed_root(PROJECT_ROOT)
    eval_loader = build_eval_dataloader(
        config, processed_root=processed_root, split=args.split,
    )
    print(f"Eval samples: {len(eval_loader.dataset)}  (split={args.split})")

    # Run evaluation.
    results = evaluate(model, eval_loader, device=device, task_mode=task_mode)
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
