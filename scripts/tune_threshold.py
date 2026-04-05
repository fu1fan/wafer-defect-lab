#!/usr/bin/env python
"""Tune the binary classification threshold for optimal recall/F1 trade-off.

Given a trained model and its run summary, this script:
1. Runs inference to get class probabilities
2. Sweeps thresholds on the abnormal-class probability
3. Reports metrics at each threshold
4. Optionally re-evaluates with the best threshold

Examples
--------
    python scripts/tune_threshold.py \
        --run-summary outputs/some_run/run_summary.json

    python scripts/tune_threshold.py \
        --run-summary outputs/some_run/run_summary.json \
        --optimize-for recall --min-precision 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from waferlab.data.dataloaders import build_eval_dataloader
from waferlab.data.processed import load_data_config
from waferlab.engine.evaluator import evaluate
from waferlab.metrics.classification import compute_metrics, format_metrics
from waferlab.registry import MODEL_REGISTRY
from waferlab.runtime import load_run_summary, resolve_device, resolve_processed_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune binary threshold")
    p.add_argument("--run-summary", type=Path, required=True)
    p.add_argument("--split", default="Test", choices=["Training", "Test", "all"])
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--optimize-for", default="f1",
        choices=["recall", "f1", "macro_f1"],
        help="Primary metric to optimize the threshold for",
    )
    p.add_argument(
        "--min-precision", type=float, default=0.3,
        help="Minimum acceptable abnormal precision",
    )
    p.add_argument(
        "--thresholds", type=str, default=None,
        help="Comma-separated thresholds to evaluate (default: auto sweep)",
    )
    return p.parse_args()


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> list[dict]:
    """Evaluate metrics at each threshold on the abnormal-class probability."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    abnormal_prob = y_prob[:, 1]  # probability of class 1 (abnormal)
    results = []

    for thr in thresholds:
        y_pred = (abnormal_prob >= thr).astype(np.int64)
        metrics = compute_metrics(y_true, y_pred, class_names=class_names)

        ab = metrics["per_class"][1] if len(metrics["per_class"]) > 1 else {}
        results.append({
            "threshold": float(thr),
            "accuracy": metrics["accuracy"],
            "abnormal_precision": ab.get("precision", 0.0),
            "abnormal_recall": ab.get("recall", 0.0),
            "abnormal_f1": ab.get("f1", 0.0),
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        })

    return results


def main() -> int:
    args = parse_args()
    summary = load_run_summary(args.run_summary)

    model_cfg = dict(summary["model"])
    train_config_entry = summary.get("train_config", {})
    if isinstance(train_config_entry, dict):
        config = train_config_entry.get("snapshot") or {}
    else:
        config = {}

    task_mode = summary.get("task_mode", "binary")
    config["task_mode"] = task_mode

    checkpoint = Path(summary["checkpoints"]["best"])
    device = resolve_device(args.device)

    model = MODEL_REGISTRY.build(model_cfg.get("arch", "resnet18"), model_cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {checkpoint}")

    processed_root = resolve_processed_root(PROJECT_ROOT)
    eval_loader = build_eval_dataloader(
        config, processed_root=processed_root, split=args.split,
    )
    print(f"Eval samples: {len(eval_loader.dataset)} (split={args.split})")

    results = evaluate(model, eval_loader, device=device, task_mode=task_mode)

    class_names = ["normal", "abnormal"]
    if args.thresholds:
        thresholds = np.array([float(t) for t in args.thresholds.split(",")])
    else:
        thresholds = np.arange(0.05, 0.96, 0.025)

    sweep = sweep_thresholds(
        results["y_true"], results["y_prob"],
        thresholds=thresholds, class_names=class_names,
    )

    # Print sweep table.
    print(f"\n{'Thr':>6s} {'Acc':>7s} {'Ab.P':>7s} {'Ab.R':>7s} {'Ab.F1':>7s} {'MacF1':>7s}")
    print("-" * 48)
    for r in sweep:
        print(
            f"{r['threshold']:6.3f} {r['accuracy']:7.4f} "
            f"{r['abnormal_precision']:7.4f} {r['abnormal_recall']:7.4f} "
            f"{r['abnormal_f1']:7.4f} {r['macro_f1']:7.4f}"
        )

    # Find best threshold.
    valid = [r for r in sweep if r["abnormal_precision"] >= args.min_precision]
    if not valid:
        print(f"\nNo threshold meets min-precision={args.min_precision}.")
        valid = sweep  # fall back to all

    if args.optimize_for == "recall":
        best = max(valid, key=lambda r: (r["abnormal_recall"], r["abnormal_f1"]))
    elif args.optimize_for == "f1":
        best = max(valid, key=lambda r: (r["abnormal_f1"], r["abnormal_recall"]))
    else:
        best = max(valid, key=lambda r: (r["macro_f1"], r["abnormal_recall"]))

    print(f"\n=== Best threshold (optimize={args.optimize_for}, min_prec={args.min_precision}) ===")
    print(f"  threshold        = {best['threshold']:.3f}")
    print(f"  accuracy         = {best['accuracy']:.4f}")
    print(f"  abnormal_prec    = {best['abnormal_precision']:.4f}")
    print(f"  abnormal_recall  = {best['abnormal_recall']:.4f}")
    print(f"  abnormal_f1      = {best['abnormal_f1']:.4f}")
    print(f"  macro_f1         = {best['macro_f1']:.4f}")

    # Save sweep results.
    output_dir = Path(summary.get("output_dir", checkpoint.parent))
    sweep_path = output_dir / f"threshold_sweep_{args.split}.json"
    with sweep_path.open("w", encoding="utf-8") as f:
        json.dump({"sweep": sweep, "best": best}, f, indent=2)
    print(f"\nSaved to {sweep_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
