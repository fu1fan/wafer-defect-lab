#!/usr/bin/env python
"""Phase F – Classifier Re-Training (cRT) and decoupled training experiments.

Loads a pretrained checkpoint, freezes the backbone, and re-trains only
the classifier head with various strategies:
  - cRT-CE: class-balanced sampling + cross-entropy
  - cRT-Focal: class-balanced sampling + focal loss
  - cRT-Reset: reinitialize head, then CB + focal
  - cRT-LargerHead: 2-layer MLP head instead of single linear
  - cRT-LabelSmooth: cross-entropy with label smoothing

Also supports full-model fine-tuning with reduced LR for backbone.

Usage:
    python scripts/crt_retrain.py \
        --checkpoint outputs/phase_d/D1_gem_tuned/best.pt \
        --output-dir outputs/phase_f/F3_crt \
        --strategy crt_focal \
        --epochs 10 \
        --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from waferlab.config import load_yaml_config
from waferlab.data.dataloaders import build_classification_dataloaders
from waferlab.data.transforms import prepare_input, DEFAULT_NORM_SCALE
from waferlab.engine.evaluator import evaluate
from waferlab.engine.losses import FocalLoss
from waferlab.metrics.classification import compute_metrics, format_metrics
from waferlab.models.resnet import FAILURE_TYPE_NAMES, FAILURE_TYPE_TO_IDX
from waferlab.registry import MODEL_REGISTRY
from waferlab.runtime import load_run_summary, resolve_device, resolve_processed_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINORITY_CLASSES = ["Center", "Donut", "Loc", "Near-full", "Scratch"]


def compute_score(metrics: dict) -> float:
    acc = metrics["accuracy"]
    mf1 = metrics["macro_f1"]
    mr = metrics["macro_recall"]
    min_rs = [pc["recall"] for pc in metrics["per_class"] if pc["class_name"] in MINORITY_CLASSES]
    min_recall_mean = float(np.mean(min_rs)) if min_rs else 0.0
    return 0.40 * mf1 + 0.25 * mr + 0.20 * min_recall_mean + 0.15 * acc


def full_metrics(y_true, y_pred):
    m = compute_metrics(y_true, y_pred, class_names=list(FAILURE_TYPE_NAMES))
    m["score"] = compute_score(m)
    min_rs = [pc["recall"] for pc in m["per_class"] if pc["class_name"] in MINORITY_CLASSES]
    m["minority_recall_mean"] = float(np.mean(min_rs))
    return m


def freeze_backbone(model):
    """Freeze all parameters except the classification head (fc layer)."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen backbone: {trainable}/{total} trainable params "
          f"({trainable/total*100:.1f}%)")


def unfreeze_backbone_with_lr(model, head_lr, backbone_lr_ratio=0.1):
    """Unfreeze everything, return param groups with different LRs."""
    for p in model.parameters():
        p.requires_grad = True
    head_params = list(model.fc.parameters())
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": head_lr * backbone_lr_ratio},
        {"params": head_params, "lr": head_lr},
    ]


def reinit_head(model, strategy="kaiming"):
    """Re-initialize the classification head weights."""
    if strategy == "kaiming":
        nn.init.kaiming_normal_(model.fc.weight, mode="fan_out", nonlinearity="relu")
        if model.fc.bias is not None:
            nn.init.zeros_(model.fc.bias)
    elif strategy == "xavier":
        nn.init.xavier_uniform_(model.fc.weight)
        if model.fc.bias is not None:
            nn.init.zeros_(model.fc.bias)
    print(f"  Re-initialized head ({strategy})")


@torch.no_grad()
def evaluate_model(model, loader, device, task_mode="multiclass"):
    """Quick evaluation returning full metrics dict."""
    results = evaluate(model, loader, device=str(device), task_mode=task_mode)
    return full_metrics(results["y_true"], results["y_pred"])


def train_head(model, train_loader, val_loader, *, criterion, optimizer, scheduler,
               epochs, device, task_mode="multiclass", norm_scale=DEFAULT_NORM_SCALE,
               grad_clip=1.0, output_dir=None, score_name="score"):
    """Train loop optimized for head-only or full fine-tuning."""
    dev = torch.device(device)
    model = model.to(dev)
    best_score = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            x = prepare_input(batch, device=dev, target_channels=model.in_channels,
                              norm_scale=norm_scale)
            if task_mode == "binary":
                labels = batch["label"].to(dev)
            else:
                labels = batch["failure_type_idx"].to(dev)

            logits = model(x)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += x.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validate
        val_metrics = evaluate_model(model, val_loader, device, task_mode)
        val_score = val_metrics[score_name]

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_score": val_score,
            "val_acc": val_metrics["accuracy"],
            "val_mf1": val_metrics["macro_f1"],
            "val_mr": val_metrics["macro_recall"],
            "val_minr": val_metrics["minority_recall_mean"],
            "lr": optimizer.param_groups[-1]["lr"],
            "elapsed": elapsed,
        }
        history.append(record)

        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:>2d}/{epochs}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_score={val_score:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"mF1={val_metrics['macro_f1']:.4f}  mR={val_metrics['macro_recall']:.4f}  "
              f"minR={val_metrics['minority_recall_mean']:.4f}  "
              f"lr={record['lr']:.2e}  {elapsed:.1f}s"
              f"{'  ★' if is_best else ''}")

        # Save checkpoint
        if output_dir is not None:
            ckpt = {
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "best_val_score": best_score,
                "epoch": epoch,
                "history": history,
            }
            torch.save(ckpt, output_dir / "last.pt")
            if is_best:
                torch.save(ckpt, output_dir / "best.pt")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(dev)
    return history, best_score


def parse_args():
    p = argparse.ArgumentParser(description="cRT / decoupled training experiments")
    p.add_argument("--run-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--strategies", type=str, nargs="+",
                   default=["crt_ce", "crt_focal", "crt_reset_focal",
                            "crt_label_smooth", "crt_balanced_finetune"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()

    # Load config and model
    summary = load_run_summary(args.run_summary)
    model_cfg = dict(summary["model"])
    task_mode = summary.get("task_mode", "multiclass")
    checkpoint_path = Path(summary["checkpoints"]["best"])
    device = resolve_device(args.device)
    processed_root = resolve_processed_root(PROJECT_ROOT)

    train_config_entry = summary.get("train_config", {})
    config = train_config_entry.get("snapshot") or {}
    config["task_mode"] = task_mode

    # Build data loaders
    print("Building data loaders...")
    loaders = build_classification_dataloaders(config, processed_root=processed_root)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")

    # Class counts for losses
    from waferlab.models.resnet import FAILURE_TYPE_TO_IDX
    from waferlab.data.dataloaders import _build_class_balanced_sampler
    from torch.utils.data import Subset
    inner = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    index_df = inner.index_df
    labels = index_df["failure_type"].map(
        lambda ft: FAILURE_TYPE_TO_IDX.get(str(ft), 0)
    ).to_numpy(dtype=np.int64)
    class_counts = np.bincount(labels, minlength=9).astype(np.float64)
    print(f"  Class counts: {class_counts.astype(int).tolist()}")

    baseline_score = 0.8017
    all_results = {}

    strategies = args.strategies

    for strategy in strategies:
        print(f"\n{'='*90}")
        print(f"STRATEGY: {strategy}")
        print(f"{'='*90}")

        # Reload model from checkpoint each time
        model = MODEL_REGISTRY.build(model_cfg.get("arch", "resnet50_gem"), model_cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        out_dir = args.output_dir / strategy
        out_dir.mkdir(parents=True, exist_ok=True)

        lr = args.lr
        epochs = args.epochs

        if strategy == "crt_ce":
            # cRT: freeze backbone, retrain head with CE + CB sampling
            freeze_backbone(model)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=1e-3
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        elif strategy == "crt_focal":
            # cRT: freeze backbone, retrain head with Focal + CB sampling
            freeze_backbone(model)
            criterion = FocalLoss(gamma=1.5, reduction="mean")
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=1e-3
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        elif strategy == "crt_reset_focal":
            # cRT with head re-initialization
            freeze_backbone(model)
            reinit_head(model, "kaiming")
            criterion = FocalLoss(gamma=1.5, reduction="mean")
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=1e-3
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        elif strategy == "crt_label_smooth":
            # cRT with label smoothing
            freeze_backbone(model)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=1e-3
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        elif strategy == "crt_balanced_finetune":
            # Balanced fine-tuning: unfreeze all, lower LR for backbone
            param_groups = unfreeze_backbone_with_lr(model, lr, backbone_lr_ratio=0.01)
            criterion = FocalLoss(gamma=1.5, reduction="mean")
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        else:
            print(f"  Unknown strategy: {strategy}, skipping")
            continue

        # Train
        history, best_score = train_head(
            model, train_loader, val_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            epochs=epochs, device=str(device), task_mode=task_mode,
            output_dir=out_dir, score_name="score",
        )

        # Final evaluation
        final_metrics = evaluate_model(model, val_loader, str(device), task_mode)
        delta = final_metrics["score"] - baseline_score
        sign = "+" if delta >= 0 else ""

        print(f"\n  RESULT: score={final_metrics['score']:.4f} ({sign}{delta:.4f})")
        print(f"  acc={final_metrics['accuracy']:.4f}  mF1={final_metrics['macro_f1']:.4f}  "
              f"mR={final_metrics['macro_recall']:.4f}  minR={final_metrics['minority_recall_mean']:.4f}")
        print(f"\n  Per-class recalls:")
        for pc in final_metrics["per_class"]:
            print(f"    {pc['class_name']:<14s} P={pc['precision']:.4f}  R={pc['recall']:.4f}  F1={pc['f1']:.4f}")

        all_results[strategy] = {
            "score": final_metrics["score"],
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "macro_recall": final_metrics["macro_recall"],
            "minority_recall_mean": final_metrics["minority_recall_mean"],
            "per_class_recall": {pc["class_name"]: pc["recall"] for pc in final_metrics["per_class"]},
            "best_epoch_score": best_score,
            "epochs": epochs,
            "lr": lr,
        }

        # Save per-strategy results
        with (out_dir / "results.json").open("w") as f:
            json.dump(all_results[strategy], f, indent=2)

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY: All strategies vs baseline")
    print(f"{'='*90}")
    print(f"  Baseline:  score={baseline_score:.4f}")
    print()
    print(f"  {'Strategy':<30s} {'Score':>7s} {'Delta':>7s} {'Acc':>7s} {'mF1':>7s} {'mR':>7s} {'minR':>7s}")
    print("  " + "-" * 85)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True):
        delta = r["score"] - baseline_score
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<30s} {r['score']:7.4f} {sign}{delta:6.4f} "
              f"{r['accuracy']:7.4f} {r['macro_f1']:7.4f} {r['macro_recall']:7.4f} "
              f"{r['minority_recall_mean']:7.4f}")

    # Save combined results
    combined_path = args.output_dir / "all_results.json"
    with combined_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
