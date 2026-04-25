#!/usr/bin/env python
"""Phase G – cRT hyperparameter sweep with logit adjustment stacking.

Runs crt_focal at multiple LR / gamma combos, then sweeps logit adjustment
tau on each checkpoint to find the best stacked score.

Usage:
    python scripts/crt_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferlab.data.dataloaders import build_classification_dataloaders
from waferlab.data.transforms import prepare_input, DEFAULT_NORM_SCALE
from waferlab.engine.evaluator import evaluate
from waferlab.engine.losses import FocalLoss
from waferlab.metrics.classification import compute_metrics, format_metrics
from waferlab.models.resnet import FAILURE_TYPE_NAMES, FAILURE_TYPE_TO_IDX
from waferlab.models.resnet50_variants import WaferClassifierGeM
from waferlab.runtime import resolve_device, resolve_processed_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINORITY_CLASSES = ["Center", "Donut", "Loc", "Near-full", "Scratch"]
MINORITY_IDX = [1, 2, 5, 6, 8]
CLASS_NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
               "Loc", "Near-full", "Random", "Scratch"]

TRAIN_COUNTS = np.array([36730, 3462, 409, 2417, 8554, 1620, 54, 609, 500])
LOG_PRIOR = np.log(TRAIN_COUNTS / TRAIN_COUNTS.sum())


def compute_score_from_arrays(y_true, y_pred, num_classes=9):
    acc = float((y_true == y_pred).mean())
    per_class_r, per_class_f1 = [], []
    per_class_detail = {}
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class_r.append(r)
        per_class_f1.append(f1)
        per_class_detail[CLASS_NAMES[c]] = {"precision": p, "recall": r, "f1": f1}
    mr = float(np.mean(per_class_r))
    mf1 = float(np.mean(per_class_f1))
    min_r = float(np.mean([per_class_r[i] for i in MINORITY_IDX]))
    score = 0.40 * mf1 + 0.25 * mr + 0.20 * min_r + 0.15 * acc
    return {
        "score": score, "accuracy": acc, "macro_f1": mf1,
        "macro_recall": mr, "minority_recall_mean": min_r,
        "per_class_recall": {CLASS_NAMES[c]: per_class_r[c] for c in range(num_classes)},
    }


def full_metrics(y_true, y_pred):
    """Compute all metrics from arrays."""
    m = compute_metrics(y_true, y_pred, class_names=FAILURE_TYPE_NAMES)
    acc = m["accuracy"]
    mf1 = m["macro_f1"]
    mr = m["macro_recall"]
    min_rs = [pc["recall"] for pc in m["per_class"]
              if pc["class_name"] in MINORITY_CLASSES]
    min_recall_mean = float(np.mean(min_rs)) if min_rs else 0.0
    m["score"] = 0.40 * mf1 + 0.25 * mr + 0.20 * min_recall_mean + 0.15 * acc
    m["minority_recall_mean"] = min_recall_mean
    return m


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


@torch.no_grad()
def evaluate_model(model, loader, device, task_mode="multiclass"):
    results = evaluate(model, loader, device=str(device), task_mode=task_mode)
    return full_metrics(results["y_true"], results["y_pred"])


@torch.no_grad()
def extract_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x = prepare_input(batch, device=device, target_channels=1,
                          norm_scale=DEFAULT_NORM_SCALE)
        labels = batch["failure_type_idx"]
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def logit_adj_sweep(logits, labels, tau_range=np.arange(-0.3, 0.31, 0.01)):
    """Sweep logit adjustment tau, return best tau and metrics."""
    best_score, best_tau = -1.0, 0.0
    for tau in tau_range:
        adjusted = logits - tau * LOG_PRIOR
        preds = adjusted.argmax(axis=1)
        m = compute_score_from_arrays(labels, preds)
        if m["score"] > best_score:
            best_score = m["score"]
            best_tau = float(tau)
    # Get full metrics at best tau
    adjusted = logits - best_tau * LOG_PRIOR
    preds = adjusted.argmax(axis=1)
    best_metrics = compute_score_from_arrays(labels, preds)
    return best_tau, best_metrics


def train_crt(model, train_loader, val_loader, *, lr, focal_gamma, epochs,
              device, output_dir):
    """Train cRT with given hyperparameters. Returns best checkpoint metrics."""
    dev = torch.device(device)
    freeze_backbone(model)
    model = model.to(dev)

    criterion = FocalLoss(gamma=focal_gamma)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            x = prepare_input(batch, device=dev, target_channels=1,
                              norm_scale=DEFAULT_NORM_SCALE)
            labels = batch["failure_type_idx"].to(dev)
            logits = model(x)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Evaluate
        val_metrics = evaluate_model(model, val_loader, dev)
        score = val_metrics["score"]

        print(f"  Ep {epoch+1:2d}/{epochs}: "
              f"loss={total_loss/total:.4f} train_acc={train_acc:.4f} | "
              f"score={score:.4f} acc={val_metrics['accuracy']:.4f} "
              f"mF1={val_metrics['macro_f1']:.4f} mR={val_metrics['macro_recall']:.4f} "
              f"minR={val_metrics['minority_recall_mean']:.4f}")

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best
    if output_dir and best_state:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state, "epoch": best_epoch,
                     "score": best_score}, output_dir / "best.pt")

    # Load best for logit extraction
    model.load_state_dict(best_state)
    return model, best_score, best_epoch


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Phase G cRT sweep")
    p.add_argument(
        "--base-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase_d" / "D1_gem_tuned" / "best.pt",
        help="Path to the Phase-D best.pt checkpoint to warm-start cRT from.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase_g",
        help="Directory under which per-experiment subdirs will be created.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device("auto")
    processed_root = resolve_processed_root(PROJECT_ROOT)
    base_checkpoint = args.base_checkpoint
    output_base = args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    # Load config for dataloaders
    summary = json.load(open(base_checkpoint.parent / "run_summary.json"))
    config = summary.get("train_config", {})
    config["task_mode"] = "multiclass"

    print("Building data loaders...")
    loaders = build_classification_dataloaders(config, processed_root=processed_root)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")

    # Rebuild train_loader with CB sampler
    from waferlab.data.dataloaders import _build_class_balanced_sampler
    from torch.utils.data import DataLoader, Subset
    sampler = _build_class_balanced_sampler(train_loader.dataset, "multiclass")
    cb_train_loader = DataLoader(
        train_loader.dataset, batch_size=64, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )

    # ─── Experiment configurations ───────────────────────────────────
    experiments = [
        # G1: LR sweep (focal gamma=1.5 fixed)
        {"name": "G1_lr5e-4", "lr": 5e-4, "gamma": 1.5, "epochs": 10},
        {"name": "G1_lr2e-3", "lr": 2e-3, "gamma": 1.5, "epochs": 10},
        {"name": "G1_lr5e-3", "lr": 5e-3, "gamma": 1.5, "epochs": 10},
        # G2: Gamma sweep (LR=1e-3 fixed as reference)
        {"name": "G2_gamma0.5", "lr": 1e-3, "gamma": 0.5, "epochs": 10},
        {"name": "G2_gamma1.0", "lr": 1e-3, "gamma": 1.0, "epochs": 10},
        {"name": "G2_gamma2.0", "lr": 1e-3, "gamma": 2.0, "epochs": 10},
    ]

    all_results = {}
    current_best_score = 0.8113  # crt_focal + LogitAdj(tau=0.11)

    for exp in experiments:
        name = exp["name"]
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {name} (LR={exp['lr']}, gamma={exp['gamma']})")
        print(f"{'='*80}")

        # Load fresh model from baseline
        model = WaferClassifierGeM(num_classes=9, pretrained=False, dropout=0.2)
        ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        exp_dir = output_base / name
        t0 = time.time()

        # Train cRT
        model, crt_score, best_epoch = train_crt(
            model, cb_train_loader, val_loader,
            lr=exp["lr"], focal_gamma=exp["gamma"],
            epochs=exp["epochs"], device=device,
            output_dir=exp_dir
        )
        train_time = time.time() - t0

        print(f"\n  cRT best: score={crt_score:.4f} at epoch {best_epoch} "
              f"({train_time:.0f}s)")

        # Extract logits and sweep logit adjustment
        logits, lab = extract_logits(model, val_loader, torch.device(device))
        best_tau, stacked_metrics = logit_adj_sweep(logits, lab)
        stacked_score = stacked_metrics["score"]

        # Also get base metrics (tau=0)
        base_preds = logits.argmax(axis=1)
        base_metrics = compute_score_from_arrays(lab, base_preds)

        print(f"  + LogitAdj(tau={best_tau:.2f}): stacked_score={stacked_score:.4f} "
              f"(Δ from baseline: {stacked_score - 0.8017:+.4f})")

        marker = " ★ NEW BEST!" if stacked_score > current_best_score else ""
        if stacked_score > current_best_score:
            current_best_score = stacked_score
        print(f"  vs current best 0.8113: {stacked_score - 0.8113:+.4f}{marker}")

        # Per-class recall for key classes
        pcr = stacked_metrics["per_class_recall"]
        print(f"  Per-class recall (stacked): "
              f"Center={pcr['Center']:.4f} Loc={pcr['Loc']:.4f} "
              f"Scratch={pcr['Scratch']:.4f} Edge-Ring={pcr['Edge-Ring']:.4f}")

        all_results[name] = {
            "lr": exp["lr"],
            "focal_gamma": exp["gamma"],
            "crt_score": round(crt_score, 4),
            "crt_best_epoch": best_epoch,
            "best_logit_adj_tau": round(best_tau, 2),
            "stacked_score": round(stacked_score, 4),
            "stacked_metrics": {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in stacked_metrics.items()},
            "base_metrics": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in base_metrics.items()},
            "train_time_s": round(train_time, 1),
        }

        # Save logits for reproducibility
        np.savez_compressed(exp_dir / "logits.npz", logits=logits, labels=lab)

    # ─── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Name':<20s} {'LR':>8s} {'γ':>5s} {'cRT':>8s} {'τ':>6s} "
          f"{'Stacked':>8s} {'Δ0.8017':>8s} {'Δ0.8113':>8s}")
    print("-" * 75)

    # Include Phase F reference
    print(f"{'[F] crt_focal':<20s} {'1e-3':>8s} {'1.5':>5s} {'0.8091':>8s} "
          f"{'0.11':>6s} {'0.8113':>8s} {'+0.0096':>8s} {'---':>8s}")

    for name, r in all_results.items():
        delta_base = r["stacked_score"] - 0.8017
        delta_best = r["stacked_score"] - 0.8113
        print(f"{name:<20s} {r['lr']:>8.0e} {r['focal_gamma']:>5.1f} "
              f"{r['crt_score']:>8.4f} {r['best_logit_adj_tau']:>6.2f} "
              f"{r['stacked_score']:>8.4f} {delta_base:>+8.4f} {delta_best:>+8.4f}")

    # Save all results
    with open(output_base / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_base / 'sweep_results.json'}")

    # Find overall best
    best_name = max(all_results, key=lambda k: all_results[k]["stacked_score"])
    best_r = all_results[best_name]
    print(f"\nBest config: {best_name} → stacked_score={best_r['stacked_score']:.4f}")


if __name__ == "__main__":
    main()
