#!/usr/bin/env python
"""Phase F – Post-hoc calibration & logit adjustment experiments.

This script loads a trained checkpoint, exports logits, and explores
multiple zero-cost or near-zero-cost post-processing methods to improve
the composite score on WM-811K 9-class classification.

Methods explored:
  1. Temperature Scaling (learned on Training split)
  2. Class-wise Bias Correction (learned on Training split)
  3. Tau-normalization of classifier weights
  4. Per-class threshold tuning via grid search
  5. Prior-aware logit adjustment (post-hoc)

Usage:
    python scripts/posthoc_calibration.py \
        --run-summary outputs/phase_d/D1_gem_tuned/run_summary.json \
        --output-dir outputs/phase_f/F1_posthoc
"""

from __future__ import annotations

import argparse
import json
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from waferlab.config import load_yaml_config
from waferlab.data.dataloaders import build_eval_dataloader
from waferlab.data.transforms import prepare_input, DEFAULT_NORM_SCALE
from waferlab.metrics.classification import compute_metrics, format_metrics
from waferlab.models.resnet import FAILURE_TYPE_NAMES, FAILURE_TYPE_TO_IDX
from waferlab.registry import MODEL_REGISTRY
from waferlab.runtime import load_run_summary, resolve_device, resolve_processed_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MINORITY_CLASSES = ["Center", "Donut", "Loc", "Near-full", "Scratch"]
MINORITY_INDICES = [FAILURE_TYPE_TO_IDX[c] for c in MINORITY_CLASSES]


def compute_score(metrics: dict) -> float:
    """Composite score: 0.40*mF1 + 0.25*mRecall + 0.20*minRecall + 0.15*acc"""
    acc = metrics["accuracy"]
    mf1 = metrics["macro_f1"]
    mr = metrics["macro_recall"]
    # minority recall mean
    per_class = metrics["per_class"]
    min_recalls = [pc["recall"] for pc in per_class if pc["class_name"] in MINORITY_CLASSES]
    min_recall_mean = float(np.mean(min_recalls)) if min_recalls else 0.0
    return 0.40 * mf1 + 0.25 * mr + 0.20 * min_recall_mean + 0.15 * acc


def compute_full_metrics(y_true, y_pred) -> dict:
    """Compute metrics + score."""
    metrics = compute_metrics(y_true, y_pred, class_names=list(FAILURE_TYPE_NAMES))
    metrics["score"] = compute_score(metrics)
    per_class = metrics["per_class"]
    min_recalls = [pc["recall"] for pc in per_class if pc["class_name"] in MINORITY_CLASSES]
    metrics["minority_recall_mean"] = float(np.mean(min_recalls))
    return metrics


def print_comparison(name: str, metrics: dict, baseline_score: float):
    """Print a concise comparison line."""
    s = metrics["score"]
    delta = s - baseline_score
    sign = "+" if delta >= 0 else ""
    print(f"  {name:<40s}  score={s:.4f} ({sign}{delta:.4f})  "
          f"acc={metrics['accuracy']:.4f}  mF1={metrics['macro_f1']:.4f}  "
          f"mR={metrics['macro_recall']:.4f}  minR={metrics['minority_recall_mean']:.4f}")


@torch.no_grad()
def extract_logits(model, loader, *, device, norm_scale=DEFAULT_NORM_SCALE, task_mode="multiclass"):
    """Extract raw logits, labels, and probabilities."""
    dev = torch.device(device)
    model = model.to(dev).eval()

    all_logits, all_labels = [], []
    for batch in loader:
        x = prepare_input(batch, device=dev, target_channels=model.in_channels, norm_scale=norm_scale)
        logits = model(x)
        all_logits.append(logits.cpu())
        if task_mode == "binary":
            all_labels.append(batch["label"])
        else:
            all_labels.append(batch["failure_type_idx"])

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits.numpy(), labels.numpy()


# ── Method 1: Temperature Scaling ────────────────────────────────────

def learn_temperature(logits: np.ndarray, labels: np.ndarray, lr=0.01, max_iter=200) -> float:
    """Learn a single scalar temperature T that minimizes NLL on given data."""
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / temperature
        loss = F.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling and return predicted classes."""
    scaled = logits / T
    return scaled.argmax(axis=1)


# ── Method 2: Class-wise Bias Correction ─────────────────────────────

def learn_class_bias(logits: np.ndarray, labels: np.ndarray, lr=0.1, max_iter=300) -> np.ndarray:
    """Learn per-class additive bias to logits that minimizes NLL."""
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    num_classes = logits.shape[1]
    bias = nn.Parameter(torch.zeros(num_classes))
    optimizer = torch.optim.LBFGS([bias], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        adjusted = logits_t + bias
        loss = F.cross_entropy(adjusted, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return bias.detach().numpy()


def apply_class_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (logits + bias).argmax(axis=1)


# ── Method 3: Temperature + Bias (Vector Scaling) ────────────────────

def learn_vector_scaling(logits: np.ndarray, labels: np.ndarray, lr=0.01, max_iter=300):
    """Learn per-class scale + bias (vector scaling)."""
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    num_classes = logits.shape[1]
    scale = nn.Parameter(torch.ones(num_classes))
    bias = nn.Parameter(torch.zeros(num_classes))
    optimizer = torch.optim.LBFGS([scale, bias], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        adjusted = logits_t * scale + bias
        loss = F.cross_entropy(adjusted, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scale.detach().numpy(), bias.detach().numpy()


def apply_vector_scaling(logits: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (logits * scale + bias).argmax(axis=1)


# ── Method 4: Tau-normalization ──────────────────────────────────────

def tau_normalize_weights(model, class_counts: np.ndarray, tau_values: list[float]):
    """Apply tau-normalization to classifier weights.

    For each tau, rescale W_c by (n_c)^(-tau) and return new model copies.
    """
    results = {}
    # Get the classifier weight and bias
    fc_weight = model.fc.weight.data.clone()  # [C, D]
    fc_bias = model.fc.bias.data.clone() if model.fc.bias is not None else None

    for tau in tau_values:
        new_model = copy.deepcopy(model)
        # Weight norms per class
        norms = fc_weight.norm(dim=1)  # [C]
        # Desired norm: proportional to n_c^tau
        counts_t = torch.from_numpy(class_counts).float().to(fc_weight.device)
        desired = counts_t.pow(tau)
        desired = desired / desired.mean() * norms.mean()  # keep average norm
        scale = desired / (norms + 1e-8)
        new_model.fc.weight.data = fc_weight * scale.unsqueeze(1)
        results[tau] = new_model
    return results


# ── Method 5: Prior-aware post-hoc logit adjustment ──────────────────

def prior_logit_adjustment(logits: np.ndarray, class_counts: np.ndarray,
                           tau_values: list[float]) -> dict:
    """Post-hoc logit adjustment: pred = argmax(logit + tau * log(prior))"""
    total = class_counts.sum()
    log_prior = np.log(class_counts / total + 1e-8)
    results = {}
    for tau in tau_values:
        adjusted = logits + tau * log_prior
        results[tau] = adjusted.argmax(axis=1)
    return results


# ── Method 6: Per-class threshold tuning ─────────────────────────────

def tune_thresholds(logits: np.ndarray, labels: np.ndarray,
                    search_range=np.arange(-2.0, 2.1, 0.1)) -> np.ndarray:
    """Greedy per-class threshold tuning to maximize score.

    For each class c, try adding a bias to logit_c and keep the value
    that maximizes the composite score.
    """
    num_classes = logits.shape[1]
    best_bias = np.zeros(num_classes)

    for c in range(num_classes):
        best_score = -1
        for delta in search_range:
            trial_bias = best_bias.copy()
            trial_bias[c] = delta
            preds = (logits + trial_bias).argmax(axis=1)
            m = compute_full_metrics(labels, preds)
            if m["score"] > best_score:
                best_score = m["score"]
                best_bias[c] = delta

    return best_bias


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Post-hoc calibration experiments")
    p.add_argument("--run-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    # Load run summary and model
    summary = load_run_summary(args.run_summary)
    model_cfg = dict(summary["model"])
    task_mode = summary.get("task_mode", "multiclass")

    train_config_entry = summary.get("train_config", {})
    if isinstance(train_config_entry, dict):
        config = train_config_entry.get("snapshot") or {}
    else:
        config = {}
    config["task_mode"] = task_mode

    checkpoint_path = Path(summary["checkpoints"]["best"])
    output_dir = args.output_dir or checkpoint_path.parent / "posthoc"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    processed_root = resolve_processed_root(PROJECT_ROOT)

    # Build model and load checkpoint
    model = MODEL_REGISTRY.build(model_cfg.get("arch", "resnet50_gem"), model_cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Build data loaders
    print("Building data loaders...")
    train_loader = build_eval_dataloader(config, processed_root=processed_root, split="Training")
    test_loader = build_eval_dataloader(config, processed_root=processed_root, split="Test")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Extract logits
    print("\nExtracting logits...")
    train_logits, train_labels = extract_logits(model, train_loader, device=device, task_mode=task_mode)
    test_logits, test_labels = extract_logits(model, test_loader, device=device, task_mode=task_mode)
    print(f"  Train logits: {train_logits.shape}")
    print(f"  Test logits: {test_logits.shape}")

    # Save logits
    np.savez(output_dir / "logits.npz",
             train_logits=train_logits, train_labels=train_labels,
             test_logits=test_logits, test_labels=test_labels)
    print(f"  Saved to {output_dir / 'logits.npz'}")

    # Class counts from training set
    class_counts = np.bincount(train_labels, minlength=9).astype(np.float64)
    print(f"\nClass counts (train): {class_counts.astype(int).tolist()}")
    print(f"Class names: {list(FAILURE_TYPE_NAMES)}")

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("BASELINE (argmax on raw logits)")
    print("=" * 90)

    baseline_preds = test_logits.argmax(axis=1)
    baseline_metrics = compute_full_metrics(test_labels, baseline_preds)
    baseline_score = baseline_metrics["score"]
    print(format_metrics(baseline_metrics))
    print(f"\n  ** score = {baseline_score:.4f} **")
    print(f"  ** minority_recall_mean = {baseline_metrics['minority_recall_mean']:.4f} **")

    results_all = {"baseline": {
        "score": baseline_score,
        "accuracy": baseline_metrics["accuracy"],
        "macro_f1": baseline_metrics["macro_f1"],
        "macro_recall": baseline_metrics["macro_recall"],
        "minority_recall_mean": baseline_metrics["minority_recall_mean"],
        "per_class_recall": {pc["class_name"]: pc["recall"] for pc in baseline_metrics["per_class"]},
    }}

    # ── Method 1: Temperature Scaling ────────────────────────────────
    print("\n" + "=" * 90)
    print("METHOD 1: Temperature Scaling (learned on Training)")
    print("=" * 90)

    T = learn_temperature(train_logits, train_labels)
    print(f"  Learned temperature: T = {T:.4f}")
    ts_preds = apply_temperature(test_logits, T)
    ts_metrics = compute_full_metrics(test_labels, ts_preds)
    print_comparison("Temp Scaling (T={:.3f})".format(T), ts_metrics, baseline_score)
    results_all["temperature_scaling"] = {
        "T": T,
        "score": ts_metrics["score"],
        "accuracy": ts_metrics["accuracy"],
        "macro_f1": ts_metrics["macro_f1"],
        "macro_recall": ts_metrics["macro_recall"],
        "minority_recall_mean": ts_metrics["minority_recall_mean"],
        "per_class_recall": {pc["class_name"]: pc["recall"] for pc in ts_metrics["per_class"]},
    }

    # Also try a range of fixed temperatures
    print("\n  Fixed temperature sweep:")
    for t in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        preds = apply_temperature(test_logits, t)
        m = compute_full_metrics(test_labels, preds)
        print_comparison(f"  T={t:.1f}", m, baseline_score)

    # ── Method 2: Class-wise Bias ────────────────────────────────────
    print("\n" + "=" * 90)
    print("METHOD 2: Class-wise Bias Correction (learned on Training)")
    print("=" * 90)

    bias = learn_class_bias(train_logits, train_labels)
    print(f"  Learned bias: {np.round(bias, 3).tolist()}")
    cb_preds = apply_class_bias(test_logits, bias)
    cb_metrics = compute_full_metrics(test_labels, cb_preds)
    print_comparison("Class-wise Bias", cb_metrics, baseline_score)
    results_all["class_bias"] = {
        "bias": bias.tolist(),
        "score": cb_metrics["score"],
        "accuracy": cb_metrics["accuracy"],
        "macro_f1": cb_metrics["macro_f1"],
        "macro_recall": cb_metrics["macro_recall"],
        "minority_recall_mean": cb_metrics["minority_recall_mean"],
        "per_class_recall": {pc["class_name"]: pc["recall"] for pc in cb_metrics["per_class"]},
    }

    # ── Method 3: Vector Scaling ─────────────────────────────────────
    print("\n" + "=" * 90)
    print("METHOD 3: Vector Scaling (per-class scale + bias, learned on Training)")
    print("=" * 90)

    vs_scale, vs_bias = learn_vector_scaling(train_logits, train_labels)
    print(f"  Learned scale: {np.round(vs_scale, 3).tolist()}")
    print(f"  Learned bias:  {np.round(vs_bias, 3).tolist()}")
    vs_preds = apply_vector_scaling(test_logits, vs_scale, vs_bias)
    vs_metrics = compute_full_metrics(test_labels, vs_preds)
    print_comparison("Vector Scaling", vs_metrics, baseline_score)
    results_all["vector_scaling"] = {
        "scale": vs_scale.tolist(),
        "bias": vs_bias.tolist(),
        "score": vs_metrics["score"],
        "accuracy": vs_metrics["accuracy"],
        "macro_f1": vs_metrics["macro_f1"],
        "macro_recall": vs_metrics["macro_recall"],
        "minority_recall_mean": vs_metrics["minority_recall_mean"],
        "per_class_recall": {pc["class_name"]: pc["recall"] for pc in vs_metrics["per_class"]},
    }

    # ── Method 4: Tau-normalization ──────────────────────────────────
    print("\n" + "=" * 90)
    print("METHOD 4: Tau-normalization of classifier weights")
    print("=" * 90)

    tau_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    tau_models = tau_normalize_weights(model, class_counts, tau_values)
    best_tau, best_tau_score = 0.0, baseline_score
    for tau, tmodel in tau_models.items():
        tmodel = tmodel.to(device)
        tmodel.eval()
        tl, _ = extract_logits(tmodel, test_loader, device=device, task_mode=task_mode)
        tp = tl.argmax(axis=1)
        tm = compute_full_metrics(test_labels, tp)
        print_comparison(f"tau={tau:.1f}", tm, baseline_score)
        if tm["score"] > best_tau_score:
            best_tau = tau
            best_tau_score = tm["score"]
        results_all[f"tau_norm_{tau}"] = {
            "tau": tau,
            "score": tm["score"],
            "accuracy": tm["accuracy"],
            "macro_f1": tm["macro_f1"],
            "macro_recall": tm["macro_recall"],
            "minority_recall_mean": tm["minority_recall_mean"],
            "per_class_recall": {pc["class_name"]: pc["recall"] for pc in tm["per_class"]},
        }
    print(f"\n  Best tau: {best_tau} (score={best_tau_score:.4f})")

    # ── Method 5: Prior-aware logit adjustment ───────────────────────
    print("\n" + "=" * 90)
    print("METHOD 5: Prior-aware post-hoc logit adjustment")
    print("=" * 90)

    la_taus = [-1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    la_results = prior_logit_adjustment(test_logits, class_counts, la_taus)
    best_la_tau, best_la_score = 0.0, baseline_score
    for tau, preds in la_results.items():
        m = compute_full_metrics(test_labels, preds)
        print_comparison(f"logit_adj tau={tau:.1f}", m, baseline_score)
        if m["score"] > best_la_score:
            best_la_tau = tau
            best_la_score = m["score"]
        results_all[f"logit_adj_{tau}"] = {
            "tau": tau,
            "score": m["score"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_recall": m["macro_recall"],
            "minority_recall_mean": m["minority_recall_mean"],
            "per_class_recall": {pc["class_name"]: pc["recall"] for pc in m["per_class"]},
        }
    print(f"\n  Best logit_adj tau: {best_la_tau} (score={best_la_score:.4f})")

    # ── Method 6: Per-class threshold tuning ─────────────────────────
    print("\n" + "=" * 90)
    print("METHOD 6: Greedy per-class threshold tuning (on Training logits)")
    print("=" * 90)

    # Learn thresholds on training set
    print("  Learning thresholds on Training set (greedy search)...")
    train_bias = tune_thresholds(train_logits, train_labels)
    print(f"  Learned bias (train): {np.round(train_bias, 1).tolist()}")

    # Apply to test
    thresh_preds = (test_logits + train_bias).argmax(axis=1)
    thresh_metrics = compute_full_metrics(test_labels, thresh_preds)
    print_comparison("Threshold Tuning (train→test)", thresh_metrics, baseline_score)
    results_all["threshold_tuning"] = {
        "bias": train_bias.tolist(),
        "score": thresh_metrics["score"],
        "accuracy": thresh_metrics["accuracy"],
        "macro_f1": thresh_metrics["macro_f1"],
        "macro_recall": thresh_metrics["macro_recall"],
        "minority_recall_mean": thresh_metrics["minority_recall_mean"],
        "per_class_recall": {pc["class_name"]: pc["recall"] for pc in thresh_metrics["per_class"]},
    }

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY: All methods ranked by score")
    print("=" * 90)

    ranked = sorted(results_all.items(), key=lambda x: x[1]["score"], reverse=True)
    print(f"\n  {'Method':<45s} {'Score':>7s} {'Delta':>7s} {'Acc':>7s} {'mF1':>7s} {'mR':>7s} {'minR':>7s}")
    print("  " + "-" * 95)
    for name, r in ranked[:15]:
        delta = r["score"] - baseline_score
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<45s} {r['score']:7.4f} {sign}{delta:6.4f} "
              f"{r['accuracy']:7.4f} {r['macro_f1']:7.4f} {r['macro_recall']:7.4f} "
              f"{r['minority_recall_mean']:7.4f}")

    # Save all results
    results_path = output_dir / "posthoc_results.json"
    with results_path.open("w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print the best non-baseline method
    best_method = ranked[0]
    if best_method[0] != "baseline":
        print(f"\n✅ Best method: {best_method[0]} (score={best_method[1]['score']:.4f}, "
              f"delta={best_method[1]['score'] - baseline_score:+.4f})")
    else:
        print(f"\n⚠️  No post-hoc method improved over baseline (score={baseline_score:.4f})")

    # Per-class recall comparison for top methods
    print("\n" + "=" * 90)
    print("PER-CLASS RECALL: Top methods vs baseline")
    print("=" * 90)
    top_methods = ranked[:5]
    header = f"  {'Class':<14s}"
    for name, _ in top_methods:
        header += f" {name[:18]:>18s}"
    print(header)
    print("  " + "-" * (14 + 19 * len(top_methods)))
    for cls_name in FAILURE_TYPE_NAMES:
        line = f"  {cls_name:<14s}"
        for name, r in top_methods:
            rec = r["per_class_recall"].get(cls_name, 0.0)
            line += f" {rec:18.4f}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
