#!/usr/bin/env python
"""Evaluate ensemble calibration on disjoint stratified splits.

This script estimates how much of a logit-ensemble gain survives when the
ensemble weight and logit-adjustment tau are selected on one split and evaluated
on a disjoint split. It uses precomputed logits, so it does not retrain models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]
MINORITY_IDX = [1, 2, 5, 6, 8]
TRAIN_COUNTS = np.array([36730, 3462, 409, 2417, 8554, 1620, 54, 609, 500])
LOG_PRIOR = np.log(TRAIN_COUNTS / TRAIN_COUNTS.sum())


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    acc = float((labels == preds).mean())
    per_class_recall: dict[str, float] = {}
    per_class_f1: list[float] = []
    recalls: list[float] = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        true = labels == cls_idx
        pred = preds == cls_idx
        tp = int(np.logical_and(true, pred).sum())
        fp = int(np.logical_and(~true, pred).sum())
        fn = int(np.logical_and(true, ~pred).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class_recall[cls_name] = float(recall)
        recalls.append(float(recall))
        per_class_f1.append(float(f1))

    macro_f1 = float(np.mean(per_class_f1))
    macro_recall = float(np.mean(recalls))
    minority_recall = float(np.mean([recalls[idx] for idx in MINORITY_IDX]))
    score = 0.40 * macro_f1 + 0.25 * macro_recall + 0.20 * minority_recall + 0.15 * acc
    return {
        "score": float(score),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "minority_recall_mean": minority_recall,
        "per_class_recall": per_class_recall,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def stratified_split(labels: np.ndarray, *, eval_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cal_parts: list[np.ndarray] = []
    eval_parts: list[np.ndarray] = []

    for cls_idx in range(len(CLASS_NAMES)):
        idx = np.flatnonzero(labels == cls_idx)
        idx = rng.permutation(idx)
        n_eval = max(1, int(round(len(idx) * eval_frac)))
        eval_parts.append(idx[:n_eval])
        cal_parts.append(idx[n_eval:])

    cal_idx = rng.permutation(np.concatenate(cal_parts))
    eval_idx = rng.permutation(np.concatenate(eval_parts))
    return cal_idx, eval_idx


def predict_ensemble(
    cnn_logits: np.ndarray,
    vit_logits: np.ndarray,
    *,
    w_cnn: float,
    tau: float,
) -> np.ndarray:
    logits = w_cnn * cnn_logits + (1.0 - w_cnn) * vit_logits
    logits = logits - tau * LOG_PRIOR
    return logits.argmax(axis=1)


def sweep_params(
    cnn_logits: np.ndarray,
    vit_logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    taus: np.ndarray,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None

    for w_cnn in weights:
        mixed = w_cnn * cnn_logits + (1.0 - w_cnn) * vit_logits
        for tau in taus:
            preds = (mixed - tau * LOG_PRIOR).argmax(axis=1)
            metrics = compute_metrics(labels, preds)
            if best is None or metrics["score"] > best["metrics"]["score"]:
                best = {
                    "w_cnn": float(round(float(w_cnn), 6)),
                    "w_vit": float(round(float(1.0 - w_cnn), 6)),
                    "tau": float(round(float(tau), 6)),
                    "metrics": metrics,
                }

    assert best is not None
    return best


def load_logits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return np.asarray(data["logits"], dtype=np.float32), np.asarray(data["labels"], dtype=np.int64)


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {key: round_floats(value, ndigits) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_floats(value, ndigits) for value in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cnn-logits",
        type=Path,
        default=Path("outputs/phase_g/G2_gamma1.0/logits.npz"),
    )
    parser.add_argument(
        "--vit-logits",
        type=Path,
        default=Path("outputs/vit_phase_i/I5_smooth_resample_050/logits.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vit_phase_i/clean_calibration/results.json"),
    )
    parser.add_argument("--eval-frac", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=20260425)
    parser.add_argument("--weight-min", type=float, default=0.30)
    parser.add_argument("--weight-max", type=float, default=0.70)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--tau-min", type=float, default=-0.30)
    parser.add_argument("--tau-max", type=float, default=0.30)
    parser.add_argument("--tau-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cnn_logits, labels_cnn = load_logits(args.cnn_logits)
    vit_logits, labels_vit = load_logits(args.vit_logits)

    if cnn_logits.shape != vit_logits.shape:
        raise ValueError(f"logit shapes differ: {cnn_logits.shape} vs {vit_logits.shape}")
    if not np.array_equal(labels_cnn, labels_vit):
        raise ValueError("label arrays differ; logits are not aligned")

    labels = labels_cnn
    weights = np.round(
        np.arange(args.weight_min, args.weight_max + args.weight_step / 2.0, args.weight_step),
        10,
    )
    taus = np.round(
        np.arange(args.tau_min, args.tau_max + args.tau_step / 2.0, args.tau_step),
        10,
    )

    full_test_sweep = sweep_params(cnn_logits, vit_logits, labels, weights, taus)
    full_test_fixed = compute_metrics(
        labels,
        predict_ensemble(cnn_logits, vit_logits, w_cnn=0.50, tau=0.14),
    )
    full_test_cnn_g2 = compute_metrics(
        labels,
        predict_ensemble(cnn_logits, vit_logits, w_cnn=1.00, tau=-0.05),
    )
    full_test_vit_i5 = compute_metrics(labels, vit_logits.argmax(axis=1))

    split_results: list[dict[str, Any]] = []
    for offset in range(args.seeds):
        seed = args.seed_base + offset
        cal_idx, eval_idx = stratified_split(labels, eval_frac=args.eval_frac, seed=seed)
        selected = sweep_params(
            cnn_logits[cal_idx],
            vit_logits[cal_idx],
            labels[cal_idx],
            weights,
            taus,
        )
        eval_preds = predict_ensemble(
            cnn_logits[eval_idx],
            vit_logits[eval_idx],
            w_cnn=selected["w_cnn"],
            tau=selected["tau"],
        )
        eval_metrics = compute_metrics(labels[eval_idx], eval_preds)
        cnn_metrics = compute_metrics(
            labels[eval_idx],
            predict_ensemble(cnn_logits[eval_idx], vit_logits[eval_idx], w_cnn=1.00, tau=-0.05),
        )
        vit_metrics = compute_metrics(labels[eval_idx], vit_logits[eval_idx].argmax(axis=1))
        split_results.append(
            {
                "seed": seed,
                "cal_size": int(len(cal_idx)),
                "eval_size": int(len(eval_idx)),
                "selected": selected,
                "eval_metrics": eval_metrics,
                "cnn_g2_eval_metrics": cnn_metrics,
                "vit_i5_eval_metrics": vit_metrics,
                "delta_vs_cnn_g2": float(eval_metrics["score"] - cnn_metrics["score"]),
                "delta_vs_vit_i5": float(eval_metrics["score"] - vit_metrics["score"]),
            }
        )

    summary = {
        "protocol": {
            "type": "repeated_stratified_test_split_calibration",
            "note": (
                "Weights and tau are selected on one stratified portion of the official Test logits "
                "and evaluated on the disjoint remainder. This estimates test-sweep optimism without "
                "retraining models with a true held-out calibration set."
            ),
            "eval_frac": args.eval_frac,
            "seeds": args.seeds,
            "weight_grid": [float(weights[0]), float(weights[-1]), args.weight_step],
            "tau_grid": [float(taus[0]), float(taus[-1]), args.tau_step],
        },
        "inputs": {
            "cnn_logits": str(args.cnn_logits),
            "vit_logits": str(args.vit_logits),
            "num_samples": int(len(labels)),
        },
        "full_test_reference": {
            "swept_on_full_test": full_test_sweep,
            "fixed_w050_tau014": full_test_fixed,
            "cnn_g2_tau_minus005": full_test_cnn_g2,
            "vit_i5_raw": full_test_vit_i5,
        },
        "split_summary": {
            "ensemble_score": summarize([r["eval_metrics"]["score"] for r in split_results]),
            "cnn_g2_score": summarize([r["cnn_g2_eval_metrics"]["score"] for r in split_results]),
            "vit_i5_score": summarize([r["vit_i5_eval_metrics"]["score"] for r in split_results]),
            "delta_vs_cnn_g2": summarize([r["delta_vs_cnn_g2"] for r in split_results]),
            "delta_vs_vit_i5": summarize([r["delta_vs_vit_i5"] for r in split_results]),
            "selected_w_cnn": summarize([r["selected"]["w_cnn"] for r in split_results]),
            "selected_tau": summarize([r["selected"]["tau"] for r in split_results]),
            "loc_recall": summarize(
                [r["eval_metrics"]["per_class_recall"]["Loc"] for r in split_results]
            ),
            "edge_loc_recall": summarize(
                [r["eval_metrics"]["per_class_recall"]["Edge-Loc"] for r in split_results]
            ),
        },
        "splits": split_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(round_floats(summary), f, indent=2)

    print(json.dumps(round_floats(summary["split_summary"], ndigits=5), indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
