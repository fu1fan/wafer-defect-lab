"""Classification metrics for wafer-level evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : ndarray [N]
        Ground-truth class indices.
    y_pred : ndarray [N]
        Predicted class indices.
    class_names : list[str], optional
        Human-readable names for each class index.

    Returns
    -------
    dict with keys: ``accuracy``, ``per_class`` (list of per-class dicts),
    ``macro_precision``, ``macro_recall``, ``macro_f1``,
    ``confusion_matrix`` (C×C ndarray).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "per_class": [], "macro_f1": 0.0}

    num_classes = max(int(y_true.max()), int(y_pred.max())) + 1

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = float(np.trace(cm)) / n

    per_class: list[dict[str, Any]] = []
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm[c, :].sum())
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        per_class.append({
            "class_idx": c,
            "class_name": name,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        })
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": cm,
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    """Return a human-readable summary string."""
    lines = [
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics.get('macro_f1', 0):.4f}  "
        f"Macro P: {metrics.get('macro_precision', 0):.4f}  "
        f"Macro R: {metrics.get('macro_recall', 0):.4f}",
        "",
        f"{'Class':<14s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}",
        "-" * 44,
    ]
    for pc in metrics.get("per_class", []):
        lines.append(
            f"{pc['class_name']:<14s} {pc['precision']:6.3f} {pc['recall']:6.3f} "
            f"{pc['f1']:6.3f} {pc['support']:8d}"
        )
    return "\n".join(lines)
