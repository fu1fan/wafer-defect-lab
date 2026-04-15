#!/usr/bin/env python
"""Export a full evaluation report for a trained wafer classifier."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from waferlab.config import load_yaml_config
from waferlab.data.dataloaders import build_eval_dataloader
from waferlab.engine.evaluator import evaluate
from waferlab.metrics.classification import compute_metrics
from waferlab.models.resnet import FAILURE_TYPE_NAMES, FAILURE_TYPE_TO_IDX
from waferlab.registry import MODEL_REGISTRY
from waferlab.runtime import load_run_summary, resolve_device, resolve_processed_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a full model report")
    p.add_argument("--run-summary", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--task-mode", choices=["binary", "multiclass"], default=None)
    p.add_argument("--split", choices=["Training", "Test", "all"], default="Test")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Root directory under which reports/YYYY-MM-DD/<run_name>/ is created.",
    )
    p.add_argument(
        "--tsne-max-samples",
        type=int,
        default=3000,
        help="Maximum number of samples used to build the t-SNE plot.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--logit-adjustment-tau",
        type=float,
        default=0.0,
        help="Apply post-hoc logit adjustment with tau * log(class_prior).",
    )
    return p.parse_args()


def _resolve_summary(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    summary_path = args.run_summary
    if summary_path is None and args.checkpoint is not None:
        candidate = args.checkpoint.resolve().parent / "run_summary.json"
        if candidate.exists():
            summary_path = candidate

    if summary_path is None:
        return {}, None
    return load_run_summary(summary_path), Path(summary_path).resolve()


def _resolve_checkpoint(
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> Path:
    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpt_best = summary.get("checkpoints", {}).get("best")
        if ckpt_best is not None:
            checkpoint = Path(ckpt_best)
    if checkpoint is None:
        raise SystemExit("Error: --checkpoint is required (or provide --run-summary).")
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise SystemExit(f"Error: checkpoint not found: {checkpoint}")
    return checkpoint


def _resolve_config_and_model(
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, list[str], Path | None]:
    task_mode = args.task_mode or summary.get("task_mode")
    config: dict[str, Any] = {}
    model_cfg: dict[str, Any]
    config_path: Path | None = None

    if "model" in summary:
        model_cfg = dict(summary["model"])
        train_config_entry = summary.get("train_config", {})
        if isinstance(train_config_entry, dict):
            config = train_config_entry.get("snapshot") or {}
            train_config_path = train_config_entry.get("path")
            if train_config_path:
                config_path = Path(train_config_path).resolve()
            if not config:
                if train_config_path and Path(train_config_path).exists():
                    config = load_yaml_config(Path(train_config_path))
        else:
            config_path = Path(train_config_entry).resolve()
            if config_path.exists():
                config = load_yaml_config(config_path)

        if not config and args.config is not None:
            config_path = args.config.resolve()
            config = load_yaml_config(config_path)
    else:
        if args.config is None:
            raise SystemExit(
                "Error: --config is required when no run_summary.json is available."
            )
        config_path = args.config.resolve()
        config = load_yaml_config(config_path)
        model_cfg = dict(config.get("model", {}))
        task_mode = task_mode or config.get("task_mode")

    if task_mode not in {"binary", "multiclass"}:
        raise SystemExit(
            "Error: task mode could not be resolved. Provide --task-mode or --run-summary."
        )

    if not isinstance(config, dict):
        config = {}

    data_cfg = dict(config.get("data", {}))
    data_config_entry = summary.get("data_config")
    data_snapshot: dict[str, Any] = {}
    if isinstance(data_config_entry, dict):
        data_snapshot = data_config_entry.get("snapshot") or {}
        data_config_path = data_config_entry.get("path")
        if not data_snapshot and data_config_path and Path(data_config_path).exists():
            data_snapshot = load_yaml_config(Path(data_config_path))
    elif data_config_entry:
        data_config_path = Path(data_config_entry)
        if data_config_path.exists():
            data_snapshot = load_yaml_config(data_config_path)

    if data_snapshot and "dataset_config" not in data_cfg:
        data_cfg["dataset_config"] = data_snapshot
    data_cfg.setdefault(
        "dataset_name",
        str(data_snapshot.get("dataset_name", "wm811k")).strip().lower() or "wm811k",
    )
    config["data"] = data_cfg
    config["model"] = model_cfg
    config["task_mode"] = task_mode
    if task_mode == "binary":
        model_cfg.setdefault("num_classes", 2)
        class_names = ["normal", "abnormal"]
    else:
        model_cfg.setdefault("num_classes", len(FAILURE_TYPE_TO_IDX))
        class_names = list(FAILURE_TYPE_NAMES)

    return config, model_cfg, task_mode, class_names, config_path


def _resolve_history_path(
    summary: dict[str, Any],
    checkpoint: Path,
) -> Path:
    candidates: list[Path] = []
    if "output_dir" in summary:
        candidates.append(Path(summary["output_dir"]).resolve() / "history.json")
    candidates.append(checkpoint.parent / "history.json")

    for path in candidates:
        if path.exists():
            return path

    raise SystemExit(
        "无法生成完整报告，因为训练曲线所需 history.json 不存在。"
    )


def _infer_run_name(
    summary: dict[str, Any],
    checkpoint: Path,
    model_cfg: dict[str, Any],
) -> str:
    if "output_dir" in summary:
        name = Path(summary["output_dir"]).name.strip()
        if name:
            return name
    if checkpoint.parent.name.strip():
        return checkpoint.parent.name.strip()
    return str(model_cfg.get("arch", "model"))


def _json_ready_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class": metrics["per_class"],
    }


def _normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=np.float64),
            where=row_sums > 0,
        )
    return normalized


def _write_matrix_csv(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def _save_confusion_matrix_artifacts(
    report_dir: Path,
    cm: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    normalized = _normalize_confusion_matrix(cm)

    counts_payload = {
        "labels": class_names,
        "matrix": cm.tolist(),
    }
    normalized_payload = {
        "labels": class_names,
        "matrix": normalized.tolist(),
    }

    with (report_dir / "confusion_matrix_counts.json").open("w", encoding="utf-8") as f:
        json.dump(counts_payload, f, indent=2, ensure_ascii=False)
    with (
        report_dir / "confusion_matrix_normalized.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(normalized_payload, f, indent=2, ensure_ascii=False)

    _write_matrix_csv(report_dir / "confusion_matrix_counts.csv", cm, class_names)
    _write_matrix_csv(
        report_dir / "confusion_matrix_normalized.csv",
        normalized,
        class_names,
    )

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.1), 5.5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall by actual class")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion Matrix (normalized by actual class)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{normalized[i, j]:.2f}\n({int(cm[i, j])})"
            color = "white" if normalized[i, j] >= 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    return normalized


def _plot_training_curves(history: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    val_loss = [float(item["val_loss"]) for item in history]
    train_acc = [float(item["train_acc"]) for item in history]
    val_acc = [float(item["val_acc"]) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val_loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[1].plot(epochs, val_acc, marker="o", label="val_acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _balanced_sample_indices(
    labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    if len(labels) <= max_samples:
        return np.arange(len(labels))

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    base = max_samples // len(unique_labels)

    selected: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []

    for label in unique_labels:
        idx = np.flatnonzero(labels == label)
        idx = rng.permutation(idx)
        take = min(len(idx), base)
        if take > 0:
            selected.append(idx[:take])
        if take < len(idx):
            leftovers.append(idx[take:])

    selected_idx = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
    remaining = max_samples - len(selected_idx)
    if remaining > 0 and leftovers:
        leftover_idx = np.concatenate(leftovers)
        leftover_idx = rng.permutation(leftover_idx)
        selected_idx = np.concatenate([selected_idx, leftover_idx[:remaining]])

    selected_idx = np.sort(selected_idx)
    return selected_idx[:max_samples]


def _plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: Path,
    *,
    model_name: str,
    split: str,
    max_samples: int,
    seed: int,
) -> int:
    if len(features) != len(labels):
        raise ValueError("Feature count does not match label count for t-SNE.")
    if len(features) < 5:
        raise ValueError("t-SNE 至少需要 5 个样本。")
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("t-SNE 至少需要 2 个类别。")

    selected = _balanced_sample_indices(labels, max_samples=max_samples, seed=seed)
    features_sel = features[selected]
    labels_sel = labels[selected]

    if len(features_sel) < 5:
        raise ValueError("下采样后样本不足，无法生成 t-SNE。")
    if len(np.unique(labels_sel)) < 2:
        raise ValueError("下采样后类别不足，无法生成 t-SNE。")

    embed_input = features_sel
    if embed_input.ndim != 2:
        raise ValueError("t-SNE features must be a 2D array.")

    if embed_input.shape[1] > 50:
        n_components = min(50, embed_input.shape[0], embed_input.shape[1])
        embed_input = PCA(n_components=n_components, random_state=seed).fit_transform(
            embed_input
        )

    perplexity = min(30.0, max(5.0, float(len(embed_input) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    embedding = tsne.fit_transform(embed_input)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10", max(len(class_names), 2))
    for class_idx in np.unique(labels_sel):
        points = embedding[labels_sel == class_idx]
        label_name = class_names[int(class_idx)]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=18,
            alpha=0.75,
            color=cmap(int(class_idx) % cmap.N),
            label=label_name,
        )

    ax.set_title(f"t-SNE of {model_name} features ({split}, n={len(labels_sel)})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8, markerscale=1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return int(len(labels_sel))


def _extract_dataset_labels(dataset: Any, task_mode: str) -> np.ndarray:
    inner = dataset.dataset if hasattr(dataset, "dataset") and hasattr(dataset, "indices") else dataset
    index_df = inner.index_df
    if hasattr(dataset, "indices"):
        index_df = index_df.iloc[list(dataset.indices)].reset_index(drop=True)
    if task_mode == "multiclass":
        return index_df["failure_type"].map(
            lambda name: FAILURE_TYPE_TO_IDX.get(str(name), 0)
        ).to_numpy(dtype=np.int64)
    return (~index_df["is_normal"]).astype(int).to_numpy(dtype=np.int64)


def _compute_log_prior(config: dict[str, Any], *, processed_root: Path, task_mode: str) -> np.ndarray:
    train_loader = build_eval_dataloader(
        config,
        processed_root=processed_root,
        split="Training",
    )
    labels = _extract_dataset_labels(train_loader.dataset, task_mode)
    if labels.size == 0:
        raise ValueError("Cannot compute class prior: no training labels found.")
    class_counts = np.bincount(labels)
    priors = class_counts / class_counts.sum()
    return np.log(np.clip(priors, 1e-12, None))


def _write_summary_markdown(
    output_path: Path,
    *,
    report_time: datetime,
    run_name: str,
    split: str,
    task_mode: str,
    checkpoint: Path,
    history_path: Path,
    metrics: dict[str, Any],
    class_names: list[str],
) -> None:
    lines = [
        f"# Model Report: {run_name}",
        "",
        "## Overview",
        "",
        f"- Generated at: `{report_time.isoformat()}`",
        f"- Split: `{split}`",
        f"- Task mode: `{task_mode}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- History: `{history_path}`",
        "",
        "## Headline Metrics",
        "",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Macro Precision: `{metrics['macro_precision']:.4f}`",
        f"- Macro Recall: `{metrics['macro_recall']:.4f}`",
        f"- Macro F1: `{metrics['macro_f1']:.4f}`",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|------|-----------|--------|----|---------|",
    ]
    for row in metrics["per_class"]:
        lines.append(
            f"| {row['class_name']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )

    lines.extend(
        [
            "",
            "## Assets",
            "",
            "- `training_curves.png`",
            "- `confusion_matrix.png`",
            "- `tsne.png`",
            "- `metrics.json`",
            "- `metadata.json`",
            "",
            f"Classes: `{', '.join(class_names)}`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    summary, summary_path = _resolve_summary(args)
    checkpoint = _resolve_checkpoint(args, summary)
    history_path = _resolve_history_path(summary, checkpoint)
    config, model_cfg, task_mode, class_names, config_path = _resolve_config_and_model(
        args,
        summary,
    )

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        raise SystemExit("无法生成完整报告，因为 history.json 为空。")

    report_time = datetime.now()
    run_name = _infer_run_name(summary, checkpoint, model_cfg)
    report_dir = (
        args.output_root.resolve()
        / report_time.strftime("%Y-%m-%d")
        / run_name
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = MODEL_REGISTRY.build(model_cfg.get("arch", "resnet18"), model_cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    processed_root = resolve_processed_root(PROJECT_ROOT)
    eval_loader = build_eval_dataloader(
        config,
        processed_root=processed_root,
        split=args.split,
    )

    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Using history    : {history_path}")
    print(f"Output report dir: {report_dir}")
    print(f"Eval samples     : {len(eval_loader.dataset)} (split={args.split})")

    results = evaluate(
        model,
        eval_loader,
        device=device,
        task_mode=task_mode,
        collect_features=True,
        collect_logits=abs(args.logit_adjustment_tau) > 0.0,
    )
    y_pred = np.asarray(results["y_pred"])
    if abs(args.logit_adjustment_tau) > 0.0:
        logits = np.asarray(results["logits"])
        log_prior = _compute_log_prior(
            config,
            processed_root=processed_root,
            task_mode=task_mode,
        )
        adjusted_logits = logits - args.logit_adjustment_tau * log_prior
        y_pred = adjusted_logits.argmax(axis=1)
    metrics = compute_metrics(
        results["y_true"],
        y_pred,
        class_names=class_names,
    )

    metrics_json = _json_ready_metrics(metrics)
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    normalized_cm = _save_confusion_matrix_artifacts(report_dir, cm, class_names)
    _plot_training_curves(history, report_dir / "training_curves.png")
    tsne_samples = _plot_tsne(
        np.asarray(results["features"]),
        np.asarray(results["y_true"]),
        class_names,
        report_dir / "tsne.png",
        model_name=run_name,
        split=args.split,
        max_samples=args.tsne_max_samples,
        seed=args.seed,
    )

    metadata = {
        "generated_at": report_time.isoformat(),
        "split": args.split,
        "device": device,
        "seed": args.seed,
        "logit_adjustment_tau": args.logit_adjustment_tau,
        "tsne_max_samples": args.tsne_max_samples,
        "tsne_used_samples": tsne_samples,
        "eval_samples": int(len(results["y_true"])),
        "task_mode": task_mode,
        "class_names": class_names,
        "run_name": run_name,
        "report_dir": str(report_dir),
        "checkpoint": str(checkpoint),
        "history_path": str(history_path),
        "config_path": str(config_path) if config_path is not None else None,
        "run_summary_path": str(summary_path) if summary_path is not None else None,
        "model": model_cfg,
        "macro_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
        },
        "confusion_matrix_trace": int(np.trace(cm)),
        "best_val_acc": summary.get("best_val_acc"),
        "normalized_confusion_matrix_diagonal": np.diag(normalized_cm).tolist(),
    }
    with (report_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    _write_summary_markdown(
        report_dir / "summary.md",
        report_time=report_time,
        run_name=run_name,
        split=args.split,
        task_mode=task_mode,
        checkpoint=checkpoint,
        history_path=history_path,
        metrics=metrics_json,
        class_names=class_names,
    )

    print("Report exported successfully.")
    print(f"- summary.md")
    print(f"- metrics.json")
    print(f"- confusion_matrix.png")
    print(f"- training_curves.png")
    print(f"- tsne.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
