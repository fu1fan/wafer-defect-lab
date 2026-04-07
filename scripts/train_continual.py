#!/usr/bin/env python
"""Class-incremental continual-learning training for wafer classification.

Splits the WM-811K multiclass labels into sequential tasks and trains
the model incrementally.  After each task, evaluates on all classes
seen so far to measure forgetting.

Supports:
- Custom task class ordering (``continual.task_order``)
- Per-task epoch budget (``continual.epochs_per_task``)
- Inverse-frequency class-weighted loss (``continual.class_weight_per_task``)
- Class-balanced sampling (``continual.balanced_sampling``)
- Exemplar-replay buffer (``continual.replay``)
- Configurable CMS slow-level freeze strategy

Examples
--------
    # Smoke test
    python scripts/train_continual.py \
        --config configs/modal/research_nest/experiments/wm811k_nested_selfmod_continual.yaml \
        --smoke-test

    # Full run
    python scripts/train_continual.py \
        --config configs/modal/research_nest/experiments/wm811k_nested_selfmod_continual.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from waferlab.config import load_yaml_config
from waferlab.data.dataloaders import build_classification_dataloaders
from waferlab.data.processed import load_data_config
from waferlab.models.resnet import FAILURE_TYPE_NAMES, FAILURE_TYPE_TO_IDX
from waferlab.registry import MODEL_REGISTRY
from waferlab.runtime import resolve_device, resolve_output_root, resolve_processed_root
from waferlab.engine.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continual-learning wafer classifier")
    p.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "modal" / "research_nest" / "experiments"
                / "wm811k_nested_selfmod_continual.yaml",
    )
    p.add_argument(
        "--data-config", type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "wm811k.yaml",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def filter_by_classes(
    dataset: Any, class_set: set[int],
) -> list[int]:
    """Return indices where failure_type_idx is in class_set.

    Uses the dataset's index_df directly for speed (avoids loading
    every sample from HDF5).
    """
    # Unwrap Subset if present.
    if isinstance(dataset, Subset):
        subset_indices = dataset.indices
        inner_ds = dataset.dataset
    else:
        subset_indices = None
        inner_ds = dataset

    # Unwrap one more level if needed (double Subset from smoke test).
    if isinstance(inner_ds, Subset):
        if subset_indices is not None:
            subset_indices = [inner_ds.indices[i] for i in subset_indices]
        else:
            subset_indices = inner_ds.indices
        inner_ds = inner_ds.dataset

    if hasattr(inner_ds, "index_df") and "failure_type" in inner_ds.index_df.columns:
        ft_series = inner_ds.index_df["failure_type"].reset_index(drop=True)
        idx_to_name = {v: k for k, v in FAILURE_TYPE_TO_IDX.items()}
        target_names = {idx_to_name[c] for c in class_set if c in idx_to_name}

        if subset_indices is not None:
            # Filter within the subset's indices and return positions
            # relative to the subset (0, 1, ..., len(subset)-1).
            result = []
            for pos, real_idx in enumerate(subset_indices):
                if str(ft_series.iloc[real_idx]) in target_names:
                    result.append(pos)
            return result
        else:
            mask = ft_series.isin(target_names)
            return list(mask[mask].index)

    # Fallback: iterate samples (slow).
    indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        ft_idx = sample.get("failure_type_idx", 0)
        if ft_idx in class_set:
            indices.append(i)
    return indices


def get_class_for_indices(dataset: Any, indices: list[int]) -> list[int]:
    """Return the class label for each index.

    Uses index_df for speed.
    """
    # Unwrap Subset.
    if isinstance(dataset, Subset):
        real_indices = [dataset.indices[i] for i in indices]
        inner_ds = dataset.dataset
    else:
        real_indices = indices
        inner_ds = dataset

    if isinstance(inner_ds, Subset):
        real_indices = [inner_ds.indices[i] for i in real_indices]
        inner_ds = inner_ds.dataset

    if hasattr(inner_ds, "index_df") and "failure_type" in inner_ds.index_df.columns:
        ft_series = inner_ds.index_df["failure_type"].reset_index(drop=True)
        return [FAILURE_TYPE_TO_IDX.get(str(ft_series.iloc[i]), 0) for i in real_indices]

    labels = []
    for idx in indices:
        sample = dataset[idx]
        labels.append(sample.get("failure_type_idx", 0))
    return labels


def compute_inverse_freq_weights(
    labels: list[int], num_classes: int, mode: str = "inv_freq",
) -> list[float]:
    """Compute class weights for CE loss.

    Modes:
      - "inv_freq": weight[c] = N / (K * count_c)  (aggressive)
      - "sqrt_inv_freq": weight[c] = sqrt(N / (K * count_c))  (moderate)

    Classes not present get weight 0.
    """
    counts = Counter(labels)
    total = len(labels)
    num_active = len(counts)
    weights = [0.0] * num_classes
    for c, cnt in counts.items():
        if c < num_classes:
            raw = total / (num_active * cnt)
            if mode == "sqrt_inv_freq":
                weights[c] = raw ** 0.5
            else:
                weights[c] = raw
    return weights


def build_balanced_sampler(
    dataset: Any, indices: list[int], labels: list[int],
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for class-balanced training."""
    counts = Counter(labels)
    sample_weights = [1.0 / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


class ExemplarBuffer:
    """Simple exemplar-replay buffer that stores K samples per class."""

    def __init__(self, exemplars_per_class: int = 100) -> None:
        self.exemplars_per_class = exemplars_per_class
        self._class_indices: dict[int, list[int]] = defaultdict(list)

    def update(
        self, dataset: Any, task_indices: list[int], task_labels: list[int],
    ) -> None:
        """Store exemplars from just-completed task."""
        class_to_idx: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in zip(task_indices, task_labels):
            class_to_idx[lbl].append(idx)

        for cls, idxs in class_to_idx.items():
            k = min(self.exemplars_per_class, len(idxs))
            self._class_indices[cls] = random.sample(idxs, k)

        total = sum(len(v) for v in self._class_indices.values())
        print(f"  [replay] buffer updated: {len(self._class_indices)} classes, {total} exemplars")

    def get_replay_indices(self, exclude_classes: set[int] | None = None) -> list[int]:
        """Return all stored exemplar indices, optionally excluding some classes."""
        indices = []
        for cls, idxs in self._class_indices.items():
            if exclude_classes and cls in exclude_classes:
                continue
            indices.extend(idxs)
        return indices

    @property
    def total_exemplars(self) -> int:
        return sum(len(v) for v in self._class_indices.values())


def evaluate_on_classes(
    model: nn.Module,
    eval_loader: DataLoader,
    class_set: set[int],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate model on samples belonging to class_set."""
    model.eval()
    correct = 0
    total = 0
    per_class_correct: dict[int, int] = defaultdict(int)
    per_class_total: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for batch in eval_loader:
            x = batch["image"].to(device) / 2.0
            y = batch["failure_type_idx"].to(device, dtype=torch.long)

            mask = torch.zeros(y.shape[0], dtype=torch.bool, device=device)
            for c in class_set:
                mask |= (y == c)
            if not mask.any():
                continue

            x_sel = x[mask]
            y_sel = y[mask]
            logits = model(x_sel)
            preds = logits.argmax(1)
            correct += (preds == y_sel).sum().item()
            total += y_sel.size(0)

            for c in class_set:
                c_mask = y_sel == c
                per_class_correct[c] += (preds[c_mask] == y_sel[c_mask]).sum().item()
                per_class_total[c] += c_mask.sum().item()

    acc = correct / max(total, 1)
    per_class_acc = {
        c: per_class_correct[c] / max(per_class_total[c], 1)
        for c in sorted(class_set)
    }
    return {"accuracy": acc, "per_class_accuracy": per_class_acc, "total": total}


def evaluate_all_classes(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    num_classes: int = 9,
) -> dict[str, Any]:
    """Full multiclass evaluation on the entire test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            x = batch["image"].to(device) / 2.0
            y = batch["failure_type_idx"].to(device, dtype=torch.long)
            logits = model(x)
            preds = logits.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    overall_acc = (all_preds == all_labels).float().mean().item()
    per_class_acc = {}
    per_class_count = {}
    for c in range(num_classes):
        c_mask = all_labels == c
        if c_mask.sum() > 0:
            per_class_acc[c] = (all_preds[c_mask] == c).float().mean().item()
            per_class_count[c] = int(c_mask.sum())
        else:
            per_class_acc[c] = 0.0
            per_class_count[c] = 0

    # Macro metrics (over classes with samples).
    active = [c for c in range(num_classes) if per_class_count[c] > 0]
    macro_acc = np.mean([per_class_acc[c] for c in active]) if active else 0.0

    return {
        "overall_accuracy": overall_acc,
        "macro_accuracy": float(macro_acc),
        "per_class_accuracy": per_class_acc,
        "per_class_count": per_class_count,
    }


def freeze_slow_cms(model: nn.Module) -> None:
    """Freeze parameters of slow CMS levels in nested blocks."""
    if not hasattr(model, "nested_blocks"):
        return
    for block in model.nested_blocks:
        if hasattr(block, "cms") and hasattr(block.cms, "blocks"):
            for name, cms_block in block.cms.blocks.items():
                if "slow" in name:
                    for p in cms_block.parameters():
                        p.requires_grad_(False)
    print("[continual] Frozen slow CMS levels")


def unfreeze_all_cms(model: nn.Module) -> None:
    """Unfreeze all CMS parameters."""
    if not hasattr(model, "nested_blocks"):
        return
    for block in model.nested_blocks:
        if hasattr(block, "cms") and hasattr(block.cms, "blocks"):
            for cms_block in block.cms.blocks.values():
                for p in cms_block.parameters():
                    p.requires_grad_(True)


# ---------- Weight Alignment (WA) – bias correction ---------------------

def weight_alignment(
    model: nn.Module,
    old_classes: set[int],
    new_classes: set[int],
) -> None:
    """Post-task Weight Alignment (Zhao et al., 2020).

    Scales the classifier weight rows of *new* classes so that their
    mean L2-norm matches the mean norm of *old* class rows.  This
    corrects the bias toward the most-recently-learned classes whose
    weights tend to have larger norms.
    """
    fc = _get_fc_layer(model)
    if fc is None:
        print("  [WA] WARNING: could not locate FC layer")
        return
    with torch.no_grad():
        w = fc.weight.data  # [num_classes, feat_dim]
        old_idx = sorted(old_classes)
        new_idx = sorted(new_classes)
        if not old_idx or not new_idx:
            return
        old_norms = w[old_idx].norm(dim=1)
        new_norms = w[new_idx].norm(dim=1)
        mean_old = old_norms.mean()
        mean_new = new_norms.mean()
        if mean_new > 1e-8:
            gamma = mean_old / mean_new
            w[new_idx] *= gamma
            print(f"  [WA] old_norm={mean_old:.4f}  new_norm={mean_new:.4f}  "
                  f"gamma={gamma:.4f}")


def _get_fc_layer(model: nn.Module) -> nn.Linear | None:
    """Return the final classification Linear layer."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "head") and isinstance(model.head, nn.Sequential):
        for m in reversed(list(model.head.modules())):
            if isinstance(m, nn.Linear):
                return m
    return None


# ---------- Cosine-normalized classifier ---------------------------------

class CosineLinear(nn.Module):
    """Cosine-normalized linear classifier (LUCIR-style).

    Removes magnitude bias: similarity = eta * cos(w_c, f).
    """

    def __init__(self, in_features: int, out_features: int, eta: float = 10.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.eta = eta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return self.eta * F.linear(x_norm, w_norm)


def replace_fc_with_cosine(model: nn.Module, eta: float = 10.0) -> None:
    """Replace the model's FC head with a CosineLinear layer."""
    fc = _get_fc_layer(model)
    if fc is None:
        print("  [cosine] WARNING: could not locate FC layer")
        return
    cosine_head = CosineLinear(fc.in_features, fc.out_features, eta=eta)
    # Copy existing weight as initialization.
    with torch.no_grad():
        cosine_head.weight.copy_(fc.weight)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = cosine_head
    elif hasattr(model, "head") and isinstance(model.head, nn.Sequential):
        # Replace the last Linear in the head.
        for i in range(len(model.head) - 1, -1, -1):
            if isinstance(model.head[i], nn.Linear):
                model.head[i] = cosine_head
                break
    print(f"  [cosine] Replaced FC with CosineLinear (eta={eta})")


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    data_config = load_data_config(args.data_config)
    data_section = config.get("data", {})
    config.setdefault("data", data_section)
    data_section["dataset_config"] = data_config

    task_mode = "multiclass"
    config["task_mode"] = task_mode
    device = resolve_device(args.device)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    train_cfg = config.setdefault("train", {})
    model_cfg = config.get("model", {})
    model_cfg["num_classes"] = len(FAILURE_TYPE_TO_IDX)
    continual_cfg = config.get("continual", {})

    num_tasks = int(continual_cfg.get("num_tasks", 3))
    classes_per_task = int(continual_cfg.get("classes_per_task", 3))
    do_freeze_slow = bool(continual_cfg.get("freeze_slow_cms", True))
    do_class_weight = bool(continual_cfg.get("class_weight_per_task", False))
    class_weight_mode = str(continual_cfg.get("class_weight_mode", "inv_freq"))
    do_balanced_sampling = bool(continual_cfg.get("balanced_sampling", False))
    max_samples_per_class = continual_cfg.get('max_samples_per_class', None)
    if max_samples_per_class is not None:
        max_samples_per_class = int(max_samples_per_class)

    # Replay buffer config.
    replay_cfg = continual_cfg.get("replay", {})
    replay_enabled = bool(replay_cfg.get("enabled", False))
    replay_per_class = int(replay_cfg.get("exemplars_per_class", 100))

    # Per-task epoch budget.
    epochs_per_task_cfg = continual_cfg.get("epochs_per_task")

    # Custom task ordering (list of lists of class indices).
    task_order_cfg = continual_cfg.get("task_order")

    # Knowledge distillation config (LwF-style).
    kd_cfg = continual_cfg.get("knowledge_distillation", {})
    kd_enabled = bool(kd_cfg.get("enabled", False))
    kd_alpha = float(kd_cfg.get("alpha", 0.5))        # weight for CE loss
    kd_temperature = float(kd_cfg.get("temperature", 2.0))
    kd_lambda = float(kd_cfg.get("lambda", 0.0))      # additive mode: CE + lambda*KD
    kd_replay_only = bool(kd_cfg.get("replay_only", False))  # KD on old-class samples only
    kd_skip_tasks: list[int] = list(kd_cfg.get("skip_tasks", []))  # task indices to skip KD

    # Weight alignment config (WA bias correction).
    wa_cfg = continual_cfg.get("weight_alignment", {})
    wa_enabled = bool(wa_cfg.get("enabled", False))

    # Cosine classifier config (LUCIR-style).
    cosine_cfg = continual_cfg.get("cosine_classifier", {})
    cosine_enabled = bool(cosine_cfg.get("enabled", False))
    cosine_eta = float(cosine_cfg.get("eta", 10.0))

    # Whether to load best.pt for the final task (can harm KD setups).
    load_best_final = bool(continual_cfg.get("load_best_final", True))

    if args.smoke_test:
        train_cfg["epochs"] = 1
        train_cfg["log_interval"] = 10

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch = model_cfg.get("arch", "nested_selfmod")
    default_output = resolve_output_root(PROJECT_ROOT) / f"{arch}_continual_{timestamp}"
    output_dir = args.output_dir or default_output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list: custom ordering or sequential chunks.
    if task_order_cfg is not None:
        tasks = [list(tc) for tc in task_order_cfg]
        num_tasks = len(tasks)
    else:
        all_classes = list(range(len(FAILURE_TYPE_TO_IDX)))
        tasks: list[list[int]] = []
        for i in range(0, len(all_classes), classes_per_task):
            task_classes = all_classes[i : i + classes_per_task]
            if task_classes:
                tasks.append(task_classes)
        tasks = tasks[:num_tasks]

    # Per-task epoch schedule.
    if epochs_per_task_cfg is not None:
        if isinstance(epochs_per_task_cfg, (list, tuple)):
            epochs_schedule = [int(e) for e in epochs_per_task_cfg]
        else:
            epochs_schedule = [int(epochs_per_task_cfg)] * num_tasks
    else:
        default_epochs = int(train_cfg.get("epochs", 5))
        epochs_schedule = [default_epochs] * num_tasks

    print(f"Continual learning: {len(tasks)} tasks")
    for ti, tc in enumerate(tasks):
        names = [FAILURE_TYPE_NAMES[c] for c in tc]
        ep = epochs_schedule[ti] if ti < len(epochs_schedule) else epochs_schedule[-1]
        print(f"  Task {ti}: classes {tc} ({names}) — {ep} epochs")
    print(f"  freeze_slow_cms={do_freeze_slow}  class_weight={do_class_weight}  "
          f"balanced_sampling={do_balanced_sampling}  replay={replay_enabled}")
    print(f"  kd={kd_enabled}(alpha={kd_alpha}, lambda={kd_lambda}, T={kd_temperature}, replay_only={kd_replay_only})  "
          f"wa={wa_enabled}  cosine_head={cosine_enabled}")
    print()

    # Build dataloaders (full multiclass set).
    processed_root = resolve_processed_root(PROJECT_ROOT)
    loaders = build_classification_dataloaders(
        config, processed_root=processed_root, smoke_test=args.smoke_test,
    )
    train_loader_full = loaders["train"]
    val_loader = loaders["val"]
    train_ds = train_loader_full.dataset

    # Build model.
    model = MODEL_REGISTRY.build(model_cfg.get("arch", "nested_selfmod"), model_cfg)
    model = model.to(dev)

    # Optionally replace FC with cosine-normalized head.
    if cosine_enabled:
        replace_fc_with_cosine(model, eta=cosine_eta)

    # Replay buffer.
    replay_buffer = ExemplarBuffer(replay_per_class) if replay_enabled else None

    # Teacher model for KD (will be set after first task).
    teacher_model: nn.Module | None = None

    # Results tracking.
    task_accuracy_matrix: list[list[float]] = []
    task_per_class_details: list[dict] = []
    all_seen_classes: set[int] = set()

    for task_idx, task_classes in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_idx}: classes {task_classes} "
              f"({[FAILURE_TYPE_NAMES[c] for c in task_classes]})")
        print(f"{'='*60}")

        all_seen_classes.update(task_classes)
        task_class_set = set(task_classes)

        # Filter training data to current task classes.
        task_indices = filter_by_classes(train_ds, task_class_set)

        if args.smoke_test:
            task_indices = task_indices[:min(128, len(task_indices))]

        if not task_indices:
            print(f"  No training samples for task {task_idx}, skipping")
            task_accuracy_matrix.append([0.0] * len(tasks))
            continue

        task_labels = get_class_for_indices(train_ds, task_indices)
        label_counts = Counter(task_labels)

        # Cap dominant classes if configured.
        if max_samples_per_class is not None:
            capped_indices = []
            capped_labels = []
            class_to_idxs: dict[int, list[int]] = defaultdict(list)
            for idx, lbl in zip(task_indices, task_labels):
                class_to_idxs[lbl].append(idx)
            for cls, idxs in class_to_idxs.items():
                k = min(max_samples_per_class, len(idxs))
                selected = random.sample(idxs, k) if k < len(idxs) else idxs
                capped_indices.extend(selected)
                capped_labels.extend([cls] * len(selected))
            # Report capping.
            new_counts = Counter(capped_labels)
            for c in sorted(label_counts):
                orig = label_counts[c]
                capped = new_counts.get(c, 0)
                if capped < orig:
                    print(f"  [cap] class {c} ({FAILURE_TYPE_NAMES[c]}): {orig} -> {capped}")
            task_indices = capped_indices
            task_labels = capped_labels
            label_counts = new_counts

        print(f"  Training samples: {len(task_indices)}")
        for c in sorted(label_counts):
            print(f"    class {c} ({FAILURE_TYPE_NAMES[c]}): {label_counts[c]}")

        # Combine with replay buffer if available.
        combined_indices = list(task_indices)
        if replay_buffer is not None and replay_buffer.total_exemplars > 0:
            replay_indices = replay_buffer.get_replay_indices(
                exclude_classes=task_class_set,
            )
            if replay_indices:
                combined_indices.extend(replay_indices)
                replay_labels = get_class_for_indices(train_ds, replay_indices)
                replay_counts = Counter(replay_labels)
                print(f"  Replay exemplars: {len(replay_indices)}")
                for c in sorted(replay_counts):
                    print(f"    class {c} ({FAILURE_TYPE_NAMES[c]}): {replay_counts[c]}")

        combined_labels = get_class_for_indices(train_ds, combined_indices)
        task_subset = Subset(train_ds, combined_indices)

        # Per-task training config.
        task_train_cfg = dict(train_cfg)
        task_epochs = epochs_schedule[min(task_idx, len(epochs_schedule) - 1)]
        if not args.smoke_test:
            task_train_cfg["epochs"] = task_epochs

        # Compute per-task class weights if enabled.
        if do_class_weight:
            cw = compute_inverse_freq_weights(combined_labels, len(FAILURE_TYPE_TO_IDX), mode=class_weight_mode)
            task_train_cfg["class_weights"] = cw
            active_cw = {c: cw[c] for c in sorted(set(combined_labels))}
            print(f"  Class weights: { {FAILURE_TYPE_NAMES[c]: f'{w:.2f}' for c, w in active_cw.items()} }")

        # Build dataloader with optional balanced sampling.
        bs = int(data_section.get("batch_size", 64))
        nw = int(data_section.get("num_workers", 4))

        if do_balanced_sampling and not args.smoke_test:
            sampler = build_balanced_sampler(train_ds, combined_indices, combined_labels)
            task_train_loader = DataLoader(
                task_subset, batch_size=bs, sampler=sampler,
                num_workers=nw, pin_memory=True,
                drop_last=len(combined_indices) > bs,
            )
        else:
            task_train_loader = DataLoader(
                task_subset, batch_size=bs, shuffle=True,
                num_workers=nw, pin_memory=True,
                drop_last=len(combined_indices) > bs,
            )

        # Freeze slow CMS for anti-forgetting (after first task).
        if task_idx > 0 and do_freeze_slow:
            freeze_slow_cms(model)

        # Snapshot teacher model for KD BEFORE training on new task.
        kd_active = kd_enabled and task_idx > 0 and task_idx not in kd_skip_tasks
        if kd_active:
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            print(f"  [KD] Teacher snapshot saved (alpha={kd_alpha}, T={kd_temperature})")

        # Create a task-specific trainer.
        task_output = output_dir / f"task_{task_idx}"
        task_train_cfg["kd_alpha"] = kd_alpha if kd_active else 1.0
        task_train_cfg["kd_temperature"] = kd_temperature
        task_train_cfg["kd_lambda"] = kd_lambda if kd_active else 0.0
        task_train_cfg["kd_replay_only"] = kd_replay_only if kd_active else False
        trainer = Trainer(
            model, task_train_loader, val_loader, task_train_cfg,
            device=device,
            output_dir=task_output,
            task_mode=task_mode,
        )
        if kd_active and teacher_model is not None:
            trainer.set_teacher(teacher_model)
            # Masked KD: only distill old-class logits (avoids noise on new classes).
            old_classes_for_kd = sorted(all_seen_classes - task_class_set)
            trainer.kd_old_classes = old_classes_for_kd
            print(f"  [KD] Masked distillation on old classes: {old_classes_for_kd}")
        trainer.fit()

        # Free teacher to reclaim memory.
        if teacher_model is not None:
            del teacher_model
            teacher_model = None
            torch.cuda.empty_cache()

        # Load best checkpoint ONLY for the final task.
        # For intermediate tasks, val_acc is on the full test set where unseen
        # classes (especially "none" at 93%) make the metric meaningless.
        # Loading best.pt for intermediate tasks would select a near-initial
        # checkpoint that barely learned the current task.
        is_final_task = (task_idx == len(tasks) - 1)
        best_ckpt = task_output / "best.pt"
        if is_final_task and load_best_final and best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=dev, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  [checkpoint] Loaded best.pt from task {task_idx} (final task)")
        else:
            print(f"  [checkpoint] Using last epoch model for task {task_idx}")

        # Post-task Weight Alignment (bias correction).
        if wa_enabled and task_idx > 0:
            old_cl = all_seen_classes - task_class_set
            weight_alignment(model, old_cl, task_class_set)

        # Unfreeze for evaluation.
        unfreeze_all_cms(model)

        # Update replay buffer after this task.
        if replay_buffer is not None:
            replay_buffer.update(train_ds, task_indices, task_labels)

        # Evaluate on all tasks.
        print(f"\n  --- Evaluation after task {task_idx} ---")
        task_accs: list[float] = []
        task_details: dict[str, Any] = {"task_idx": task_idx}
        for eval_task_idx, eval_task_classes in enumerate(tasks):
            eval_class_set = set(eval_task_classes)
            if eval_class_set & all_seen_classes:
                result = evaluate_on_classes(
                    model, val_loader, eval_class_set, dev,
                )
                acc = result["accuracy"]
                pc_acc = result["per_class_accuracy"]
            else:
                acc = 0.0
                pc_acc = {}
            task_accs.append(acc)
            task_details[f"task_{eval_task_idx}_acc"] = acc
            task_details[f"task_{eval_task_idx}_per_class"] = pc_acc
            status = "SEEN" if eval_class_set <= all_seen_classes else "partial/unseen"
            print(f"  Task {eval_task_idx}: acc={acc:.4f} ({status})")
            for c, ca in sorted(pc_acc.items()):
                print(f"    class {c} ({FAILURE_TYPE_NAMES[c]}): {ca:.4f}")

        task_accuracy_matrix.append(task_accs)
        task_per_class_details.append(task_details)

    # Full test-set evaluation after all tasks.
    print(f"\n{'='*60}")
    print("Full Test-Set Evaluation")
    print(f"{'='*60}")
    full_eval = evaluate_all_classes(model, val_loader, dev, len(FAILURE_TYPE_TO_IDX))
    print(f"Overall accuracy: {full_eval['overall_accuracy']:.4f}")
    print(f"Macro accuracy:   {full_eval['macro_accuracy']:.4f}")
    for c in range(len(FAILURE_TYPE_TO_IDX)):
        ca = full_eval['per_class_accuracy'].get(c, 0.0)
        cn = full_eval['per_class_count'].get(c, 0)
        print(f"  class {c} ({FAILURE_TYPE_NAMES[c]}): acc={ca:.4f}  n={cn}")

    # Compute final metrics.
    num_completed = len(task_accuracy_matrix)
    if num_completed > 0:
        final_accs = task_accuracy_matrix[-1]
        avg_accuracy_final = np.mean([
            final_accs[i] for i in range(min(num_completed, len(final_accs)))
        ])

        forgetting_values = []
        for i in range(num_completed - 1):
            max_acc_i = max(
                task_accuracy_matrix[j][i] for j in range(i, num_completed)
            )
            final_acc_i = final_accs[i]
            forgetting_values.append(max_acc_i - final_acc_i)
        avg_forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0
    else:
        avg_accuracy_final = 0.0
        avg_forgetting = 0.0

    # Save results.
    results = {
        "num_tasks": len(tasks),
        "tasks": [
            {"classes": tc, "names": [FAILURE_TYPE_NAMES[c] for c in tc]}
            for tc in tasks
        ],
        "task_accuracy_matrix": task_accuracy_matrix,
        "task_per_class_details": task_per_class_details,
        "avg_accuracy_final": float(avg_accuracy_final),
        "avg_forgetting": float(avg_forgetting),
        "full_test_eval": full_eval,
        "config_summary": {
            "freeze_slow_cms": do_freeze_slow,
            "class_weight_per_task": do_class_weight,
            "balanced_sampling": do_balanced_sampling,
            "replay_enabled": replay_enabled,
            "replay_per_class": replay_per_class if replay_enabled else None,
            "epochs_per_task": epochs_schedule,
            "kd_enabled": kd_enabled,
            "kd_alpha": kd_alpha if kd_enabled else None,
            "kd_lambda": kd_lambda if kd_enabled else None,
            "kd_temperature": kd_temperature if kd_enabled else None,
            "wa_enabled": wa_enabled,
            "cosine_classifier": cosine_enabled,
        },
        "created_at": datetime.now().isoformat(),
    }
    results_path = output_dir / "continual_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Continual Learning Results")
    print(f"{'='*60}")
    print(f"Avg accuracy (final): {avg_accuracy_final:.4f}")
    print(f"Avg forgetting:       {avg_forgetting:.4f}")
    print(f"Full test macro acc:  {full_eval['macro_accuracy']:.4f}")
    print(f"Results saved to {results_path}")

    # Save final model.
    torch.save(
        {"model_state_dict": model.state_dict()},
        output_dir / "final_model.pt",
    )
    print(f"Final model saved to {output_dir / 'final_model.pt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
