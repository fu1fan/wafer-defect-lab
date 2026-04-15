"""Evaluation utilities for wafer-level classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.transforms import prepare_input, DEFAULT_NORM_SCALE


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str = "cuda",
    task_mode: str = "binary",
    norm_scale: float = DEFAULT_NORM_SCALE,
    collect_features: bool = False,
    collect_logits: bool = False,
) -> dict[str, np.ndarray]:
    """Run inference on *loader* and collect predictions.

    Returns
    -------
    dict with keys ``y_true``, ``y_pred``, ``y_prob`` (softmax), ``sample_ids``.
    When ``collect_features=True``, also returns ``features`` containing the
    model's ``forward_features`` output for each sample.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[np.ndarray] = []
    all_ids: list[int] = []
    all_features: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    if collect_features and not hasattr(model, "forward_features"):
        raise AttributeError(
            "collect_features=True requires the model to implement "
            "`forward_features(x)`."
        )

    for batch in loader:
        x = prepare_input(
            batch,
            device=dev,
            target_channels=model.in_channels,
            norm_scale=norm_scale,
        )

        feat = model.forward_features(x) if collect_features else None
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

        if task_mode == "binary":
            labels = batch["label"].numpy()
        else:
            labels = batch["failure_type_idx"].numpy()

        all_true.extend(labels.tolist())
        all_pred.extend(preds.tolist())
        all_prob.append(probs)
        if "sample_id" in batch:
            all_ids.extend(batch["sample_id"].numpy().tolist())
        if feat is not None:
            all_features.append(feat.detach().cpu().numpy())
        if collect_logits:
            all_logits.append(logits.detach().cpu().numpy())

    results = {
        "y_true": np.array(all_true, dtype=np.int64),
        "y_pred": np.array(all_pred, dtype=np.int64),
        "y_prob": np.concatenate(all_prob, axis=0),
        "sample_ids": (
            np.array(all_ids, dtype=np.int64)
            if all_ids
            else np.array([], dtype=np.int64)
        ),
    }
    if collect_features:
        results["features"] = (
            np.concatenate(all_features, axis=0)
            if all_features
            else np.empty((0, 0), dtype=np.float32)
        )
    if collect_logits:
        results["logits"] = (
            np.concatenate(all_logits, axis=0)
            if all_logits
            else np.empty((0, 0), dtype=np.float32)
        )
    return results
