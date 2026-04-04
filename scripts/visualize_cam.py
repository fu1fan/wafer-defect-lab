#!/usr/bin/env python
"""Generate GradCAM heatmaps for a trained wafer-level classifier.

These heatmaps are *interpretability aids* showing which spatial regions the
classifier focuses on.  They are **not** anomaly-localization maps (which would
require a normality-distribution method like PaDiM / PatchCore).

Examples
--------
    python scripts/visualize_cam.py \
        --checkpoint outputs/wm811k_resnet_baseline/best.pt \
        --num-samples 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from waferlab.data.datasets import WM811KProcessedDataset
from waferlab.data.transforms import prepare_input
from waferlab.models.classifier import FAILURE_TYPE_NAMES, FAILURE_TYPE_TO_IDX, build_classifier
from waferlab.runtime import resolve_device, resolve_processed_root
from waferlab.visualize.cam import GradCAM

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GradCAM heatmaps")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path,
                   default=PROJECT_ROOT / "configs" / "train" / "wm811k_resnet_baseline.yaml")
    p.add_argument("--task-mode", choices=["binary", "multiclass"], default=None)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--split", choices=["Training", "Test", "all"], default="Test")
    p.add_argument("--only-abnormal", action="store_true",
                   help="Only visualise samples predicted or labeled as abnormal")
    return p.parse_args()


def _save_heatmaps_matplotlib(
    images: np.ndarray,
    heatmaps: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray | None,
    sample_ids: np.ndarray | None,
    output_dir: Path,
    class_names: list[str],
) -> None:
    """Save a grid of wafer maps overlaid with GradCAM heatmaps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed – saving raw .npy instead")
        np.save(output_dir / "cam_heatmaps.npy", heatmaps)
        np.save(output_dir / "cam_images.npy", images)
        return

    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols * 2 == 1:
        axes = axes[:, np.newaxis]

    for i in range(n):
        r = i // cols
        c = i % cols

        wafer = images[i, 0]  # [H, W]
        cam = heatmaps[i]     # [H, W]

        pred_name = class_names[predictions[i]] if predictions[i] < len(class_names) else str(predictions[i])
        title = f"pred: {pred_name}"
        if labels is not None:
            gt_name = class_names[labels[i]] if labels[i] < len(class_names) else str(labels[i])
            title = f"gt: {gt_name} | {title}"
        if sample_ids is not None:
            title = f"id={sample_ids[i]}  {title}"

        # Original wafer map
        ax_orig = axes[r, c * 2]
        ax_orig.imshow(wafer, cmap="gray", vmin=0, vmax=2)
        ax_orig.set_title(title, fontsize=8)
        ax_orig.axis("off")

        # GradCAM overlay
        ax_cam = axes[r, c * 2 + 1]
        ax_cam.imshow(wafer, cmap="gray", vmin=0, vmax=2)
        ax_cam.imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        ax_cam.set_title("GradCAM response", fontsize=8)
        ax_cam.axis("off")

    # Hide unused axes.
    for i in range(n, rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    fig.suptitle(
        "GradCAM – classifier attention (interpretability only, not anomaly localization)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = output_dir / "gradcam_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"GradCAM grid saved to {out_path}")


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    task_mode = args.task_mode or config.get("task_mode", "binary")

    model_cfg = config.get("model", {})
    if task_mode == "binary":
        model_cfg["num_classes"] = 2
        class_names = ["normal", "abnormal"]
    else:
        model_cfg["num_classes"] = len(FAILURE_TYPE_TO_IDX)
        class_names = list(FAILURE_TYPE_NAMES)

    output_dir = args.output_dir or (args.checkpoint.parent / "cam")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    device = resolve_device(args.device)
    model = build_classifier(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Build dataset.
    processed_root = resolve_processed_root(PROJECT_ROOT)
    filters: dict = {}
    if args.split != "all":
        filters["split_label"] = args.split
    if args.only_abnormal:
        filters["is_normal"] = False

    ds = WM811KProcessedDataset(
        processed_root,
        subset="labeled",
        include_metadata=False,
        return_masks=False,
        return_float=True,
        filters=filters,
    )

    n = min(args.num_samples, len(ds))
    # Pick evenly-spaced samples for diversity.
    indices = np.linspace(0, len(ds) - 1, n, dtype=int).tolist()
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=n, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    dev = torch.device(device)
    x_norm = prepare_input(batch, device=dev, target_channels=model.in_channels)

    cam = GradCAM(model)
    heatmaps = cam(x_norm)
    cam.remove_hooks()

    with torch.no_grad():
        preds = model(x_norm).argmax(1).cpu().numpy()

    labels = batch.get("label")
    labels = labels.numpy() if labels is not None else None
    sample_ids = batch.get("sample_id")
    sample_ids = sample_ids.numpy() if sample_ids is not None else None

    _save_heatmaps_matplotlib(
        batch["image"].numpy(), heatmaps, preds, labels, sample_ids,
        output_dir, class_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
