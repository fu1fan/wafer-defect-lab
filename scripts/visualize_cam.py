#!/usr/bin/env python
"""Generate GradCAM heatmaps for a trained wafer-level classifier.

These heatmaps are *interpretability aids* showing which spatial regions the
classifier focuses on.  They are **not** anomaly-localization maps (which would
require a normality-distribution method like PaDiM / PatchCore).

Examples
--------
    # Preferred: drive entirely from a run summary produced by training
    python scripts/visualize_cam.py \
        --run-summary outputs/wm811k_resnet_baseline/run_summary.json

    # Legacy: manual arguments (still supported)
    python scripts/visualize_cam.py \
        --checkpoint outputs/wm811k_resnet_baseline/best.pt \
        --num-samples 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from waferlab.config import load_yaml_config
from waferlab.data.datasets import WM811KProcessedDataset
from waferlab.data.transforms import prepare_input
from waferlab.models.resnet import FAILURE_TYPE_NAMES, FAILURE_TYPE_TO_IDX
from waferlab.registry import MODEL_REGISTRY
from waferlab.runtime import load_run_summary, resolve_device, resolve_processed_root
from waferlab.visualize.cam import GradCAM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GradCAM heatmaps")
    p.add_argument(
        "--run-summary", type=Path, default=None,
        help="Path to run_summary.json produced by train_classifier.py. "
             "When provided, checkpoint / config / task-mode / output-dir "
             "are read from it automatically (CLI flags still override).",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
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

    # ── Resolve parameters: run_summary provides defaults, CLI overrides ──
    summary: dict = {}
    if args.run_summary is not None:
        summary = load_run_summary(args.run_summary)

    # Checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpt_best = summary.get("checkpoints", {}).get("best")
        if ckpt_best is not None:
            checkpoint = Path(ckpt_best)
    if checkpoint is None:
        raise SystemExit(
            "Error: --checkpoint is required (or provide --run-summary)."
        )

    # Task mode
    task_mode = args.task_mode
    if task_mode is None:
        task_mode = summary.get("task_mode")

    # Model config — prefer the snapshot stored in run_summary so that
    # arch / num_classes / in_channels are always consistent with the
    # checkpoint.
    model_cfg: dict
    if "model" in summary:
        model_cfg = dict(summary["model"])
    else:
        config_path = args.config or PROJECT_ROOT / "configs" / "modal" / "experiments" / "wm811k_resnet18_baseline.yaml"
        config = load_yaml_config(config_path)
        model_cfg = config.get("model", {})
        task_mode = task_mode or config.get("task_mode", "binary")

    task_mode = task_mode or "binary"

    if task_mode == "binary":
        model_cfg.setdefault("num_classes", 2)
        class_names = ["normal", "abnormal"]
    else:
        model_cfg.setdefault("num_classes", len(FAILURE_TYPE_TO_IDX))
        class_names = list(FAILURE_TYPE_NAMES)

    # Output directory: CLI > run_summary output_dir/cam > checkpoint sibling
    output_dir: Path
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif "output_dir" in summary:
        output_dir = Path(summary["output_dir"]) / "cam"
    else:
        output_dir = checkpoint.parent / "cam"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    device = resolve_device(args.device)
    model = MODEL_REGISTRY.build(model_cfg.get("arch", "resnet18"), model_cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

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
