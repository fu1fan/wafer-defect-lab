#!/usr/bin/env python
"""Batch hyperparameter search for ResNet50 multiclass WM-811K.

Runs multiple experiments sequentially with different hyperparameters,
each for a reduced number of epochs (screening mode).
Evaluates each and writes a consolidated results table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_classifier.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_classifier.py"
BASE_CONFIG = PROJECT_ROOT / "configs" / "modal" / "baseline" / "experiments" / "wm811k_resnet50_cb_sampler_focal_heavy_aug_multiclass.yaml"

MINORITY_CLASSES = ["Center", "Donut", "Loc", "Near-full", "Scratch"]
MINORITY_INDICES = [1, 2, 5, 6, 8]  # indices in FAILURE_TYPE_NAMES


def compute_score(metrics: dict) -> float:
    """Compute the unified scoring function."""
    macro_f1 = metrics.get("macro_f1", 0)
    macro_recall = metrics.get("macro_recall", 0)
    accuracy = metrics.get("accuracy", 0)
    per_class = metrics.get("per_class", [])
    minority_recalls = []
    for pc in per_class:
        if pc["class_name"] in MINORITY_CLASSES:
            minority_recalls.append(pc["recall"])
    minority_recall_mean = np.mean(minority_recalls) if minority_recalls else 0
    return 0.40 * macro_f1 + 0.25 * macro_recall + 0.20 * minority_recall_mean + 0.15 * accuracy


def generate_config(
    output_dir: Path,
    config_name: str,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.2,
    focal_gamma: float = 2.0,
    epochs: int = 15,
    sampler: str = "class_balanced",
    translate_frac: float = 0.08,
    scale_min: float = 0.95,
    scale_max: float = 1.05,
    batch_size: int = 64,
    scheduler: str = "cosine",
) -> Path:
    """Generate a YAML experiment config with the given hyperparameters."""
    config_dir = PROJECT_ROOT / "configs" / "modal" / "baseline" / "experiments" / "search"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config_name}.yaml"

    sampler_line = f"  sampler: {sampler}" if sampler else "  sampler: \"\""

    content = f"""# Auto-generated search config: {config_name}
_base_:
  - ../../../base/wm811k_classifier.yaml
  - ../../models/resnet50.yaml

task_mode: multiclass

model:
  arch: resnet50
  in_channels: 1
  pretrained: false
  dropout: {dropout}

train:
  epochs: {epochs}
  lr: {lr}
  weight_decay: {weight_decay}
  grad_clip: 1.0
  loss_type: focal
  focal_gamma: {focal_gamma}
  optimizer: adamw
  scheduler: {scheduler}

data:
{sampler_line}
  batch_size: {batch_size}
  augment:
    random_flip: true
    random_rotate90: true
    random_translate_frac: {translate_frac}
    random_scale_min: {scale_min}
    random_scale_max: {scale_max}
"""
    config_path.write_text(content)
    return config_path


def run_experiment(config_path: Path, output_dir: Path) -> dict | None:
    """Run training + evaluation, return metrics dict or None on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]
    print(f"\n{'='*70}")
    print(f"TRAINING: {output_dir.name}")
    print(f"Config: {config_path}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(train_cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"Training took {elapsed:.0f}s (exit code {result.returncode})")
    if result.returncode != 0:
        print(f"ERROR: Training failed for {output_dir.name}")
        return None

    # Evaluate
    run_summary = output_dir / "run_summary.json"
    if not run_summary.exists():
        print(f"ERROR: No run_summary.json found for {output_dir.name}")
        return None

    eval_cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--run-summary", str(run_summary),
    ]
    print(f"\nEVALUATING: {output_dir.name}")
    result = subprocess.run(eval_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed for {output_dir.name}")
        return None

    # Load metrics
    metrics_path = output_dir / "eval_metrics_Test.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only generate configs")
    parser.add_argument("--subset", type=str, default=None,
                        help="Comma-separated experiment IDs to run (e.g. 'B1,B3,B5')")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_root = PROJECT_ROOT / "outputs" / "phase_b_search"
    search_root.mkdir(parents=True, exist_ok=True)

    # Define search grid
    experiments = {
        # B1: R0 reference at 15 epochs (to compare apples-to-apples)
        "B1_r0_ref": dict(
            lr=1e-3, weight_decay=1e-3, dropout=0.2, focal_gamma=2.0,
            translate_frac=0.08, scale_min=0.95, scale_max=1.05,
        ),
        # B2: Lower LR, lower gamma (less focal effect)
        "B2_lr5e4_gamma1p5": dict(
            lr=5e-4, weight_decay=1e-3, dropout=0.2, focal_gamma=1.5,
        ),
        # B3: Lower LR, higher dropout, lower gamma
        "B3_lr3e4_wd5e4_drop03": dict(
            lr=3e-4, weight_decay=5e-4, dropout=0.3, focal_gamma=1.5,
        ),
        # B4: Higher gamma, stronger regularization
        "B4_wd5e4_drop03_gamma2p5": dict(
            lr=1e-3, weight_decay=5e-4, dropout=0.3, focal_gamma=2.5,
        ),
        # B5: Moderate LR with stronger WD
        "B5_lr5e4_wd1e3_drop02": dict(
            lr=5e-4, weight_decay=1e-3, dropout=0.2, focal_gamma=2.0,
        ),
        # B6: Minimal focal (gamma=1.0), strong WD, low dropout
        "B6_gamma1p0_wd1e3_drop01": dict(
            lr=1e-3, weight_decay=1e-3, dropout=0.1, focal_gamma=1.0,
        ),
        # B7: Stronger augmentation
        "B7_strong_aug": dict(
            lr=5e-4, weight_decay=5e-4, dropout=0.2, focal_gamma=2.0,
            translate_frac=0.12, scale_min=0.90, scale_max=1.10,
        ),
        # B8: No class-balanced sampler (rely only on focal)
        "B8_no_sampler": dict(
            lr=1e-3, weight_decay=1e-3, dropout=0.2, focal_gamma=2.0,
            sampler="",
        ),
        # B9: Lower LR + higher gamma + dropout 0.3
        "B9_lr5e4_gamma2p5_drop03": dict(
            lr=5e-4, weight_decay=1e-3, dropout=0.3, focal_gamma=2.5,
        ),
    }

    # Filter if subset requested
    if args.subset:
        subset_ids = [s.strip() for s in args.subset.split(",")]
        experiments = {k: v for k, v in experiments.items() if k in subset_ids}

    # Default params
    defaults = dict(
        epochs=15, sampler="class_balanced",
        translate_frac=0.08, scale_min=0.95, scale_max=1.05,
        batch_size=64, scheduler="cosine",
    )

    # Generate configs
    configs = {}
    for exp_name, overrides in experiments.items():
        params = {**defaults, **overrides}
        config_path = generate_config(
            search_root / exp_name,
            config_name=exp_name,
            **params,
        )
        configs[exp_name] = (config_path, params)
        print(f"Generated: {config_path}")

    if args.dry_run:
        print("\nDry run complete. Configs generated.")
        return

    # Run experiments
    all_results = {}
    for exp_name, (config_path, params) in configs.items():
        output_dir = search_root / exp_name
        metrics = run_experiment(config_path, output_dir)
        if metrics is not None:
            score = compute_score(metrics)
            all_results[exp_name] = {
                "params": params,
                "metrics": {
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_precision": metrics["macro_precision"],
                },
                "per_class": metrics.get("per_class", []),
                "score": score,
            }
            # Extract minority recalls
            minority_recalls = {}
            for pc in metrics.get("per_class", []):
                if pc["class_name"] in MINORITY_CLASSES:
                    minority_recalls[pc["class_name"]] = pc["recall"]
            all_results[exp_name]["minority_recalls"] = minority_recalls
            all_results[exp_name]["minority_recall_mean"] = np.mean(list(minority_recalls.values()))

    # Print summary table
    print(f"\n{'='*100}")
    print("PHASE B SEARCH RESULTS SUMMARY")
    print(f"{'='*100}")
    header = f"{'Experiment':<28s} {'Score':>6s} {'mF1':>6s} {'mRec':>6s} {'minRec':>6s} {'Acc':>6s} | {'Center':>6s} {'Donut':>6s} {'Loc':>6s} {'NrFul':>6s} {'Scratch':>6s}"
    print(header)
    print("-" * len(header))

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True)
    for name, r in sorted_results:
        mr = r.get("minority_recalls", {})
        print(
            f"{name:<28s} {r['score']:6.4f} {r['metrics']['macro_f1']:6.4f} "
            f"{r['metrics']['macro_recall']:6.4f} {r.get('minority_recall_mean', 0):6.4f} "
            f"{r['metrics']['accuracy']:6.4f} | "
            f"{mr.get('Center', 0):6.3f} {mr.get('Donut', 0):6.3f} "
            f"{mr.get('Loc', 0):6.3f} {mr.get('Near-full', 0):6.3f} "
            f"{mr.get('Scratch', 0):6.3f}"
        )

    # Save results
    results_path = search_root / "search_results.json"
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Identify top 3
    top3 = sorted_results[:3]
    print(f"\nTOP 3 CONFIGURATIONS:")
    for i, (name, r) in enumerate(top3, 1):
        print(f"  {i}. {name} (score={r['score']:.4f}, mF1={r['metrics']['macro_f1']:.4f})")


if __name__ == "__main__":
    main()
