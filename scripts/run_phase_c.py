#!/usr/bin/env python
"""Phase C: Structural enhancement experiments for ResNet50.

Runs GeM, CBAM, and GeM+CBAM variants with R0 hyperparameters.
Each experiment runs for 20 epochs (full training).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_classifier.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_classifier.py"

MINORITY_CLASSES = ["Center", "Donut", "Loc", "Near-full", "Scratch"]


def compute_score(metrics: dict) -> float:
    macro_f1 = metrics.get("macro_f1", 0)
    macro_recall = metrics.get("macro_recall", 0)
    accuracy = metrics.get("accuracy", 0)
    per_class = metrics.get("per_class", [])
    minority_recalls = [pc["recall"] for pc in per_class if pc["class_name"] in MINORITY_CLASSES]
    minority_recall_mean = np.mean(minority_recalls) if minority_recalls else 0
    return 0.40 * macro_f1 + 0.25 * macro_recall + 0.20 * minority_recall_mean + 0.15 * accuracy


def run_experiment(config_path: Path, output_dir: Path) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    eval_file = output_dir / "eval_metrics_Test.json"
    if eval_file.exists() and eval_file.stat().st_size > 0:
        print(f"  [SKIP] Already evaluated: {output_dir.name}")
        with eval_file.open() as f:
            return json.load(f)

    # Train
    print(f"  Training {output_dir.name} ...")
    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] Training failed for {output_dir.name}")
        print(result.stderr[-500:] if result.stderr else "No stderr")
        return None

    # Eval
    run_summary = output_dir / "run_summary.json"
    if not run_summary.exists():
        print(f"  [FAIL] No run_summary for {output_dir.name}")
        return None

    print(f"  Evaluating {output_dir.name} ...")
    eval_cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--run-summary", str(run_summary),
        "--split", "Test",
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] Eval failed for {output_dir.name}")
        print(result.stderr[-500:] if result.stderr else "No stderr")
        return None

    if eval_file.exists() and eval_file.stat().st_size > 0:
        with eval_file.open() as f:
            return json.load(f)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default=None,
                        help="Comma-separated experiment IDs to run")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    args = parser.parse_args()

    phase_c_root = PROJECT_ROOT / "outputs" / "phase_c"
    phase_c_root.mkdir(parents=True, exist_ok=True)

    experiments = {
        "C1_gem": PROJECT_ROOT / "configs" / "modal" / "baseline" / "experiments" / "phase_c" / "resnet50_gem.yaml",
        "C2_cbam": PROJECT_ROOT / "configs" / "modal" / "baseline" / "experiments" / "phase_c" / "resnet50_cbam.yaml",
        "C3_gem_cbam": PROJECT_ROOT / "configs" / "modal" / "baseline" / "experiments" / "phase_c" / "resnet50_gem_cbam.yaml",
    }

    if args.subset:
        subset_ids = [s.strip() for s in args.subset.split(",")]
        experiments = {k: v for k, v in experiments.items() if k in subset_ids}

    all_results = {}
    for exp_name, config_path in experiments.items():
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")

        output_dir = phase_c_root / exp_name
        metrics = run_experiment(config_path, output_dir)
        if metrics is not None:
            score = compute_score(metrics)
            minority_recalls = {}
            for pc in metrics.get("per_class", []):
                if pc["class_name"] in MINORITY_CLASSES:
                    minority_recalls[pc["class_name"]] = pc["recall"]

            all_results[exp_name] = {
                "metrics": {
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_precision": metrics["macro_precision"],
                },
                "per_class": metrics.get("per_class", []),
                "minority_recalls": minority_recalls,
                "minority_recall_mean": float(np.mean(list(minority_recalls.values()))) if minority_recalls else 0,
                "score": score,
            }

    # Print summary
    print(f"\n{'='*100}")
    print("PHASE C STRUCTURAL RESULTS")
    print(f"{'='*100}")
    header = f"{'Experiment':<20s} {'Score':>6s} {'mF1':>6s} {'mRec':>6s} {'minRec':>6s} {'Acc':>6s} | {'Center':>6s} {'Donut':>6s} {'Loc':>6s} {'NrFul':>6s} {'Scratch':>6s}"
    print(header)
    print("-" * len(header))

    # Add R0 reference
    r0_eval = PROJECT_ROOT / "outputs" / "r0_resnet50_cb_focal_heavy_aug" / "eval_metrics_Test.json"
    if r0_eval.exists():
        with r0_eval.open() as f:
            r0_metrics = json.load(f)
        r0_score = compute_score(r0_metrics)
        r0_mr = {}
        for pc in r0_metrics.get("per_class", []):
            if pc["class_name"] in MINORITY_CLASSES:
                r0_mr[pc["class_name"]] = pc["recall"]
        r0_minrec = float(np.mean(list(r0_mr.values()))) if r0_mr else 0
        print(
            f"{'R0_baseline':<20s} {r0_score:6.4f} {r0_metrics['macro_f1']:6.4f} "
            f"{r0_metrics['macro_recall']:6.4f} {r0_minrec:6.4f} "
            f"{r0_metrics['accuracy']:6.4f} | "
            f"{r0_mr.get('Center', 0):6.3f} {r0_mr.get('Donut', 0):6.3f} "
            f"{r0_mr.get('Loc', 0):6.3f} {r0_mr.get('Near-full', 0):6.3f} "
            f"{r0_mr.get('Scratch', 0):6.3f}"
        )

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True)
    for name, r in sorted_results:
        mr = r.get("minority_recalls", {})
        print(
            f"{name:<20s} {r['score']:6.4f} {r['metrics']['macro_f1']:6.4f} "
            f"{r['metrics']['macro_recall']:6.4f} {r.get('minority_recall_mean', 0):6.4f} "
            f"{r['metrics']['accuracy']:6.4f} | "
            f"{mr.get('Center', 0):6.3f} {mr.get('Donut', 0):6.3f} "
            f"{mr.get('Loc', 0):6.3f} {mr.get('Near-full', 0):6.3f} "
            f"{mr.get('Scratch', 0):6.3f}"
        )

    # Save results
    results_path = phase_c_root / "phase_c_results.json"
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
