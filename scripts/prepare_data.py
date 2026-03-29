# scripts/prepare_data.py
"""
Data initialization script for the wafer defect project.

Downloads and organizes the required Kaggle datasets. Future data preparation
steps (e.g. normalization, train/val splitting) will be added here.

Dependencies:
    pip install -r requirements.txt

Before first use, configure Kaggle credentials, for example:
    export KAGGLE_USERNAME=...
    export KAGGLE_KEY=...

Examples:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --dataset WM-811K
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing from src/ without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from waferlab.data import build_interim_dataset, download_dataset, resolve_dataset_names  # noqa: E402
from waferlab.runtime import resolve_interim_root, resolve_raw_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize data for the wafer defect project"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["auto"],
        help="Dataset(s) to download. Options: WM-811K, MixedWM38, auto/all",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If target directory exists, delete and re-download",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = resolve_raw_root(PROJECT_ROOT)
    interim_root = resolve_interim_root(PROJECT_ROOT)

    print("[STEP1] Starting data download...")

    # Download datasets
    try:
        dataset_names = resolve_dataset_names(args.dataset)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    print(f"Output directory: {output_root}")
    print(f"Datasets to download: {', '.join(dataset_names)}")

    for dataset_name in dataset_names:
        try:
            download_dataset(dataset_name, output_root=output_root, force=args.force)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        
    print("[STEP2] Building interim datasets...")
    for dataset_name in dataset_names:
        try:
            build_interim_dataset(
                dataset_name,
                raw_root=output_root,
                interim_root=interim_root,
                force=args.force,
            )
        except Exception as exc:
            print(
                f"[error] Failed to build {dataset_name} interim dataset: {exc}",
                file=sys.stderr,
            )
            return 1

    print("All datasets processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
