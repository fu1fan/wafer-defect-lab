"""Build processed training-ready datasets from interim wafer-map artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from waferlab.data.processed import build_processed_dataset, load_data_config
from waferlab.runtime import resolve_interim_root, resolve_processed_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed wafer datasets from interim artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "wm811k.yaml",
        help="Path to the dataset processing config YAML.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild processed artifacts even if they already exist.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        choices=["labeled", "unlabeled"],
        default=None,
        help="Only build the specified subset(s). Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_data_config(args.config)
    dataset_name = str(config.get("dataset_name", "")).strip()
    if not dataset_name:
        print("[error] `dataset_name` is required in the config file.", file=sys.stderr)
        return 2

    interim_root = resolve_interim_root(PROJECT_ROOT)
    processed_root = resolve_processed_root(PROJECT_ROOT)

    print(f"[process] Dataset: {dataset_name}")
    print(f"[process] Config: {args.config}")

    try:
        artifacts = build_processed_dataset(
            dataset_name,
            interim_root=interim_root,
            processed_root=processed_root,
            config=config,
            force=args.force,
            subsets=args.subset,
        )
    except NotImplementedError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[error] Failed to process dataset: {exc}", file=sys.stderr)
        return 1

    for subset_name, subset_artifacts in artifacts.items():
        print(f"[done] {subset_name}: {subset_artifacts.h5_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
