"""Processed dataset builders for training-ready wafer map artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .interim_io import ensure_h5py, write_index_artifacts


PROCESSED_SCHEMA_VERSION = "1"
WM811K_TARGET_SIZE = (224, 224)


@dataclass(frozen=True)
class ProcessedArtifacts:
    h5_path: Path
    index_path: Path
    parquet_path: Path


def load_data_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML data config file."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must define a mapping: {config_path}")
    return config


def build_processed_dataset(
    dataset_name: str,
    *,
    interim_root: Path,
    processed_root: Path,
    config: dict[str, Any],
    force: bool = False,
) -> dict[str, ProcessedArtifacts]:
    """Build processed artifacts for a supported dataset."""
    normalized = dataset_name.lower()
    if normalized == "wm811k":
        return build_wm811k_processed_dataset(
            interim_root=interim_root,
            processed_root=processed_root,
            config=config,
            force=force,
        )
    raise NotImplementedError(
        f"Processed builder for dataset '{dataset_name}' is not implemented yet."
    )


def build_wm811k_processed_dataset(
    *,
    interim_root: Path,
    processed_root: Path,
    config: dict[str, Any],
    force: bool = False,
) -> dict[str, ProcessedArtifacts]:
    """Build training-ready WM-811K processed artifacts."""
    resize_mode = str(config.get("resize_mode", "nearest")).lower()
    if resize_mode != "nearest":
        raise ValueError(
            "WM-811K processing currently supports only `resize_mode: nearest`."
        )

    storage_config = config.get("storage", {})
    if storage_config is None:
        storage_config = {}
    if not isinstance(storage_config, dict):
        raise ValueError("`storage` in the config must be a mapping.")

    compression = str(storage_config.get("compression", "lzf")).lower()
    if compression != "lzf":
        raise ValueError(
            "WM-811K processing currently supports only `storage.compression: lzf`."
        )

    chunk_config = storage_config.get("chunks", {})
    if chunk_config is None:
        chunk_config = {}
    if not isinstance(chunk_config, dict):
        raise ValueError("`storage.chunks` in the config must be a mapping.")

    chunks_enabled = bool(chunk_config.get("enabled", True))
    samples_per_chunk = int(chunk_config.get("samples_per_chunk", 64))
    if samples_per_chunk <= 0:
        raise ValueError("`storage.chunks.samples_per_chunk` must be a positive integer.")

    interim_h5_path = interim_root / "wm811k.h5"
    interim_index_path = interim_root / "wm811k_index.parquet"
    if not interim_h5_path.exists() or not interim_index_path.exists():
        raise FileNotFoundError(
            "WM-811K interim artifacts were not found. Run `prepare_data.py` first."
        )

    index_df = pd.read_parquet(interim_index_path)
    if "is_labeled" not in index_df.columns:
        raise ValueError("WM-811K interim index is missing required column `is_labeled`.")

    target_root = processed_root / "wm811k"
    labeled_df = index_df[index_df["is_labeled"]].reset_index(drop=True)
    unlabeled_df = index_df[~index_df["is_labeled"]].reset_index(drop=True)

    artifacts = {
        "labeled": _build_wm811k_subset(
            subset_name="labeled",
            subset_df=labeled_df,
            interim_h5_path=interim_h5_path,
            subset_root=target_root / "labeled",
            force=force,
            compression=compression,
            chunks_enabled=chunks_enabled,
            samples_per_chunk=samples_per_chunk,
        ),
        "unlabeled": _build_wm811k_subset(
            subset_name="unlabeled",
            subset_df=unlabeled_df,
            interim_h5_path=interim_h5_path,
            subset_root=target_root / "unlabeled",
            force=force,
            compression=compression,
            chunks_enabled=chunks_enabled,
            samples_per_chunk=samples_per_chunk,
        ),
    }
    return artifacts


def _build_wm811k_subset(
    *,
    subset_name: str,
    subset_df: pd.DataFrame,
    interim_h5_path: Path,
    subset_root: Path,
    force: bool,
    compression: str,
    chunks_enabled: bool,
    samples_per_chunk: int,
) -> ProcessedArtifacts:
    artifacts = _get_processed_artifacts(subset_root, f"wm811k_{subset_name}_224")
    if force:
        _remove_processed_artifacts(artifacts)
    elif _processed_artifacts_exist(artifacts):
        print(f"[skip] WM-811K {subset_name} processed dataset already exists: {artifacts.h5_path}")
        return artifacts

    subset_root.mkdir(parents=True, exist_ok=True)
    ensure_h5py()
    h5py = ensure_h5py()

    num_samples = len(subset_df)
    sample_ids = subset_df["sample_id"].to_numpy(dtype=np.int64, copy=False)

    processed_index = subset_df.copy()
    processed_index = processed_index.rename(
        columns={
            "height": "src_height",
            "width": "src_width",
            "map_numel": "src_numel",
        }
    )
    processed_index["processed_height"] = WM811K_TARGET_SIZE[0]
    processed_index["processed_width"] = WM811K_TARGET_SIZE[1]
    processed_index["subset"] = subset_name

    with h5py.File(interim_h5_path, "r") as src_h5, h5py.File(artifacts.h5_path, "w") as dst_h5:
        _write_processed_attrs(
            dst_h5,
            subset_name=subset_name,
            source_file=str(interim_h5_path),
            num_samples=num_samples,
            compression=compression,
            chunks_enabled=chunks_enabled,
            samples_per_chunk=samples_per_chunk,
        )

        map_chunks = _resolve_map_chunks(
            num_samples=num_samples,
            chunks_enabled=chunks_enabled,
            samples_per_chunk=samples_per_chunk,
        )
        maps_ds = dst_h5.create_dataset(
            "maps",
            shape=(num_samples, 1, WM811K_TARGET_SIZE[0], WM811K_TARGET_SIZE[1]),
            dtype=np.uint8,
            compression=compression,
            chunks=map_chunks,
        )

        dst_h5.create_dataset("sample_id", data=sample_ids, dtype=np.int64)

        for out_idx, row in enumerate(processed_index.itertuples(index=False)):
            wafer_map = _restore_wafer_map(
                flattened_map=src_h5["maps"][int(row.sample_id)],
                map_shape=src_h5["map_shape"][int(row.sample_id)],
            )
            resized_map = _pad_and_resize_wafer_map(wafer_map, target_size=WM811K_TARGET_SIZE)
            maps_ds[out_idx, 0] = resized_map

            if out_idx > 0 and out_idx % 50_000 == 0:
                print(f"[build] Processed {out_idx}/{num_samples} WM-811K {subset_name} samples...")

    write_index_artifacts(processed_index, artifacts)
    print(f"[done] WM-811K {subset_name} processed HDF5 saved to: {artifacts.h5_path}")
    print(f"[done] WM-811K {subset_name} processed parquet saved to: {artifacts.parquet_path}")
    print(f"[done] WM-811K {subset_name} processed CSV saved to: {artifacts.index_path}")
    return artifacts


def _restore_wafer_map(flattened_map: np.ndarray, map_shape: np.ndarray) -> np.ndarray:
    shape = tuple(int(dim) for dim in np.asarray(map_shape, dtype=np.int32))
    return np.asarray(flattened_map, dtype=np.uint8).reshape(shape)


def _pad_and_resize_wafer_map(
    wafer_map: np.ndarray,
    *,
    target_size: tuple[int, int],
) -> np.ndarray:
    square_map = _pad_to_square(wafer_map, fill_value=0)
    return _resize_nearest(square_map, target_size=target_size)


def _pad_to_square(array: np.ndarray, *, fill_value: int) -> np.ndarray:
    height, width = array.shape
    side = max(height, width)
    padded = np.full((side, side), fill_value, dtype=array.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    padded[top : top + height, left : left + width] = array
    return padded


def _resize_nearest(
    array: np.ndarray,
    *,
    target_size: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_size
    src_h, src_w = array.shape
    row_idx = np.floor(np.arange(target_h) * src_h / target_h).astype(np.int64)
    col_idx = np.floor(np.arange(target_w) * src_w / target_w).astype(np.int64)
    row_idx = np.clip(row_idx, 0, src_h - 1)
    col_idx = np.clip(col_idx, 0, src_w - 1)
    return array[row_idx[:, None], col_idx[None, :]]


def _get_processed_artifacts(root: Path, dataset_slug: str) -> ProcessedArtifacts:
    return ProcessedArtifacts(
        h5_path=root / f"{dataset_slug}.h5",
        index_path=root / f"{dataset_slug}_index.csv",
        parquet_path=root / f"{dataset_slug}_index.parquet",
    )


def _remove_processed_artifacts(artifacts: ProcessedArtifacts) -> None:
    artifacts.h5_path.unlink(missing_ok=True)
    artifacts.index_path.unlink(missing_ok=True)
    artifacts.parquet_path.unlink(missing_ok=True)


def _processed_artifacts_exist(artifacts: ProcessedArtifacts) -> bool:
    return (
        artifacts.h5_path.exists()
        and artifacts.index_path.exists()
        and artifacts.parquet_path.exists()
    )


def _resolve_map_chunks(
    *,
    num_samples: int,
    chunks_enabled: bool,
    samples_per_chunk: int,
) -> tuple[int, int, int, int] | None:
    if not chunks_enabled or num_samples == 0:
        return None
    chunk_samples = min(samples_per_chunk, num_samples)
    return (chunk_samples, 1, WM811K_TARGET_SIZE[0], WM811K_TARGET_SIZE[1])


def _write_processed_attrs(
    h5_file,
    *,
    subset_name: str,
    source_file: str,
    num_samples: int,
    compression: str,
    chunks_enabled: bool,
    samples_per_chunk: int,
) -> None:
    h5_file.attrs["processed_schema_version"] = PROCESSED_SCHEMA_VERSION
    h5_file.attrs["dataset_name"] = "WM-811K"
    h5_file.attrs["subset_name"] = subset_name
    h5_file.attrs["source_file"] = source_file
    h5_file.attrs["num_samples"] = int(num_samples)
    h5_file.attrs["target_size"] = str(WM811K_TARGET_SIZE)
    h5_file.attrs["resize_mode"] = "nearest"
    h5_file.attrs["storage_keys"] = "maps,sample_id"
    h5_file.attrs["compression"] = compression
    h5_file.attrs["chunks_enabled"] = bool(chunks_enabled)
    h5_file.attrs["samples_per_chunk"] = int(samples_per_chunk)
    h5_file.attrs["padding_strategy"] = "center pad to square with constant background=0"
    h5_file.attrs["map_value_semantics"] = "0=background, 1=normal_die, 2=defect_die"
    h5_file.attrs["dynamic_masks"] = "wafer_mask=(map>0), defect_mask=(map==2)"
