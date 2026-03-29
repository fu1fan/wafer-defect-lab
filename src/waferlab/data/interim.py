"""Dataset-specific interim preprocessing entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .download import DATASETS
from .interim_io import (
    InterimArtifacts,
    artifacts_are_current,
    ensure_h5py,
    get_interim_artifacts,
    remove_artifacts,
    write_common_attrs,
    write_dense_map_dataset,
    write_index_artifacts,
    write_string_dataset,
    write_vlen_map_dataset,
)
from .pandas_compat import read_legacy_pickle


MIXEDWM38_LABEL_NAMES = [
    "Center",
    "Donut",
    "Edge_Loc",
    "Edge_Ring",
    "Loc",
    "Near_Full",
    "Scratch",
    "Random",
]


def _normalize_label_field(value: Any) -> str:
    """Flatten the legacy WM-811K nested label container into a plain string."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return str(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        current = value
        while isinstance(current, (list, tuple)) and current:
            current = current[0]
        return "" if isinstance(current, (list, tuple)) else str(current)
    if value is None:
        return ""
    return str(value)


def _prepare_artifacts(
    dataset_slug: str,
    interim_root: Path,
    required_keys: set[str],
    force: bool,
) -> tuple[InterimArtifacts, bool]:
    artifacts = get_interim_artifacts(interim_root, dataset_slug)
    if force:
        remove_artifacts(artifacts)
    elif artifacts_are_current(artifacts, required_keys):
        print(f"[skip] {dataset_slug} interim dataset already exists: {artifacts.h5_path}")
        return artifacts, False
    elif artifacts.h5_path.exists() or artifacts.index_path.exists():
        print(f"[build] Existing {dataset_slug} interim artifacts are outdated. Rebuilding...")
        remove_artifacts(artifacts)
    return artifacts, True


def build_wm811k_interim_dataset(
    raw_root: Path,
    interim_root: Path,
    force: bool = False,
) -> InterimArtifacts:
    """Build the WM-811K interim dataset."""
    ensure_h5py()
    required_keys = {
        "maps",
        "map_shape",
        "sample_id",
        "orig_index",
        "die_size",
        "wafer_index",
        "lot_name",
        "split_label",
        "failure_type",
        "is_labeled",
        "label_count",
        "is_normal",
    }
    artifacts, should_build = _prepare_artifacts("wm811k", interim_root, required_keys, force)
    if not should_build:
        return artifacts

    raw_path = raw_root / DATASETS["WM-811K"]["target_dir"] / "LSWMD.pkl"
    print("[load] Loading WM-811K with legacy pickle compatibility...")
    df: pd.DataFrame = cast(pd.DataFrame, read_legacy_pickle(raw_path))
    print(f"[done] WM-811K loaded successfully: {df.shape}")

    split_label = df["trianTestLabel"].map(_normalize_label_field)
    failure_type = df["failureType"].map(_normalize_label_field)
    is_labeled = (split_label != "") & (failure_type != "")
    label_count = is_labeled.astype(np.int8)
    is_normal = failure_type.eq("none")

    sample_id = np.arange(len(df), dtype=np.int64)
    index_df: pd.DataFrame = df.loc[:, ["lotName", "waferIndex", "dieSize"]].copy()
    index_df = index_df.rename(
        columns={
            "lotName": "lot_name",
            "waferIndex": "wafer_index",
            "dieSize": "die_size",
        },
    )
    index_df.insert(0, "sample_id", sample_id)
    index_df.insert(1, "orig_index", df.index.to_numpy(dtype=np.int64, copy=False))
    index_df["split_label"] = split_label
    index_df["failure_type"] = failure_type
    index_df["is_labeled"] = is_labeled.to_numpy(dtype=bool, copy=False)
    index_df["label_count"] = label_count.to_numpy(dtype=np.int8, copy=False)
    index_df["is_normal"] = is_normal.to_numpy(dtype=bool, copy=False)

    h5py = ensure_h5py()
    print(f"[build] Writing WM-811K interim HDF5: {artifacts.h5_path}")
    with h5py.File(artifacts.h5_path, "w") as h5_file:
        write_common_attrs(
            h5_file,
            dataset_name="WM-811K",
            source_file="data/raw/wm811k/LSWMD.pkl",
            num_samples=len(df),
            map_encoding="flattened uint8 arrays with shapes stored in /map_shape",
            map_value_semantics="0=background, 1=normal_die, 2=defect_die",
            label_schema="single-label with optional unlabeled samples",
        )
        map_shapes = write_vlen_map_dataset(
            h5_file,
            name="maps",
            maps=df["waferMap"],
            num_samples=len(df),
            progress_label="WM-811K wafer maps",
        )
        h5_file.create_dataset("map_shape", data=map_shapes, dtype=np.int32)
        h5_file.create_dataset("sample_id", data=sample_id, dtype=np.int64)
        h5_file.create_dataset(
            "orig_index",
            data=index_df["orig_index"].to_numpy(dtype=np.int64, copy=False),
            dtype=np.int64,
        )
        h5_file.create_dataset(
            "die_size",
            data=index_df["die_size"].to_numpy(dtype=np.int32, copy=False),
            dtype=np.int32,
        )
        h5_file.create_dataset(
            "wafer_index",
            data=index_df["wafer_index"].to_numpy(dtype=np.int16, copy=False),
            dtype=np.int16,
        )
        write_string_dataset(h5_file, "lot_name", index_df["lot_name"].astype(str))
        write_string_dataset(h5_file, "split_label", index_df["split_label"].astype(str))
        write_string_dataset(h5_file, "failure_type", index_df["failure_type"].astype(str))
        h5_file.create_dataset(
            "is_labeled",
            data=index_df["is_labeled"].to_numpy(dtype=bool, copy=False),
            dtype=np.bool_,
        )
        h5_file.create_dataset(
            "label_count",
            data=index_df["label_count"].to_numpy(dtype=np.int8, copy=False),
            dtype=np.int8,
        )
        h5_file.create_dataset(
            "is_normal",
            data=index_df["is_normal"].to_numpy(dtype=bool, copy=False),
            dtype=np.bool_,
        )

    index_df["height"] = map_shapes[:, 0]
    index_df["width"] = map_shapes[:, 1]
    index_df["map_numel"] = map_shapes[:, 0] * map_shapes[:, 1]
    write_index_artifacts(index_df, artifacts)
    print(f"[done] WM-811K interim HDF5 saved to: {artifacts.h5_path}")
    print(f"[done] WM-811K index parquet saved to: {artifacts.parquet_path}")
    print(f"[done] WM-811K index CSV saved to: {artifacts.index_path}")
    return artifacts


def build_mixedwm38_interim_dataset(
    raw_root: Path,
    interim_root: Path,
    force: bool = False,
) -> InterimArtifacts:
    """Build the MixedWM38 interim dataset."""
    ensure_h5py()
    required_keys = {
        "maps",
        "labels",
        "label_names",
        "sample_id",
        "orig_index",
        "label_count",
        "is_labeled",
        "is_normal",
        "has_mixed_defect",
    }
    artifacts, should_build = _prepare_artifacts("mixedwm38", interim_root, required_keys, force)
    if not should_build:
        return artifacts

    raw_path = raw_root / DATASETS["MixedWM38"]["target_dir"] / "Wafer_Map_Datasets.npz"
    print("[load] Loading MixedWM38 NPZ arrays...")
    with np.load(raw_path, allow_pickle=True) as mixed_data:
        maps = np.asarray(mixed_data["arr_0"], dtype=np.uint8)
        labels = np.asarray(mixed_data["arr_1"], dtype=np.uint8)

    if maps.ndim != 3 or labels.ndim != 2 or maps.shape[0] != labels.shape[0]:
        raise ValueError(
            "MixedWM38 NPZ structure is invalid. Expected maps[N,H,W] and labels[N,C]."
        )

    sample_id = np.arange(len(maps), dtype=np.int64)
    label_count = labels.sum(axis=1, dtype=np.int16)
    is_normal = label_count == 0
    has_mixed_defect = label_count > 1
    is_labeled = np.ones(len(maps), dtype=bool)

    index_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "orig_index": sample_id,
            "height": maps.shape[1],
            "width": maps.shape[2],
            "label_count": label_count.astype(np.int16, copy=False),
            "is_labeled": is_labeled,
            "is_normal": is_normal,
            "has_mixed_defect": has_mixed_defect,
        }
    )
    for label_idx, label_name in enumerate(MIXEDWM38_LABEL_NAMES):
        index_df[f"label_{label_name.lower()}"] = labels[:, label_idx].astype(np.uint8, copy=False)

    h5py = ensure_h5py()
    print(f"[build] Writing MixedWM38 interim HDF5: {artifacts.h5_path}")
    with h5py.File(artifacts.h5_path, "w") as h5_file:
        write_common_attrs(
            h5_file,
            dataset_name="MixedWM38",
            source_file="data/raw/mixedwm38/Wafer_Map_Datasets.npz",
            num_samples=len(maps),
            map_encoding="dense uint8 tensor stored in /maps with shape [N, 52, 52]",
            map_value_semantics="0=blank, 1=normal_die, 2=defect_die, 3=unexpected raw value observed in source NPZ",
            label_schema="8-dimensional multi-hot labels for mixed defect patterns",
        )
        write_dense_map_dataset(h5_file, name="maps", maps=maps, dtype=np.uint8)
        h5_file.create_dataset("labels", data=labels, dtype=np.uint8)
        write_string_dataset(h5_file, "label_names", MIXEDWM38_LABEL_NAMES)
        h5_file.create_dataset("sample_id", data=sample_id, dtype=np.int64)
        h5_file.create_dataset("orig_index", data=sample_id, dtype=np.int64)
        h5_file.create_dataset("label_count", data=label_count, dtype=np.int16)
        h5_file.create_dataset("is_labeled", data=is_labeled, dtype=np.bool_)
        h5_file.create_dataset("is_normal", data=is_normal, dtype=np.bool_)
        h5_file.create_dataset("has_mixed_defect", data=has_mixed_defect, dtype=np.bool_)

    write_index_artifacts(index_df, artifacts)
    print(f"[done] MixedWM38 interim HDF5 saved to: {artifacts.h5_path}")
    print(f"[done] MixedWM38 index parquet saved to: {artifacts.parquet_path}")
    print(f"[done] MixedWM38 index CSV saved to: {artifacts.index_path}")
    return artifacts


def build_interim_dataset(
    dataset_name: str,
    raw_root: Path,
    interim_root: Path,
    force: bool = False,
) -> InterimArtifacts:
    """Dispatch interim preprocessing by dataset name."""
    if dataset_name == "WM-811K":
        return build_wm811k_interim_dataset(raw_root=raw_root, interim_root=interim_root, force=force)
    if dataset_name == "MixedWM38":
        return build_mixedwm38_interim_dataset(
            raw_root=raw_root,
            interim_root=interim_root,
            force=force,
        )
    raise ValueError(f"Unsupported dataset for interim preprocessing: {dataset_name}")


def build_interim_datasets(
    dataset_names: list[str],
    raw_root: Path,
    interim_root: Path,
    force: bool = False,
) -> dict[str, InterimArtifacts]:
    """Build interim datasets for every requested raw dataset."""
    artifacts: dict[str, InterimArtifacts] = {}
    for dataset_name in dataset_names:
        artifacts[dataset_name] = build_interim_dataset(
            dataset_name,
            raw_root=raw_root,
            interim_root=interim_root,
            force=force,
        )
    return artifacts
