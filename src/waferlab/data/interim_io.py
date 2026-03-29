"""Shared helpers for building interim HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INTERIM_SCHEMA_VERSION = "3"


@dataclass(frozen=True)
class InterimArtifacts:
    h5_path: Path
    index_path: Path
    parquet_path: Path


def ensure_h5py():
    """Import h5py lazily so non-HDF5 code paths stay lightweight."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "`h5py` is required to build interim datasets. "
            "Install it with `pip install h5py`."
        ) from exc
    return h5py


def get_interim_artifacts(interim_root: Path, dataset_slug: str) -> InterimArtifacts:
    interim_root.mkdir(parents=True, exist_ok=True)
    return InterimArtifacts(
        h5_path=interim_root / f"{dataset_slug}.h5",
        index_path=interim_root / f"{dataset_slug}_index.csv",
        parquet_path=interim_root / f"{dataset_slug}_index.parquet",
    )


def remove_artifacts(artifacts: InterimArtifacts) -> None:
    artifacts.h5_path.unlink(missing_ok=True)
    artifacts.index_path.unlink(missing_ok=True)
    artifacts.parquet_path.unlink(missing_ok=True)


def artifacts_are_current(
    artifacts: InterimArtifacts,
    required_keys: set[str],
) -> bool:
    """Return True when the existing HDF5/CSV pair matches the current schema."""
    if (
        not artifacts.h5_path.exists()
        or not artifacts.index_path.exists()
        or not artifacts.parquet_path.exists()
    ):
        return False

    h5py = ensure_h5py()
    try:
        with h5py.File(artifacts.h5_path, "r") as h5_file:
            version = str(h5_file.attrs.get("interim_schema_version", ""))
            if version != INTERIM_SCHEMA_VERSION:
                return False
            if not required_keys.issubset(set(h5_file.keys())):
                return False
    except OSError:
        return False

    return True


def write_common_attrs(
    h5_file,
    *,
    dataset_name: str,
    source_file: str,
    num_samples: int,
    map_encoding: str,
    map_value_semantics: str,
    label_schema: str,
) -> None:
    h5_file.attrs["interim_schema_version"] = INTERIM_SCHEMA_VERSION
    h5_file.attrs["dataset_name"] = dataset_name
    h5_file.attrs["source_file"] = source_file
    h5_file.attrs["num_samples"] = int(num_samples)
    h5_file.attrs["map_encoding"] = map_encoding
    h5_file.attrs["map_value_semantics"] = map_value_semantics
    h5_file.attrs["label_schema"] = label_schema


def write_string_dataset(h5_file, name: str, values: Iterable[str]) -> None:
    h5py = ensure_h5py()
    string_dtype = h5py.string_dtype(encoding="utf-8")
    h5_file.create_dataset(
        name,
        data=np.asarray(list(values), dtype=object),
        dtype=string_dtype,
    )


def write_vlen_map_dataset(
    h5_file,
    *,
    name: str,
    maps: Iterable[np.ndarray],
    num_samples: int,
    progress_label: str,
    progress_every: int = 100_000,
) -> np.ndarray:
    """Write variable-sized 2D maps as flattened uint8 vectors plus shapes."""
    h5py = ensure_h5py()
    shapes = np.empty((num_samples, 2), dtype=np.int32)
    maps_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    maps_ds = h5_file.create_dataset(name, shape=(num_samples,), dtype=maps_dtype)

    for idx, wafer_map in enumerate(maps):
        wafer_array = np.asarray(wafer_map, dtype=np.uint8)
        shapes[idx] = wafer_array.shape
        maps_ds[idx] = wafer_array.reshape(-1)
        if idx > 0 and idx % progress_every == 0:
            print(f"[build] Serialized {idx}/{num_samples} {progress_label}...")

    return shapes


def write_dense_map_dataset(
    h5_file,
    *,
    name: str,
    maps: np.ndarray,
    dtype: np.dtype | type[np.generic] = np.uint8,
) -> None:
    h5_file.create_dataset(name, data=np.asarray(maps, dtype=dtype), dtype=dtype)


def write_index_artifacts(index_df: pd.DataFrame, artifacts: InterimArtifacts) -> None:
    """Write both fast parquet and human-readable CSV index files."""
    try:
        index_df.to_parquet(artifacts.parquet_path, index=False)
    except (ImportError, ValueError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Writing parquet index files requires a parquet engine such as `pyarrow`."
        ) from exc

    index_df.to_csv(artifacts.index_path, index=False)
