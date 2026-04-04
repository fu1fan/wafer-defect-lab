"""PyTorch dataset wrappers for processed wafer-map artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .interim_io import ensure_h5py
from .processed import validate_wm811k_processed_preprocess_mode


class WM811KProcessedDataset(Dataset[dict[str, Any]]):
    """Read processed WM-811K samples from HDF5 + index artifacts."""

    def __init__(
        self,
        processed_root: str | Path,
        *,
        subset: str = "labeled",
        transform: Any | None = None,
        include_metadata: bool = False,
        return_masks: bool = True,
        return_float: bool = True,
        filters: Mapping[str, Any] | None = None,
        dataset_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._h5_file: Any | None = None  # init early to avoid __del__ crash

        self.processed_root = Path(processed_root)
        self.subset = subset
        self.transform = transform
        self.include_metadata = include_metadata
        self.return_masks = return_masks
        self.return_float = return_float
        self.filters = dict(filters or {})
        self.dataset_config = dict(dataset_config or {})

        if dataset_config is not None and not isinstance(dataset_config, Mapping):
            raise ValueError("`dataset_config` must be a mapping when provided.")

        subset_root = self.processed_root / "wm811k" / subset
        self.h5_path = subset_root / f"wm811k_{subset}_224.h5"

        if self.dataset_config:
            validate_wm811k_processed_preprocess_mode(
                data_config=self.dataset_config,
                processed_root=self.processed_root,
                subset=self.subset,
            )

        # Prefer parquet index; fall back to CSV if unavailable.
        parquet_path = subset_root / f"wm811k_{subset}_224_index.parquet"
        csv_path = subset_root / f"wm811k_{subset}_224_index.csv"
        if parquet_path.exists():
            self.index_path = parquet_path
        elif csv_path.exists():
            self.index_path = csv_path
        else:
            raise FileNotFoundError(
                f"Processed index not found (tried .parquet and .csv): {parquet_path}"
            )

        if not self.h5_path.exists():
            raise FileNotFoundError(f"Processed HDF5 not found: {self.h5_path}")

        if self.index_path.suffix == ".parquet":
            self.index_df = pd.read_parquet(self.index_path)
        else:
            self.index_df = pd.read_csv(self.index_path)
        # Store original HDF5 row indices before filtering so that
        # __getitem__ can fetch the correct row from the HDF5 file.
        self.index_df["_h5_idx"] = np.arange(len(self.index_df), dtype=np.int64)
        self.index_df = _apply_filters(self.index_df, self.filters)
        self._h5_indices = self.index_df["_h5_idx"].to_numpy(dtype=np.int64)
        self._h5_file: Any | None = None

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.index_df.iloc[index]
        h5_idx = int(self._h5_indices[index])
        maps_ds = cast(Any, self._get_h5_file()["maps"])

        wafer_map = torch.from_numpy(maps_ds[h5_idx])
        if self.return_float:
            wafer_map = wafer_map.to(dtype=torch.float32)

        sample: dict[str, Any] = {
            "image": wafer_map,
            "sample_id": int(row["sample_id"]),
        }

        if "is_labeled" in row.index:
            sample["is_labeled"] = bool(row["is_labeled"])
        if "is_normal" in row.index:
            sample["is_normal"] = bool(row["is_normal"])
            sample["label"] = 0 if bool(row["is_normal"]) else 1

        if self.return_masks:
            source_map = wafer_map if not self.return_float else wafer_map.to(dtype=torch.uint8)
            sample["wafer_mask"] = (source_map > 0).to(dtype=torch.uint8)
            sample["defect_mask"] = (source_map == 2).to(dtype=torch.uint8)

        if self.include_metadata:
            sample["metadata"] = row.to_dict()

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except (AttributeError, TypeError):
                pass
            self._h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _get_h5_file(self):
        if self._h5_file is None:
            h5py = ensure_h5py()
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __del__(self) -> None:
        self.close()


def _apply_filters(index_df: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    filtered_df = index_df
    for column, expected in filters.items():
        if column not in filtered_df.columns:
            raise KeyError(f"Filter column not found in index: {column}")
        if isinstance(expected, list):
            filtered_df = filtered_df[filtered_df[column].isin(expected)]
        else:
            filtered_df = filtered_df[filtered_df[column] == expected]
    return filtered_df.reset_index(drop=True)


# ── Registry integration ─────────────────────────────────────────────

from ..registry import DATASET_REGISTRY  # noqa: E402


@DATASET_REGISTRY.register("wm811k")
def build_wm811k_dataset(config: dict[str, Any]) -> WM811KProcessedDataset:
    """Build a :class:`WM811KProcessedDataset` from a config dict."""
    filters = config.get("filters", {})
    if filters is None:
        filters = {}
    return WM811KProcessedDataset(
        config["processed_root"],
        subset=str(config.get("subset", "labeled")),
        transform=config.get("transform"),
        include_metadata=bool(config.get("include_metadata", False)),
        return_masks=bool(config.get("return_masks", True)),
        return_float=bool(config.get("return_float", True)),
        filters=filters,
        dataset_config=config.get("dataset_config"),
    )
