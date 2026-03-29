"""PyTorch dataset wrappers for processed wafer-map artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd
import torch
from torch.utils.data import Dataset

from .interim_io import ensure_h5py


class WM811KProcessedDataset(Dataset[dict[str, Any]]):
    """Read processed WM-811K samples from HDF5 + parquet index artifacts."""

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
    ) -> None:
        self.processed_root = Path(processed_root)
        self.subset = subset
        self.transform = transform
        self.include_metadata = include_metadata
        self.return_masks = return_masks
        self.return_float = return_float
        self.filters = dict(filters or {})

        subset_root = self.processed_root / "wm811k" / subset
        self.h5_path = subset_root / f"wm811k_{subset}_224.h5"
        self.index_path = subset_root / f"wm811k_{subset}_224_index.parquet"

        if not self.h5_path.exists():
            raise FileNotFoundError(f"Processed HDF5 not found: {self.h5_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Processed index not found: {self.index_path}")

        self.index_df = pd.read_parquet(self.index_path)
        self.index_df = _apply_filters(self.index_df, self.filters)
        self._h5_file: Any | None = None

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.index_df.iloc[index]
        maps_ds = cast(Any, self._get_h5_file()["maps"])

        wafer_map = torch.from_numpy(maps_ds[index])
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
