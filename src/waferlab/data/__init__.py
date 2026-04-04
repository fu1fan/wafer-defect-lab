"""Data package exports and auto-discovery hooks for data builders."""

import importlib
import pkgutil


def _auto_import_data_modules() -> None:
    """Import every module under ``waferlab.data`` to trigger registrations."""
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        __path__, prefix=__name__ + "."
    ):
        importlib.import_module(modname)


_auto_import_data_modules()

from .dataloaders import (
    build_classification_dataloaders,
    build_dataloaders,
    build_eval_dataloader,
    build_wm811k_dataloader,
    load_dataloader_config,
)
from .download import DATASETS, AUTO_ALIASES, resolve_dataset_names, download_dataset
from .datasets import WM811KProcessedDataset, build_wm811k_dataset
from .interim import (
    MIXEDWM38_LABEL_NAMES,
    build_interim_dataset,
    build_interim_datasets,
    build_mixedwm38_interim_dataset,
    build_wm811k_interim_dataset,
)
from .processed import build_processed_dataset, build_wm811k_processed_dataset, load_data_config
from .transforms import (
    WaferAugmentation,
    InjectFailureTypeIdx,
    compose,
    prepare_input,
    DEFAULT_NORM_SCALE,
)

__all__ = [
    "DATASETS",
    "AUTO_ALIASES",
    "WM811KProcessedDataset",
    "build_classification_dataloaders",
    "build_dataloaders",
    "build_eval_dataloader",
    "build_wm811k_dataloader",
    "build_wm811k_dataset",
    "load_dataloader_config",
    "resolve_dataset_names",
    "download_dataset",
    "MIXEDWM38_LABEL_NAMES",
    "build_interim_dataset",
    "build_interim_datasets",
    "build_mixedwm38_interim_dataset",
    "build_wm811k_interim_dataset",
    "build_processed_dataset",
    "build_wm811k_processed_dataset",
    "load_data_config",
    "WaferAugmentation",
    "InjectFailureTypeIdx",
    "compose",
    "prepare_input",
    "DEFAULT_NORM_SCALE",
]
