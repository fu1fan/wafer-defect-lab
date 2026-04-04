"""Model package – auto-imports all submodules to trigger @register decorators."""

import importlib
import pkgutil


def _auto_import_models() -> None:
    """Import every module under ``waferlab.models`` so that their
    ``@MODEL_REGISTRY.register(...)`` decorators are executed."""
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        __path__, prefix=__name__ + "."
    ):
        importlib.import_module(modname)


_auto_import_models()

# Re-export commonly used symbols for convenience.
from .resnet import (  # noqa: E402, F401
    FAILURE_TYPE_NAMES,
    FAILURE_TYPE_TO_IDX,
    WaferClassifier,
    build_classifier,
)

__all__ = [
    "FAILURE_TYPE_NAMES",
    "FAILURE_TYPE_TO_IDX",
    "WaferClassifier",
    "build_classifier",
]
