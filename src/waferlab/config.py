"""Shared YAML config loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings, with *override* taking precedence."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with optional recursive ``_base_`` inheritance."""
    return _load_yaml_config(Path(path), stack=())


def _load_yaml_config(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"Config inheritance cycle detected: {chain}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, Mapping):
        raise ValueError(f"Config file must define a mapping: {resolved_path}")

    base_entry = payload.get("_base_")
    merged: dict[str, Any] = {}
    if base_entry is not None:
        if isinstance(base_entry, str):
            base_paths = [base_entry]
        elif isinstance(base_entry, list) and all(isinstance(item, str) for item in base_entry):
            base_paths = list(base_entry)
        else:
            raise ValueError(f"`_base_` must be a string or list of strings: {resolved_path}")

        for raw_base in base_paths:
            base_path = (resolved_path.parent / raw_base).resolve()
            base_config = _load_yaml_config(base_path, stack=(*stack, resolved_path))
            merged = _deep_merge(merged, base_config)

    current = {key: value for key, value in payload.items() if key != "_base_"}
    return _deep_merge(merged, current)
