"""Lightweight registry for models, optimizers, and schedulers.

Usage::

    from waferlab.registry import MODEL_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

    # Register a custom model builder (place this in models/xxx.py)
    @MODEL_REGISTRY.register("my_vit")
    def build_my_vit(config):
        ...

    # Build from config
    model = MODEL_REGISTRY.build("resnet18", config)
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


class Registry:
    """Name -> factory mapping with a decorator-based registration API.

    Parameters
    ----------
    name : str
        Human-readable name used in error messages.
    discover_package : str or None
        If set, the package will be lazily imported the first time
        :meth:`build`, :meth:`keys`, or ``in`` is used, which triggers
        auto-discovery of all modules in that package (and their
        ``@register`` decorators).
    """

    def __init__(self, name: str, *, discover_package: str | None = None) -> None:
        self.name = name
        self._registry: dict[str, Callable] = {}
        self._discover_package = discover_package
        self._discovered = False

    def _ensure_discovered(self) -> None:
        """Lazily import *discover_package* to trigger registrations."""
        if not self._discovered and self._discover_package:
            self._discovered = True
            importlib.import_module(self._discover_package)

    def register(self, key: str) -> Callable:
        """Decorator that registers a builder function under *key*."""
        def wrapper(fn: Callable) -> Callable:
            if key in self._registry:
                raise KeyError(f"{self.name} already has '{key}' registered.")
            self._registry[key] = fn
            return fn
        return wrapper

    def build(self, key: str, config: dict[str, Any]) -> Any:
        """Look up *key* and call the builder with *config*."""
        self._ensure_discovered()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(none)"
            raise KeyError(
                f"Unknown {self.name} '{key}'. Available: {available}"
            )
        return self._registry[key](config)

    def keys(self) -> list[str]:
        self._ensure_discovered()
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        self._ensure_discovered()
        return key in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, keys={self.keys()})"


# ── Global registries ────────────────────────────────────────────────

MODEL_REGISTRY = Registry("model", discover_package="waferlab.models")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")


# ── Built-in optimizer builders ──────────────────────────────────────

@OPTIMIZER_REGISTRY.register("adamw")
def _build_adamw(config: dict[str, Any]):
    import torch.optim as optim
    params = config["params"]  # must be injected by caller
    return optim.AdamW(
        params,
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


@OPTIMIZER_REGISTRY.register("sgd")
def _build_sgd(config: dict[str, Any]):
    import torch.optim as optim
    params = config["params"]
    return optim.SGD(
        params,
        lr=float(config.get("lr", 1e-2)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


# ── Built-in scheduler builders ─────────────────────────────────────

@SCHEDULER_REGISTRY.register("cosine")
def _build_cosine(config: dict[str, Any]):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    optimizer = config["optimizer"]
    return CosineAnnealingLR(
        optimizer,
        T_max=int(config.get("epochs", 30)),
    )


@SCHEDULER_REGISTRY.register("step")
def _build_step(config: dict[str, Any]):
    from torch.optim.lr_scheduler import StepLR
    optimizer = config["optimizer"]
    return StepLR(
        optimizer,
        step_size=int(config.get("step_size", 10)),
        gamma=float(config.get("gamma", 0.1)),
    )


@SCHEDULER_REGISTRY.register("none")
def _build_none(config: dict[str, Any]):
    """No-op scheduler (constant LR)."""
    return None
