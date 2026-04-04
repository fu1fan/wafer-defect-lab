"""Lightweight registry for models, optimizers, and schedulers.

Usage::

    from waferlab.registry import MODEL_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

    # Register a custom model builder
    @MODEL_REGISTRY.register("my_vit")
    def build_my_vit(config):
        ...

    # Build from config
    model = MODEL_REGISTRY.build("resnet18", config)
"""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    """Name -> factory mapping with a decorator-based registration API."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, Callable] = {}

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
        if key not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(none)"
            raise KeyError(
                f"Unknown {self.name} '{key}'. Available: {available}"
            )
        return self._registry[key](config)

    def keys(self) -> list[str]:
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, keys={self.keys()})"


# ── Global registries ────────────────────────────────────────────────

MODEL_REGISTRY = Registry("model")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")


# ── Built-in model builders ──────────────────────────────────────────

@MODEL_REGISTRY.register("resnet18")
def _build_resnet18(config: dict[str, Any]):
    from .models.resnet import WaferClassifier
    return WaferClassifier(arch="resnet18", **_classifier_kwargs(config))


@MODEL_REGISTRY.register("resnet34")
def _build_resnet34(config: dict[str, Any]):
    from .models.resnet import WaferClassifier
    return WaferClassifier(arch="resnet34", **_classifier_kwargs(config))


@MODEL_REGISTRY.register("resnet50")
def _build_resnet50(config: dict[str, Any]):
    from .models.resnet import WaferClassifier
    return WaferClassifier(arch="resnet50", **_classifier_kwargs(config))


def _classifier_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 2)),
        "in_channels": int(config.get("in_channels", 1)),
        "pretrained": bool(config.get("pretrained", False)),
        "dropout": float(config.get("dropout", 0.0)),
    }


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
