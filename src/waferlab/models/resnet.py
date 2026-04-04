"""Wafer-level classification model built on ResNet backbones.

Supports:
- Single-channel (1ch) input with modified conv1, or 3-channel repeat for
  leveraging ImageNet-pretrained weights.
- Binary (normal/abnormal) or multi-class (failure-type pattern) head.
- Spatial feature extraction for downstream anomaly methods (PaDiM, PatchCore).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models


# Failure-type class names used in multi-class mode (WM-811K).
# Index 0 = normal; indices 1–8 = anomaly patterns.
FAILURE_TYPE_NAMES: list[str] = [
    "none",       # 0 – normal
    "Center",     # 1
    "Donut",      # 2
    "Edge-Loc",   # 3
    "Edge-Ring",  # 4
    "Loc",        # 5
    "Near-full",  # 6
    "Random",     # 7
    "Scratch",    # 8
]

FAILURE_TYPE_TO_IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(FAILURE_TYPE_NAMES)
}


def _get_resnet_constructor(arch: str):
    constructors = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if arch not in constructors:
        raise ValueError(f"Unsupported backbone: {arch}. Choose from {list(constructors)}")
    return constructors[arch]


class WaferClassifier(nn.Module):
    """ResNet-based wafer-level classifier.

    Parameters
    ----------
    arch : str
        ``"resnet18"`` / ``"resnet34"`` / ``"resnet50"``.
    num_classes : int
        2 for binary (normal/abnormal), 9 for WM-811K failure-type patterns.
    in_channels : int
        1 = native single-channel wafer map; 3 = channel-replicated input.
    pretrained : bool
        Load ImageNet weights (effective only when ``in_channels == 3``,
        or when ``in_channels == 1`` with averaged conv1 weights).
    dropout : float
        Dropout probability before the final FC layer.
    """

    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = _get_resnet_constructor(arch)(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,  # type: ignore[arg-type]
                stride=old_conv.stride,  # type: ignore[arg-type]
                padding=old_conv.padding,  # type: ignore[arg-type]
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    backbone.conv1.weight.copy_(
                        old_conv.weight.mean(dim=1, keepdim=True)
                    )

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # type: ignore[assignment]

        self.backbone = backbone
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled feature vector *before* the FC head.

        Useful for feature-bank anomaly methods (PaDiM, PatchCore).
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits ``[B, num_classes]``."""
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_cam_target_layer(self) -> nn.Module:
        """Return the last conv block for GradCAM / response-map hooks."""
        return self.backbone.layer4  # type: ignore[return-value]


def build_classifier(config: dict[str, Any]) -> WaferClassifier:
    """Build a :class:`WaferClassifier` from a flat config dict."""
    return WaferClassifier(
        arch=str(config.get("arch", "resnet18")),
        num_classes=int(config.get("num_classes", 2)),
        in_channels=int(config.get("in_channels", 1)),
        pretrained=bool(config.get("pretrained", False)),
        dropout=float(config.get("dropout", 0.0)),
    )


# ── Registry integration ─────────────────────────────────────────────

from waferlab.registry import MODEL_REGISTRY  # noqa: E402


def _classifier_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 2)),
        "in_channels": int(config.get("in_channels", 1)),
        "pretrained": bool(config.get("pretrained", False)),
        "dropout": float(config.get("dropout", 0.0)),
    }


@MODEL_REGISTRY.register("resnet18")
def _build_resnet18(config: dict[str, Any]) -> WaferClassifier:
    return WaferClassifier(arch="resnet18", **_classifier_kwargs(config))


@MODEL_REGISTRY.register("resnet34")
def _build_resnet34(config: dict[str, Any]) -> WaferClassifier:
    return WaferClassifier(arch="resnet34", **_classifier_kwargs(config))


@MODEL_REGISTRY.register("resnet50")
def _build_resnet50(config: dict[str, Any]) -> WaferClassifier:
    return WaferClassifier(arch="resnet50", **_classifier_kwargs(config))
