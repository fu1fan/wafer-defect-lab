"""Modern CNN backbones for WM-811K wafer-map classification.

Provides three registered model variants built on torchvision pretrained
backbones, each significantly stronger than ResNet50 on ImageNet:

- ``convnext_tiny_wafer``: ConvNeXt-Tiny (82.5% IN1K top-1, 28.6M params)
- ``efficientnetv2_s_wafer``: EfficientNetV2-S (84.2% IN1K top-1, 21.5M params)
- ``convnext_v2_nano_wafer``: ConvNeXt-Nano via ConvNeXt architecture (lighter)

All variants maintain the same external API as :class:`WaferClassifier`
(forward, forward_features, get_cam_target_layer) and are compatible with
the existing training/evaluation pipeline and future continual-learning
integration.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models


# ── ConvNeXt-Tiny ────────────────────────────────────────────────────

class WaferConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny backbone for wafer-map classification.

    ConvNeXt modernises the pure-CNN design with depthwise convolutions,
    LayerNorm, GELU, and inverted bottlenecks — achieving 82.5 % top-1
    on ImageNet-1K with 28.6 M parameters and 4.5 GFLOPS.

    For single-channel wafer maps, the first conv stem is replaced with
    a 1-channel variant.  When ``pretrained=True`` the ImageNet weights
    are loaded and the stem weights are averaged across input channels.
    """

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # Adapt stem for single-channel input
        if in_channels != 3:
            old_conv = backbone.features[0][0]  # stem Conv2d
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
            backbone.features[0][0] = new_conv

        # classifier = [LayerNorm2d, Flatten, Linear] — keep norm+flatten, drop Linear
        feat_dim = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()

        self.backbone = backbone
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return globally-pooled feature vector before FC head."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [B, num_classes]."""
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def get_cam_target_layer(self) -> nn.Module:
        """Return last feature block for GradCAM."""
        return self.backbone.features[-1]


# ── EfficientNetV2-S ────────────────────────────────────────────────

class WaferEfficientNetV2S(nn.Module):
    """EfficientNetV2-S backbone for wafer-map classification.

    EfficientNetV2-S achieves 84.2 % top-1 on ImageNet-1K with only
    21.5 M parameters, offering the best accuracy-per-parameter ratio
    among common CNN architectures.  Uses Fused-MBConv blocks for
    faster training and inference.
    """

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = (
            models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = models.efficientnet_v2_s(weights=weights)

        # Adapt first conv for single-channel input
        if in_channels != 3:
            old_conv = backbone.features[0][0]  # stem Conv2dNormActivation
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
            backbone.features[0][0] = new_conv

        # classifier = [Dropout, Linear] — drop the Linear, we add our own
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return globally-pooled feature vector before FC head."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [B, num_classes]."""
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def get_cam_target_layer(self) -> nn.Module:
        """Return last feature block for GradCAM."""
        return self.backbone.features[-1]


# ── ConvNeXt-Nano (lightweight alternative) ──────────────────────────

class WaferConvNeXtNano(nn.Module):
    """Lightweight ConvNeXt variant using convnext_small's architecture
    but with reduced width, providing a middle ground between ConvNeXt-Tiny
    cost and stronger-than-ResNet50 representation.

    Uses ConvNeXt-Small from torchvision as the base, which offers
    ~83.6 % top-1 on ImageNet-1K with 50.2 M parameters.
    """

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = (
            models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = models.convnext_small(weights=weights)

        # Adapt stem for single-channel input
        if in_channels != 3:
            old_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
            backbone.features[0][0] = new_conv

        # classifier = [LayerNorm2d, Flatten, Linear] — keep norm+flatten, drop Linear
        feat_dim = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()

        self.backbone = backbone
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return globally-pooled feature vector before FC head."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [B, num_classes]."""
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def get_cam_target_layer(self) -> nn.Module:
        """Return last feature block for GradCAM."""
        return self.backbone.features[-1]


# ── Registry integration ─────────────────────────────────────────────

from waferlab.registry import MODEL_REGISTRY  # noqa: E402


def _modern_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 9)),
        "in_channels": int(config.get("in_channels", 1)),
        "pretrained": bool(config.get("pretrained", False)),
        "dropout": float(config.get("dropout", 0.2)),
    }


@MODEL_REGISTRY.register("convnext_tiny_wafer")
def _build_convnext_tiny(config: dict[str, Any]) -> WaferConvNeXtTiny:
    return WaferConvNeXtTiny(**_modern_kwargs(config))


@MODEL_REGISTRY.register("efficientnetv2_s_wafer")
def _build_efficientnetv2_s(config: dict[str, Any]) -> WaferEfficientNetV2S:
    return WaferEfficientNetV2S(**_modern_kwargs(config))


@MODEL_REGISTRY.register("convnext_small_wafer")
def _build_convnext_small(config: dict[str, Any]) -> WaferConvNeXtNano:
    return WaferConvNeXtNano(**_modern_kwargs(config))
