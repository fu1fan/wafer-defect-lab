"""ResNet50 variants with GeM pooling and/or CBAM attention for WM-811K.

Provides three registered model variants:
- ``resnet50_gem``: ResNet50 + GeM pooling (replaces GAP)
- ``resnet50_cbam``: ResNet50 + CBAM after layer3 and layer4
- ``resnet50_gem_cbam``: ResNet50 + GeM pooling + CBAM

All variants maintain the same external API as :class:`WaferClassifier`
(forward, forward_features, get_cam_target_layer).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── GeM Pooling ──────────────────────────────────────────────────────

class GeMPooling(nn.Module):
    """Generalized Mean Pooling with learnable power parameter.

    When p=1 → average pooling; p→∞ → max pooling.
    Beneficial for sparse anomaly patterns where high-activation regions
    carry more discriminative information.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.clamp(min=self.eps)
            .pow(self.p)
            .mean(dim=(2, 3))
            .pow(1.0 / self.p)
        )


# ── CBAM (Convolutional Block Attention Module) ──────────────────────

class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM.

    Uses both average-pooled and max-pooled features through a shared MLP
    to produce per-channel attention weights.
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg_pool = x.mean(dim=(2, 3))                    # [B, C]
        max_pool = x.amax(dim=(2, 3))                    # [B, C]
        attn = torch.sigmoid(
            self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        )
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module of CBAM.

    Concatenates channel-wise avg and max features, then applies a 7×7 conv
    to produce a spatial attention map. Particularly useful for wafer defect
    patterns where spatial location is highly informative.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)             # [B, 1, H, W]
        max_out = x.amax(dim=1, keepdim=True)             # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)    # [B, 2, H, W]
        attn = torch.sigmoid(self.conv(combined))          # [B, 1, H, W]
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018).

    Sequentially applies channel attention then spatial attention.
    Inserted after ResNet layer blocks to refine feature maps with
    both channel and spatial focus — beneficial for wafer defect patterns
    that have characteristic spatial distributions.
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ── Helper ───────────────────────────────────────────────────────────

def _get_resnet_constructor(arch: str):
    constructors = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if arch not in constructors:
        raise ValueError(f"Unsupported backbone: {arch}")
    return constructors[arch]


def _adapt_conv1(backbone: nn.Module, in_channels: int, pretrained: bool) -> None:
    """Replace conv1 for non-3-channel input."""
    if in_channels != 3:
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )


# ── ResNet50 + GeM ───────────────────────────────────────────────────

class WaferClassifierGeM(nn.Module):
    """ResNet50 with GeM pooling replacing global average pooling.

    GeM learns an optimal aggregation between average and max pooling,
    which helps capture sparse defect activations more effectively.
    """

    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 9,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout: float = 0.2,
        gem_p: float = 3.0,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = _get_resnet_constructor(arch)(weights=weights)
        _adapt_conv1(backbone, in_channels, pretrained)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone.avgpool = nn.Identity()

        self.backbone = backbone
        self.gem = GeMPooling(p=gem_p)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # backbone forward up to (but not including) avgpool/fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return self.gem(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone.layer4


# ── ResNet50 + CBAM ──────────────────────────────────────────────────

class WaferClassifierCBAM(nn.Module):
    """ResNet50 with CBAM attention after layer3 and layer4.

    CBAM is inserted after the two deepest residual stages where feature
    maps are most semantically rich. This placement:
    - layer3 CBAM: refines mid-level spatial patterns (defect shape)
    - layer4 CBAM: refines high-level semantic features (defect type)
    """

    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 9,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout: float = 0.2,
        cbam_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = _get_resnet_constructor(arch)(weights=weights)
        _adapt_conv1(backbone, in_channels, pretrained)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # CBAM after layer3 (1024 channels for ResNet50) and layer4 (2048)
        layer3_channels = backbone.layer3[-1].bn3.num_features if hasattr(backbone.layer3[-1], 'bn3') else backbone.layer3[-1].bn2.num_features
        layer4_channels = feat_dim
        self.cbam3 = CBAM(layer3_channels, reduction=cbam_reduction)
        self.cbam4 = CBAM(layer4_channels, reduction=cbam_reduction)

        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.cbam3(x)
        x = self.backbone.layer4(x)
        x = self.cbam4(x)
        x = self.backbone.avgpool(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone.layer4


# ── ResNet50 + GeM + CBAM ────────────────────────────────────────────

class WaferClassifierGeMCBAM(nn.Module):
    """ResNet50 with both GeM pooling and CBAM attention.

    Combines spatial attention refinement (CBAM) with learnable
    aggregation (GeM) for maximum feature quality.
    """

    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 9,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout: float = 0.2,
        gem_p: float = 3.0,
        cbam_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = _get_resnet_constructor(arch)(weights=weights)
        _adapt_conv1(backbone, in_channels, pretrained)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone.avgpool = nn.Identity()

        self.backbone = backbone

        layer3_channels = backbone.layer3[-1].bn3.num_features if hasattr(backbone.layer3[-1], 'bn3') else backbone.layer3[-1].bn2.num_features
        layer4_channels = feat_dim
        self.cbam3 = CBAM(layer3_channels, reduction=cbam_reduction)
        self.cbam4 = CBAM(layer4_channels, reduction=cbam_reduction)
        self.gem = GeMPooling(p=gem_p)

        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.cbam3(x)
        x = self.backbone.layer4(x)
        x = self.cbam4(x)
        return self.gem(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone.layer4


# ── Registry integration ─────────────────────────────────────────────

from waferlab.registry import MODEL_REGISTRY  # noqa: E402


def _gem_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 9)),
        "in_channels": int(config.get("in_channels", 1)),
        "pretrained": bool(config.get("pretrained", False)),
        "dropout": float(config.get("dropout", 0.2)),
        "gem_p": float(config.get("gem_p", 3.0)),
    }


def _cbam_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 9)),
        "in_channels": int(config.get("in_channels", 1)),
        "pretrained": bool(config.get("pretrained", False)),
        "dropout": float(config.get("dropout", 0.2)),
        "cbam_reduction": int(config.get("cbam_reduction", 16)),
    }


def _gem_cbam_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 9)),
        "in_channels": int(config.get("in_channels", 1)),
        "pretrained": bool(config.get("pretrained", False)),
        "dropout": float(config.get("dropout", 0.2)),
        "gem_p": float(config.get("gem_p", 3.0)),
        "cbam_reduction": int(config.get("cbam_reduction", 16)),
    }


@MODEL_REGISTRY.register("resnet50_gem")
def _build_resnet50_gem(config: dict[str, Any]) -> WaferClassifierGeM:
    return WaferClassifierGeM(arch="resnet50", **_gem_kwargs(config))


@MODEL_REGISTRY.register("resnet50_cbam")
def _build_resnet50_cbam(config: dict[str, Any]) -> WaferClassifierCBAM:
    return WaferClassifierCBAM(arch="resnet50", **_cbam_kwargs(config))


@MODEL_REGISTRY.register("resnet50_gem_cbam")
def _build_resnet50_gem_cbam(config: dict[str, Any]) -> WaferClassifierGeMCBAM:
    return WaferClassifierGeMCBAM(arch="resnet50", **_gem_cbam_kwargs(config))
