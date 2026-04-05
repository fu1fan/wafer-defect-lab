"""ResNet18 variant optimized for abnormal-class recall in wafer classification.

Improvements over baseline :class:`WaferClassifier`:

- **GeM (Generalized Mean) pooling** replaces standard global average pooling,
  enabling the model to learn a sharper spatial aggregation that emphasizes
  higher-activation regions – beneficial for sparse anomaly patterns.
- **SE (Squeeze-and-Excitation) channel attention** between pooling and the
  classifier head, helping the model selectively amplify informative channels.
- **Two-layer classification head** with BatchNorm and GELU activation for
  better decision-boundary shaping in the severely imbalanced binary setting.

The backbone is still a standard ``torchvision.models.resnet18``, so this
remains firmly in the ResNet18 family.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── Utility layers ───────────────────────────────────────────────────


class GeMPooling(nn.Module):
    """Generalized Mean Pooling.

    When ``p == 1`` this reduces to average pooling; as ``p → ∞`` it
    approaches max pooling.  The learnable *p* lets the network decide
    the right trade-off.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, C]
        return (
            x.clamp(min=self.eps)
            .pow(self.p)
            .mean(dim=(2, 3))
            .pow(1.0 / self.p)
        )


class SEAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (1-D version for pooled features)."""

    def __init__(self, in_features: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(in_features // reduction, 1)
        self.fc1 = nn.Linear(in_features, mid)
        self.fc2 = nn.Linear(mid, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.relu(self.fc1(x))
        s = torch.sigmoid(self.fc2(s))
        return x * s


# ── Model ────────────────────────────────────────────────────────────


def _get_resnet_constructor(arch: str):
    constructors = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if arch not in constructors:
        raise ValueError(f"Unsupported backbone: {arch}")
    return constructors[arch]


class WaferClassifierRecallOpt(nn.Module):
    """ResNet-based wafer classifier with recall-oriented head improvements.

    Parameters
    ----------
    arch : str
        Backbone architecture (``"resnet18"`` / ``"resnet34"`` / ``"resnet50"``).
    num_classes : int
        2 for binary, 9 for multi-class.
    in_channels : int
        1 = single-channel wafer map; 3 = channel-replicated.
    pretrained : bool
        Load ImageNet-pretrained weights.
    dropout : float
        Dropout probability in the classification head.
    gem_p : float
        Initial power parameter for GeM pooling.
    se_reduction : int
        SE attention reduction ratio.
    head_hidden : int
        Hidden-layer width of the two-layer classification head.
        Set to 0 to use a single linear layer (similar to baseline).
    """

    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout: float = 0.2,
        gem_p: float = 3.0,
        se_reduction: int = 4,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = _get_resnet_constructor(arch)(weights=weights)

        # Adapt conv1 for non-3-channel input.
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
        backbone.avgpool = nn.Identity()  # type: ignore[assignment]

        # Store backbone layers individually so we can stop before avgpool.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.gem = GeMPooling(p=gem_p)
        self.se = SEAttention(feat_dim, reduction=se_reduction)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, head_hidden),
                nn.BatchNorm1d(head_hidden),
                nn.GELU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Linear(feat_dim, num_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from backbone (before pooling)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, C, H, W]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Global-pooled feature vector before the classification head."""
        feat_map = self._forward_backbone(x)  # [B, C, H, W]
        feat = self.gem(feat_map)              # [B, C]
        feat = self.se(feat)                   # [B, C]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits ``[B, num_classes]``."""
        feat = self.forward_features(x)
        return self.head(self.drop(feat))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_cam_target_layer(self) -> nn.Module:
        """Return the last conv block for GradCAM hooks."""
        return self.layer4


# ── Registry integration ─────────────────────────────────────────────

from waferlab.registry import MODEL_REGISTRY  # noqa: E402


def _recall_opt_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 2)),
        "in_channels": int(config.get("in_channels", 1)),
        "pretrained": bool(config.get("pretrained", False)),
        "dropout": float(config.get("dropout", 0.2)),
        "gem_p": float(config.get("gem_p", 3.0)),
        "se_reduction": int(config.get("se_reduction", 4)),
        "head_hidden": int(config.get("head_hidden", 256)),
    }


@MODEL_REGISTRY.register("resnet18_recall_opt")
def _build_resnet18_recall_opt(config: dict[str, Any]) -> WaferClassifierRecallOpt:
    return WaferClassifierRecallOpt(arch="resnet18", **_recall_opt_kwargs(config))
