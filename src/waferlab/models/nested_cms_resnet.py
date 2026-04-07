"""Nested CMS ResNet Classifier – Full CNN backbone + vector-level Nested Learning.

A de-tokenized architecture that replaces the patch-embed → token-sequence
pipeline with a direct CNN → global-pool → CMS/SelfModifier pipeline:

- **Full ResNet backbone** (ResNet18 or ResNet34) extracts spatial features
  and pools them to a single feature vector per image.
- **CMS + SelfModifier** operate on the pooled vector (shape ``[B, D]``),
  treating each sample's feature vector as a single "token".
- **Classification head** produces 9-class logits.

Why this is better for wafer defect maps:
1. Defect patterns are **global spatial patterns** (rings, scratches, edge
   clusters) that are already well-captured by CNN global pooling. Token
   sequences add unnecessary overhead without benefit for this domain.
2. By eliminating 196-token sequences, compute is reduced ~100× in the
   nested blocks while retaining the CMS multi-frequency memory and
   SelfModifier continual-learning mechanisms.
3. The CMS still provides fast/slow memory levels for anti-forgetting,
   and the SelfModifier still predicts update directions from error signals.

Nested Learning preservation:
- CMS multi-frequency memory hierarchy is fully intact.
- SelfModifier error-conditioned updates work identically on vectors.
- LevelClock scheduling, surprise gating, and teach-signal propagation
  are all preserved through ``forward_with_teach()``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from torchvision import models

from waferlab.engine.nested_learning import (
    CMS,
    LevelSpec,
    NestedBlock,
    NestedBlockConfig,
    SelfModifier,
)
from waferlab.registry import MODEL_REGISTRY


class NestedCMSResNetClassifier(nn.Module):
    """Full ResNet backbone + vector-level CMS/SelfModifier classifier.

    Architecture::

        Input [B, 1, 224, 224]
          → Full ResNet backbone (conv1..layer4)   → [B, 512, 7, 7]
          → Global Average Pool                     → [B, 512]
          → Projection (optional)                   → [B, D]
          → N × NestedBlock (CMS + SelfMod)         → [B, D]  (as [B, 1, D])
          → LayerNorm → Dropout → FC                → [B, num_classes]

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    in_channels : int
        Input image channels (1 for wafer maps).
    backbone : str
        ResNet variant: ``"resnet18"`` or ``"resnet34"``.
    feat_dim : int
        Dimension after projection (and CMS/SelfMod operating dim).
    num_nested_blocks : int
        Number of stacked NestedBlocks.
    cms_levels : list of dict or None
        CMS level specs. Default: fast (period=1) + slow (period=4).
    cms_hidden_multiplier : int
        MLP expansion ratio in CMS blocks.
    self_mod_hidden : int
        SelfModifier MLP expansion ratio.
    surprise_threshold : float or None
        Surprise gating threshold.
    dropout : float
        Dropout before classification head.
    pretrained : bool
        Use ImageNet-pretrained backbone.
    """

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 1,
        backbone: str = "resnet18",
        feat_dim: int = 256,
        num_nested_blocks: int = 2,
        cms_levels: Sequence[dict] | None = None,
        cms_hidden_multiplier: int = 4,
        self_mod_hidden: int = 4,
        surprise_threshold: float | None = None,
        dropout: float = 0.1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ── Full ResNet Backbone ──────────────────────────────────
        constructors = {"resnet18": models.resnet18, "resnet34": models.resnet34}
        if backbone not in constructors:
            raise ValueError(f"Unsupported backbone: {backbone}")
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = constructors[backbone](weights=weights)

        # Adapt conv1 for single-channel input.
        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,  # type: ignore[arg-type]
                stride=old_conv.stride,  # type: ignore[arg-type]
                padding=old_conv.padding,  # type: ignore[arg-type]
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    resnet.conv1.weight.copy_(
                        old_conv.weight.mean(dim=1, keepdim=True)
                    )

        # Extract backbone: conv1..layer4 + avgpool.
        backbone_dim = 512 if backbone in ("resnet18", "resnet34") else 2048
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Projection to CMS dimension ──────────────────────────
        if feat_dim != backbone_dim:
            self.proj = nn.Sequential(
                nn.Linear(backbone_dim, feat_dim),
                nn.GELU(),
            )
        else:
            self.proj = nn.Identity()

        # ── Nested Blocks (operate on [B, 1, D] pseudo-tokens) ──
        if cms_levels is None:
            level_specs = [
                LevelSpec(name="cms_fast", update_period=1),
                LevelSpec(name="cms_slow", update_period=4),
            ]
        else:
            level_specs = [
                LevelSpec(
                    name=str(lv.get("name", f"cms_{i}")),
                    update_period=int(lv.get("update_period", 1)),
                    warmup_steps=int(lv.get("warmup_steps", 0)),
                )
                for i, lv in enumerate(cms_levels)
            ]

        block_cfg = NestedBlockConfig(
            dim=feat_dim,
            cms_levels=level_specs,
            cms_hidden_multiplier=cms_hidden_multiplier,
            cms_use_layernorm=True,
            activation="gelu",
            self_mod_hidden=self_mod_hidden,
            surprise_threshold=surprise_threshold,
        )
        self.nested_blocks = nn.ModuleList(
            [NestedBlock(block_cfg) for _ in range(num_nested_blocks)]
        )

        # ── Classification Head ────────────────────────────────────
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.fc:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classification head.

        Returns
        -------
        Tensor [B, feat_dim]
        """
        feat = self.backbone(x)                       # [B, C, H, W]
        feat = self.global_pool(feat).flatten(1)      # [B, C]
        feat = self.proj(feat)                        # [B, D]

        # NestedBlocks expect [B, T, D] – use T=1 for vector features.
        tokens = feat.unsqueeze(1)                    # [B, 1, D]
        for block in self.nested_blocks:
            tokens = block(tokens)
        pooled = tokens.squeeze(1)                    # [B, D]

        pooled = self.norm(pooled)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits ``[B, num_classes]``."""
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def forward_with_teach(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        """Forward with teach-signal propagation to nested blocks.

        Parameters
        ----------
        x : Tensor [B, C, H, W]
        teach_signal : Tensor [B, 1, D] or [B, D]
            If 2D, will be unsqueezed to [B, 1, D].
        surprise_value : float or None
        """
        feat = self.backbone(x)
        feat = self.global_pool(feat).flatten(1)
        feat = self.proj(feat)
        tokens = feat.unsqueeze(1)  # [B, 1, D]

        # Ensure teach_signal is [B, 1, D].
        if teach_signal is not None and teach_signal.dim() == 2:
            teach_signal = teach_signal.unsqueeze(1)

        for block in self.nested_blocks:
            tokens = block(
                tokens,
                teach_signal=teach_signal,
                surprise_value=surprise_value,
            )

        pooled = tokens.squeeze(1)
        pooled = self.norm(pooled)
        return self.fc(self.drop(pooled))

    # ------------------------------------------------------------------
    # Helpers (compatible with existing codebase)
    # ------------------------------------------------------------------

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone[-1]  # layer4

    def get_token_dim(self) -> int:
        return self.fc.in_features

    def get_num_tokens(self, input_size: int = 224) -> int:
        return 1  # single vector, no token sequence


# ── Registry integration ─────────────────────────────────────────────


def _cms_resnet_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 9)),
        "in_channels": int(config.get("in_channels", 1)),
        "backbone": str(config.get("backbone", "resnet18")),
        "feat_dim": int(config.get("feat_dim", 256)),
        "num_nested_blocks": int(config.get("num_nested_blocks", 2)),
        "cms_levels": config.get("cms_levels"),
        "cms_hidden_multiplier": int(config.get("cms_hidden_multiplier", 4)),
        "self_mod_hidden": int(config.get("self_mod_hidden", 4)),
        "surprise_threshold": config.get("surprise_threshold"),
        "dropout": float(config.get("dropout", 0.1)),
        "pretrained": bool(config.get("pretrained", False)),
    }


@MODEL_REGISTRY.register("nested_cms_resnet")
def _build_nested_cms_resnet(config: dict[str, Any]) -> NestedCMSResNetClassifier:
    return NestedCMSResNetClassifier(**_cms_resnet_kwargs(config))
