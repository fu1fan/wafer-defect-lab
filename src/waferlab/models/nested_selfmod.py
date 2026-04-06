"""Nested Self-Mod Classifier – CNN stem + Nested Learning HOPE blocks.

A hybrid model that combines:
- **CNN stem** (ResNet18 stages 1-2) for spatial feature extraction
- **Feature-map tokenization** (flatten spatial dims → token sequence)
- **NestedBlock stack** (CMS + SelfModifier + surprise gating) for
  multi-frequency memory processing
- **Classification head** on globally pooled token features

This is the main ``research_nest`` model, directly incorporating the
core mechanisms from the Nested Learning / HOPE framework (CMS,
SelfModifier, surprise gating, LevelSpec multi-frequency scheduling).
"""

from __future__ import annotations

from dataclasses import dataclass
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


class NestedSelfModClassifier(nn.Module):
    """CNN stem + Nested Learning HOPE blocks classifier.

    Architecture::

        Input [B, 1, 224, 224]
          → CNN stem (ResNet18 conv1..layer2)   → [B, 128, 28, 28]
          → PatchEmbed (strided conv)           → [B, D, H', W']
          → Flatten to tokens                   → [B, T, D]
          → N × NestedBlock (CMS + SelfMod)    → [B, T, D]
          → Global average pool                 → [B, D]
          → LayerNorm → Dropout → FC            → [B, num_classes]

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    in_channels : int
        Input image channels (1 for wafer maps).
    token_dim : int
        Dimension of tokens fed to NestedBlocks.
    num_nested_blocks : int
        Number of stacked NestedBlocks.
    patch_stride : int
        Stride for the patch embedding conv (controls token count).
    cms_levels : list of dict
        CMS level specifications, each with ``name`` and ``update_period``.
    cms_hidden_multiplier : int
        MLP expansion ratio in CMS blocks.
    self_mod_hidden : int
        SelfModifier MLP expansion ratio.
    surprise_threshold : float or None
        Surprise gating threshold (None = no gating).
    dropout : float
        Dropout before classification head.
    pretrained_stem : bool
        Use ImageNet-pretrained ResNet18 stem.
    """

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 1,
        token_dim: int = 192,
        num_nested_blocks: int = 3,
        patch_stride: int = 2,
        cms_levels: Sequence[dict] | None = None,
        cms_hidden_multiplier: int = 4,
        self_mod_hidden: int = 4,
        surprise_threshold: float | None = None,
        dropout: float = 0.1,
        pretrained_stem: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ── CNN Stem (ResNet18 stages 1-2) ────────────────────────
        weights = "IMAGENET1K_V1" if pretrained_stem else None
        backbone = models.resnet18(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,  # type: ignore[arg-type]
                stride=old_conv.stride,  # type: ignore[arg-type]
                padding=old_conv.padding,  # type: ignore[arg-type]
                bias=False,
            )
            if pretrained_stem:
                with torch.no_grad():
                    backbone.conv1.weight.copy_(
                        old_conv.weight.mean(dim=1, keepdim=True)
                    )

        # Keep conv1 → bn1 → relu → maxpool → layer1 → layer2
        # Output: [B, 128, 28, 28] for 224x224 input
        self.stem = nn.Sequential(
            backbone.conv1,   # -> [B, 64, 112, 112]
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # -> [B, 64, 56, 56]
            backbone.layer1,   # -> [B, 64, 56, 56]
            backbone.layer2,   # -> [B, 128, 28, 28]
        )
        stem_dim = 128  # ResNet18 layer2 output channels

        # ── Patch Embedding (feature map → tokens) ────────────────
        self.patch_embed = nn.Conv2d(
            stem_dim, token_dim,
            kernel_size=patch_stride, stride=patch_stride,
        )
        # For 28x28 input with stride 2: → [B, token_dim, 14, 14] → 196 tokens

        # ── Nested Blocks ──────────────────────────────────────────
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
            dim=token_dim,
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
        self.norm = nn.LayerNorm(token_dim)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(token_dim, num_classes)

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
        Tensor [B, token_dim]
            Globally pooled token features.
        """
        # CNN stem
        feat = self.stem(x)                     # [B, 128, 28, 28]

        # Patch embed → tokens
        feat = self.patch_embed(feat)            # [B, D, H', W']
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2) # [B, T, D]

        # Nested blocks (no teach_signal during standard forward)
        for block in self.nested_blocks:
            tokens = block(tokens)

        # Global average pool
        pooled = tokens.mean(dim=1)             # [B, D]
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
        teach_signal : Tensor [B, T, D] or None
            If provided, triggers inner CMS updates in each block.
        surprise_value : float or None
        """
        feat = self.stem(x)
        feat = self.patch_embed(feat)
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)

        for block in self.nested_blocks:
            tokens = block(
                tokens,
                teach_signal=teach_signal,
                surprise_value=surprise_value,
            )

        pooled = tokens.mean(dim=1)
        pooled = self.norm(pooled)
        return self.fc(self.drop(pooled))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_cam_target_layer(self) -> nn.Module:
        return self.stem[-1]  # layer2

    def get_token_dim(self) -> int:
        return self.fc.in_features

    def get_num_tokens(self, input_size: int = 224) -> int:
        """Compute token count for a given input spatial size."""
        # stem: 224 → 28 (ResNet18 conv1..layer2)
        feat_size = input_size // 8
        # patch_embed stride
        ps = self.patch_embed.stride[0]  # type: ignore[index]
        tok_size = feat_size // ps
        return tok_size * tok_size


# ── Registry integration ─────────────────────────────────────────────


def _nested_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 9)),
        "in_channels": int(config.get("in_channels", 1)),
        "token_dim": int(config.get("token_dim", 192)),
        "num_nested_blocks": int(config.get("num_nested_blocks", 3)),
        "patch_stride": int(config.get("patch_stride", 2)),
        "cms_levels": config.get("cms_levels"),
        "cms_hidden_multiplier": int(config.get("cms_hidden_multiplier", 4)),
        "self_mod_hidden": int(config.get("self_mod_hidden", 4)),
        "surprise_threshold": config.get("surprise_threshold"),
        "dropout": float(config.get("dropout", 0.1)),
        "pretrained_stem": bool(config.get("pretrained_stem", False)),
    }


@MODEL_REGISTRY.register("nested_selfmod")
def _build_nested_selfmod(config: dict[str, Any]) -> NestedSelfModClassifier:
    return NestedSelfModClassifier(**_nested_kwargs(config))
