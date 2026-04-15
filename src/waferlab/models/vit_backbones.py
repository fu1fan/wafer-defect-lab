"""ViT-family backbones for WM-811K wafer-map classification.

Provides three registered model variants built on ``timm`` pretrained
backbones, covering pure ViT and hybrid architectures:

- ``deit3_small_wafer``: DeiT III-S/16 (supervised IN22K→IN1K, 21.7M)
- ``eva02_small_wafer``: EVA-02-S/14 (MIM IN22K, 21.6M)
- ``caformer_s18_wafer``: CAFormer-S18 (MetaFormer conv+attn, 23.2M)

All variants maintain the same external API as :class:`WaferClassifier`
(forward, forward_features, get_cam_target_layer) and are compatible with
the existing training/evaluation pipeline.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

from waferlab.registry import MODEL_REGISTRY


class TimmViTWrapper(nn.Module):
    """Generic wrapper for timm ViT-family models adapted for wafer maps.

    Handles:
    - Single-channel input adaptation (averages RGB patch_embed weights)
    - Custom classification head with dropout
    - Consistent API (forward, forward_features, get_cam_target_layer)

    Parameters
    ----------
    timm_name : str
        Model name in ``timm.list_models()``.
    num_classes : int
        Number of output classes.
    in_channels : int
        Input channels (1 for wafer maps).
    pretrained : bool
        Whether to load pretrained weights.
    dropout : float
        Dropout rate before classification head.
    drop_path_rate : float
        Stochastic depth rate (ViT-specific regularization).
    img_size : int
        Input image size (needed for models with fixed pos-embed).
    """

    def __init__(
        self,
        timm_name: str,
        num_classes: int = 9,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
        drop_path_rate: float = 0.0,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        create_kwargs: dict[str, Any] = {
            "num_classes": 0,  # remove original head
            "pretrained": pretrained,
            "drop_path_rate": drop_path_rate,
        }
        # Only pass img_size for ViT models that accept it
        if "caformer" not in timm_name and "convformer" not in timm_name:
            create_kwargs["img_size"] = img_size

        self.backbone = timm.create_model(timm_name, **create_kwargs)
        self.feat_dim = self.backbone.num_features

        # Adapt first layer for single-channel input
        if in_channels != 3:
            self._adapt_input_channels()

        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def _adapt_input_channels(self) -> None:
        """Replace patch_embed / stem conv for single-channel input."""
        # timm models expose patch_embed.proj for ViTs,
        # or stem/conv for hybrid/MetaFormer models
        conv = self._find_first_conv()
        if conv is None:
            return

        new_conv = nn.Conv2d(
            self.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)

        self._replace_first_conv(new_conv)

    def _find_first_conv(self) -> nn.Conv2d | None:
        """Locate the first Conv2d in the backbone."""
        # ViT / DeiT / EVA: patch_embed.proj
        if hasattr(self.backbone, "patch_embed"):
            pe = self.backbone.patch_embed
            if hasattr(pe, "proj") and isinstance(pe.proj, nn.Conv2d):
                return pe.proj
            # Some models use backbone.proj directly
            if hasattr(pe, "backbone"):
                for m in pe.backbone.modules():
                    if isinstance(m, nn.Conv2d):
                        return m
        # MetaFormer / CAFormer: stem or downsample_layers[0]
        if hasattr(self.backbone, "stem"):
            for m in self.backbone.stem.modules():
                if isinstance(m, nn.Conv2d):
                    return m
        if hasattr(self.backbone, "downsample_layers"):
            for m in self.backbone.downsample_layers[0].modules():
                if isinstance(m, nn.Conv2d):
                    return m
        # Fallback: first Conv2d anywhere
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                return m
        return None

    def _replace_first_conv(self, new_conv: nn.Conv2d) -> None:
        """Replace the first Conv2d with the adapted version."""
        if hasattr(self.backbone, "patch_embed"):
            pe = self.backbone.patch_embed
            if hasattr(pe, "proj") and isinstance(pe.proj, nn.Conv2d):
                pe.proj = new_conv
                return
        if hasattr(self.backbone, "stem"):
            for name, m in self.backbone.stem.named_modules():
                if isinstance(m, nn.Conv2d):
                    parts = name.split(".")
                    parent = self.backbone.stem
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], new_conv)
                    return
        if hasattr(self.backbone, "downsample_layers"):
            layer0 = self.backbone.downsample_layers[0]
            for name, m in layer0.named_modules():
                if isinstance(m, nn.Conv2d):
                    parts = name.split(".")
                    parent = layer0
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], new_conv)
                    return

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return globally-pooled feature vector before FC head."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [B, num_classes]."""
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def get_cam_target_layer(self) -> nn.Module:
        """Return last feature block for GradCAM."""
        # ViT models: last transformer block
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        # MetaFormer: last stage
        if hasattr(self.backbone, "stages"):
            return self.backbone.stages[-1]
        # Fallback
        children = list(self.backbone.children())
        return children[-2] if len(children) > 1 else children[-1]


# ── Registry integration ─────────────────────────────────────────────


def _vit_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract standard kwargs from config dict."""
    return {
        "num_classes": int(config.get("num_classes", 9)),
        "in_channels": int(config.get("in_channels", 1)),
        "pretrained": bool(config.get("pretrained", True)),
        "dropout": float(config.get("dropout", 0.2)),
        "drop_path_rate": float(config.get("drop_path_rate", 0.0)),
        "img_size": int(config.get("img_size", 224)),
    }


@MODEL_REGISTRY.register("deit3_small_wafer")
def _build_deit3_small(config: dict[str, Any]) -> TimmViTWrapper:
    return TimmViTWrapper(
        timm_name="deit3_small_patch16_224.fb_in22k_ft_in1k",
        **_vit_kwargs(config),
    )


@MODEL_REGISTRY.register("eva02_small_wafer")
def _build_eva02_small(config: dict[str, Any]) -> TimmViTWrapper:
    return TimmViTWrapper(
        timm_name="eva02_small_patch14_224.mim_in22k",
        **_vit_kwargs(config),
    )


@MODEL_REGISTRY.register("caformer_s18_wafer")
def _build_caformer_s18(config: dict[str, Any]) -> TimmViTWrapper:
    return TimmViTWrapper(
        timm_name="caformer_s18.sail_in1k",
        **_vit_kwargs(config),
    )
