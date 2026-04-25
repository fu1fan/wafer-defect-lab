"""CAFormer + HOPE SelfMod/CMS classifiers for continual learning."""

from __future__ import annotations

from typing import Any, Sequence

import timm
import torch
import torch.nn as nn

from waferlab.engine.nested_learning import (
    HOPESelfModBlock,
    HOPESelfModBlockConfig,
    LevelSpec,
)
from waferlab.registry import MODEL_REGISTRY


class CAFormerHOPEClassifier(nn.Module):
    """CAFormer-S18 backbone with shared HOPE blocks over tokens or pooled vector."""

    def __init__(
        self,
        *,
        num_classes: int = 9,
        in_channels: int = 1,
        base_arch: str = "caformer_s18_wafer",
        timm_name: str = "caformer_s18.sail_in1k",
        token_mode: str = "spatial",
        pretrained: bool = True,
        token_dim: int = 256,
        token_stage: int = 3,
        num_hope_blocks: int = 2,
        cms_levels: Sequence[dict] | None = None,
        cms_hidden_multiplier: int = 4,
        cms_use_layernorm: bool = True,
        self_mod_lr: float = 1e-3,
        selfmod_local_conv_window: int | None = 4,
        selfmod_adaptive_q: bool = False,
        selfmod_use_skip: bool = True,
        cms_chunk_reduction: str = "sum",
        cms_flush_partial_at_end: bool = False,
        surprise_threshold: float | None = None,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if base_arch != "caformer_s18_wafer":
            raise ValueError(f"Unsupported base_arch={base_arch!r}; expected caformer_s18_wafer")
        token_mode = str(token_mode).lower()
        if token_mode not in {"spatial", "pooled"}:
            raise ValueError("token_mode must be 'spatial' or 'pooled'")
        if token_stage < 0 or token_stage > 3:
            raise ValueError("token_stage must be in [0, 3] for CAFormer-S18")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.token_mode = token_mode
        self.token_stage = token_stage
        self.token_dim = token_dim

        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.backbone_dim = int(self.backbone.num_features)
        self.stage_dims = [64, 128, 320, 512]
        self.token_source_dim = self.backbone_dim if token_mode == "pooled" else self.stage_dims[token_stage]
        self._adapt_input_channels()

        self.proj = nn.Linear(self.token_source_dim, token_dim)

        if cms_levels is None:
            level_specs = [
                LevelSpec(name="cms_fast", update_period=1),
                LevelSpec(name="cms_slow", update_period=4),
            ]
        else:
            level_specs = [
                LevelSpec(
                    name=str(level.get("name", f"cms_{i}")),
                    update_period=int(level.get("update_period", 1)),
                    warmup_steps=int(level.get("warmup_steps", 0)),
                )
                for i, level in enumerate(cms_levels)
            ]

        block_cfg = HOPESelfModBlockConfig(
            dim=token_dim,
            cms_levels=level_specs,
            cms_hidden_multiplier=cms_hidden_multiplier,
            cms_use_layernorm=cms_use_layernorm,
            self_mod_lr=self_mod_lr,
            selfmod_local_conv_window=selfmod_local_conv_window,
            selfmod_adaptive_q=selfmod_adaptive_q,
            selfmod_use_skip=selfmod_use_skip,
            cms_chunk_reduction=cms_chunk_reduction,
            cms_flush_partial_at_end=cms_flush_partial_at_end,
            surprise_threshold=surprise_threshold,
        )
        self.hope_blocks = nn.ModuleList(
            HOPESelfModBlock(block_cfg) for _ in range(num_hope_blocks)
        )
        # Alias kept for existing continual helpers (freeze_slow_cms, etc.).
        self.nested_blocks = self.hope_blocks

        self.norm = nn.LayerNorm(token_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(token_dim, num_classes)
        self._init_head()

    def _init_head(self) -> None:
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def _find_first_conv(self) -> nn.Conv2d | None:
        if hasattr(self.backbone, "stem"):
            for module in self.backbone.stem.modules():
                if isinstance(module, nn.Conv2d):
                    return module
        if hasattr(self.backbone, "downsample_layers"):
            for module in self.backbone.downsample_layers[0].modules():
                if isinstance(module, nn.Conv2d):
                    return module
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                return module
        return None

    def _replace_first_conv(self, new_conv: nn.Conv2d) -> None:
        if hasattr(self.backbone, "stem"):
            for name, module in self.backbone.stem.named_modules():
                if isinstance(module, nn.Conv2d):
                    parent = self.backbone.stem
                    parts = name.split(".")
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], new_conv)
                    return
        raise RuntimeError("Could not replace CAFormer first convolution")

    def _adapt_input_channels(self) -> None:
        if self.in_channels == 3:
            return
        conv = self._find_first_conv()
        if conv is None:
            return
        new_conv = nn.Conv2d(
            self.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        self._replace_first_conv(new_conv)

    def _forward_stage_map(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.stem(x)
        for i, stage in enumerate(self.backbone.stages):
            feat = stage(feat)
            if i == self.token_stage:
                return feat
        return feat

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-HOPE tokens [B, T, D]."""
        if self.token_mode == "pooled":
            feat = self.backbone(x)
            tokens = feat.unsqueeze(1)
        else:
            feat_map = self._forward_stage_map(x)
            tokens = feat_map.flatten(2).transpose(1, 2)
        return self.proj(tokens)

    def forward_from_tokens(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        current = tokens
        if teach_signal is not None and teach_signal.dim() == 2:
            teach_signal = teach_signal.unsqueeze(1)
        for block in self.hope_blocks:
            current = block(
                current,
                teach_signal=teach_signal,
                surprise_value=surprise_value,
            )
        pooled = current.mean(dim=1)
        return self.norm(pooled)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        return self.forward_from_tokens(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    def forward_with_teach(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        feat = self.forward_from_tokens(
            tokens,
            teach_signal=teach_signal,
            surprise_value=surprise_value,
        )
        return self.fc(self.drop(feat))

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone.stages[self.token_stage]

    def get_token_dim(self) -> int:
        return self.token_dim

    def get_num_tokens(self, input_size: int = 224) -> int:
        if self.token_mode == "pooled":
            return 1
        # CAFormer-S18 spatial sizes: stem/stage0 56, stage1 28, stage2 14, stage3 7.
        stride_by_stage = [4, 8, 16, 32]
        side = input_size // stride_by_stage[self.token_stage]
        return side * side


def _caformer_hope_kwargs(config: dict[str, Any], *, token_mode_default: str) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 9)),
        "in_channels": int(config.get("in_channels", 1)),
        "base_arch": str(config.get("base_arch", "caformer_s18_wafer")),
        "token_mode": str(config.get("token_mode", token_mode_default)),
        "pretrained": bool(config.get("pretrained", True)),
        "token_dim": int(config.get("token_dim", 256)),
        "token_stage": int(config.get("token_stage", 3)),
        "num_hope_blocks": int(config.get("num_hope_blocks", 2)),
        "cms_levels": config.get("cms_levels"),
        "cms_hidden_multiplier": int(config.get("cms_hidden_multiplier", 4)),
        "cms_use_layernorm": bool(config.get("cms_use_layernorm", True)),
        "self_mod_lr": float(config.get("self_mod_lr", 1e-3)),
        "selfmod_local_conv_window": config.get("selfmod_local_conv_window", 4),
        "selfmod_adaptive_q": bool(config.get("selfmod_adaptive_q", False)),
        "selfmod_use_skip": bool(config.get("selfmod_use_skip", True)),
        "cms_chunk_reduction": str(config.get("cms_chunk_reduction", "sum")),
        "cms_flush_partial_at_end": bool(config.get("cms_flush_partial_at_end", False)),
        "surprise_threshold": config.get("surprise_threshold"),
        "dropout": float(config.get("dropout", 0.1)),
        "drop_path_rate": float(config.get("drop_path_rate", 0.0)),
    }


@MODEL_REGISTRY.register("caformer_hope_token")
def _build_caformer_hope_token(config: dict[str, Any]) -> CAFormerHOPEClassifier:
    return CAFormerHOPEClassifier(
        **_caformer_hope_kwargs(config, token_mode_default="spatial")
    )


@MODEL_REGISTRY.register("caformer_hope_pooled")
def _build_caformer_hope_pooled(config: dict[str, Any]) -> CAFormerHOPEClassifier:
    return CAFormerHOPEClassifier(
        **_caformer_hope_kwargs(config, token_mode_default="pooled")
    )
