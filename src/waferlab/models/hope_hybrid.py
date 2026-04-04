"""HOPE-Hybrid: Hierarchical backbone for 2D wafer-map classification.

A 4-stage hybrid model that mixes plain convolution (stages 1-2) with
HOPE blocks (stages 3-4).  Designed for single-channel 224x224 wafer
maps but works with any ``in_channels``.

HOPE block combines:
- Large-kernel depthwise convolution  (local spatial mixing)
- Pointwise expand-project MLP        (channel mixing)
- SE-style channel gating             (lightweight global context)
- Layer scale + DropPath               (training stability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility layers ───────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth (per-sample drop of the residual branch)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) >= self.drop_prob
        return x * keep / (1.0 - self.drop_prob)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel gating."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.gelu(self.fc1(s))
        s = self.fc2(s).sigmoid()
        return x * s


# ── Block configs ────────────────────────────────────────────────────

@dataclass
class HOPEBlockConfig:
    """Parameters governing a single HOPE block.

    Parameters
    ----------
    dim : int
        Number of input/output channels.
    kernel_size : int
        Depthwise convolution kernel size (odd, >= 3).
    expand_ratio : float
        Channel expansion factor in the pointwise MLP.
    se_reduction : int
        Squeeze-and-Excitation reduction ratio.
    layer_scale_init : float
        Initial value for the learnable layer-scale parameter.
        Set to 0 to disable.
    drop_path : float
        Drop-path probability for stochastic depth.
    """

    dim: int = 192
    kernel_size: int = 7
    expand_ratio: float = 4.0
    se_reduction: int = 4
    layer_scale_init: float = 1e-6
    drop_path: float = 0.0


# ── HOPE block ───────────────────────────────────────────────────────

class HOPEBlock(nn.Module):
    """Hybrid-Order Pattern Encoding block.

    Pre-norm residual block: LN -> DWConv -> PW-expand -> GELU ->
    PW-project -> SE -> LayerScale -> DropPath -> residual.
    """

    def __init__(self, cfg: HOPEBlockConfig) -> None:
        super().__init__()
        dim = cfg.dim
        hidden = int(dim * cfg.expand_ratio)
        padding = cfg.kernel_size // 2

        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(
            dim, dim, cfg.kernel_size, padding=padding, groups=dim, bias=True,
        )
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)
        self.se = SEBlock(dim, reduction=cfg.se_reduction)

        # Learnable per-channel scale (layer scale).
        if cfg.layer_scale_init > 0:
            self.gamma = nn.Parameter(
                cfg.layer_scale_init * torch.ones(dim), requires_grad=True,
            )
        else:
            self.gamma = None  # type: ignore[assignment]

        self.drop_path = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # Channels-last LayerNorm: (B,C,H,W) -> permute -> LN -> permute
        out = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.dwconv(out)
        out = self.pw1(out)
        out = self.act(out)
        out = self.pw2(out)
        out = self.se(out)
        if self.gamma is not None:
            out = out * self.gamma.view(1, -1, 1, 1)
        out = self.drop_path(out)
        return shortcut + out


# ── Plain conv block for early stages ────────────────────────────────

class ConvBlock(nn.Module):
    """BatchNorm-ReLU conv block for the first two stages."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(shortcut + out)


# ── Downsample layer ─────────────────────────────────────────────────

class Downsample(nn.Module):
    """Spatial halving + channel expansion via strided conv."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


# ── Full model ───────────────────────────────────────────────────────

class HOPEHybridClassifier(nn.Module):
    """4-stage hybrid backbone with a classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 = binary, 9 = WM-811K multiclass).
    in_channels : int
        Input image channels (1 for wafer maps).
    dims : tuple[int, ...]
        Channel dimensions for each of the 4 stages.
    depths : tuple[int, ...]
        Number of blocks in each stage.
    hope_kernel_size : int
        Depthwise kernel size used in HOPE blocks (stages 3-4).
    hope_expand_ratio : float
        Channel expansion ratio in HOPE blocks.
    se_reduction : int
        SE reduction ratio.
    layer_scale_init : float
        Initial value for layer-scale in HOPE blocks.
    drop_path_rate : float
        Maximum stochastic-depth rate (linearly interpolated across blocks).
    dropout : float
        Dropout before the final FC head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        dims: tuple[int, ...] = (48, 96, 192, 384),
        depths: tuple[int, ...] = (2, 2, 4, 2),
        hope_kernel_size: int = 7,
        hope_expand_ratio: float = 4.0,
        se_reduction: int = 4,
        layer_scale_init: float = 1e-6,
        drop_path_rate: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(dims) == 4 and len(depths) == 4
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Stem: patch-embed style, 4x4 stride 4 -> H/4 x W/4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0]),
        )

        # Per-block stochastic depth schedule (linear increase).
        total_blocks = sum(depths)
        dp_rates = [drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]
        block_idx = 0

        # Stage 1-2: plain ConvBlocks.
        self.stage1 = nn.Sequential(*[ConvBlock(dims[0]) for _ in range(depths[0])])
        block_idx += depths[0]

        self.down1 = Downsample(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[ConvBlock(dims[1]) for _ in range(depths[1])])
        block_idx += depths[1]

        # Stage 3-4: HOPE blocks.
        self.down2 = Downsample(dims[1], dims[2])
        self.stage3 = nn.Sequential(*[
            HOPEBlock(HOPEBlockConfig(
                dim=dims[2],
                kernel_size=hope_kernel_size,
                expand_ratio=hope_expand_ratio,
                se_reduction=se_reduction,
                layer_scale_init=layer_scale_init,
                drop_path=dp_rates[block_idx + i],
            )) for i in range(depths[2])
        ])
        block_idx += depths[2]

        self.down3 = Downsample(dims[2], dims[3])
        self.stage4 = nn.Sequential(*[
            HOPEBlock(HOPEBlockConfig(
                dim=dims[3],
                kernel_size=hope_kernel_size,
                expand_ratio=hope_expand_ratio,
                se_reduction=se_reduction,
                layer_scale_init=layer_scale_init,
                drop_path=dp_rates[block_idx + i],
            )) for i in range(depths[3])
        ])

        self.norm = nn.LayerNorm(dims[3])
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(dims[3], num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled feature vector before the FC head."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = x.mean(dim=(2, 3))  # global average pooling
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits ``[B, num_classes]``."""
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_cam_target_layer(self) -> nn.Module:
        """Return the last spatial stage for GradCAM hooks."""
        return self.stage4


# ── Registry integration ─────────────────────────────────────────────

from waferlab.registry import MODEL_REGISTRY  # noqa: E402


def _parse_int_tuple(v: Any, n: int, default: tuple) -> tuple[int, ...]:
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return (int(v),) * n


def _hope_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": int(config.get("num_classes", 2)),
        "in_channels": int(config.get("in_channels", 1)),
        "dims": _parse_int_tuple(config.get("dims"), 4, (48, 96, 192, 384)),
        "depths": _parse_int_tuple(config.get("depths"), 4, (2, 2, 4, 2)),
        "hope_kernel_size": int(config.get("hope_kernel_size", 7)),
        "hope_expand_ratio": float(config.get("hope_expand_ratio", 4.0)),
        "se_reduction": int(config.get("se_reduction", 4)),
        "layer_scale_init": float(config.get("layer_scale_init", 1e-6)),
        "drop_path_rate": float(config.get("drop_path_rate", 0.1)),
        "dropout": float(config.get("dropout", 0.0)),
    }


@MODEL_REGISTRY.register("hope_hybrid")
def _build_hope_hybrid(config: dict[str, Any]) -> HOPEHybridClassifier:
    return HOPEHybridClassifier(**_hope_kwargs(config))
