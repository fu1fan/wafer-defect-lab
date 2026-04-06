"""Continuum Memory System (CMS) – multi-frequency residual MLP blocks.

Localized from ``nested_learning.cms``.  Each CMS level is a residual
MLP: ``x + MLP(LN(x))``.  Multiple levels with different update
frequencies form the CMS hierarchy.

The anti-forgetting design:
- Slow levels (high ``update_period``) stabilize as *knowledge anchors*.
- Fast levels (low ``update_period``) adapt quickly to new data.
- Surprise gating (external) controls *when* updates fire.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .levels import LevelSpec


class CMSBlock(nn.Module):
    """Single CMS level: residual MLP with optional LayerNorm and grad clipping."""

    def __init__(
        self,
        dim: int,
        hidden_multiplier: int = 4,
        activation: str = "gelu",
        grad_clip: float = 1.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        hidden = dim * hidden_multiplier
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            act = nn.GELU()
        norm: nn.Module = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.net = nn.Sequential(
            norm,
            nn.Linear(dim, hidden),
            act,
            nn.Linear(hidden, dim),
        )
        self.grad_clip = grad_clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.net(x)
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                norm = delta.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / self.grad_clip, min=1.0)
            delta = delta / scale
        return x + delta


class CMS(nn.Module):
    """Continuum Memory System with multi-frequency level blocks.

    Parameters
    ----------
    dim : int
        Feature / token dimension.
    levels : Sequence[LevelSpec]
        Ordered level specifications (fast → slow).
    hidden_multiplier : int
        MLP expansion ratio inside each CMSBlock.
    activation : str
        Activation function name.
    use_layernorm : bool
        Whether to apply LayerNorm before the MLP in each block.
    """

    def __init__(
        self,
        *,
        dim: int,
        levels: Sequence[LevelSpec],
        hidden_multiplier: int = 4,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.level_specs: tuple[LevelSpec, ...] = tuple(levels)
        self.blocks = nn.ModuleDict(
            {
                spec.name: CMSBlock(
                    dim,
                    hidden_multiplier=hidden_multiplier,
                    activation=activation,
                    grad_clip=1.0,
                    use_layernorm=use_layernorm,
                )
                for spec in self.level_specs
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        current = x
        inputs: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, torch.Tensor] = {}
        for spec in self.level_specs:
            block = self.blocks[spec.name]
            inputs[spec.name] = current
            current = block(current)
            outputs[spec.name] = current
        if return_intermediates:
            return current, inputs, outputs
        return current
