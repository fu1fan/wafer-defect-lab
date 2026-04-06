"""NestedBlock – HOPE-inspired block for image token sequences.

Combines:
 1. CMS (multi-frequency residual MLP levels)
 2. SelfModifier (error-conditioned update direction)
 3. Surprise gating (teach-signal norm threshold)
 4. LevelClock (multi-frequency update scheduling)

Designed to operate on flattened feature-map tokens of shape ``[B, T, D]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cms import CMS
from .levels import LevelClock, LevelSpec
from .self_modifier import SelfModifier


@dataclass
class NestedBlockConfig:
    dim: int = 128
    cms_levels: Sequence[LevelSpec] = field(default_factory=lambda: [
        LevelSpec(name="cms_fast", update_period=1),
        LevelSpec(name="cms_slow", update_period=4),
    ])
    cms_hidden_multiplier: int = 4
    cms_use_layernorm: bool = True
    activation: str = "gelu"
    self_mod_hidden: int = 4
    surprise_threshold: float | None = None


class NestedBlock(nn.Module):
    """HOPE-style nested block: SelfModifier → CMS pipeline.

    During training, when a ``teach_signal`` is provided:
      1. The SelfModifier predicts an update direction from (input, CMS output, teach_signal).
      2. Each CMS level is updated via local gradient descent on a δ-shaped loss
         (MSE between CMS prediction and error-shifted target), gated by the
         surprise threshold and level clock.

    During inference (no teach_signal), the block is a simple forward:
      ``x → CMS(x)`` (residual MLP stack).
    """

    def __init__(self, config: NestedBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
            use_layernorm=config.cms_use_layernorm,
        )
        self.self_modifier = SelfModifier(
            config.dim, hidden_multiplier=config.self_mod_hidden,
        )
        self.level_clock = LevelClock(config.cms_levels)
        self.surprise_threshold = config.surprise_threshold

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor [B, T, D]
            Input token sequence.
        teach_signal : Tensor [B, T, D] or None
            Error signal (e.g. gradient proxy).  When provided, triggers
            inner memory updates.
        surprise_value : float or None
            Scalar surprise metric (e.g. per-batch loss).  Used for gating.

        Returns
        -------
        Tensor [B, T, D]
        """
        cms_out = self.cms(x)

        if teach_signal is not None and self.training:
            self._update_cms(x, cms_out, teach_signal, surprise_value)

        self.level_clock.tick()
        return cms_out

    # ------------------------------------------------------------------
    # Inner update (CMS levels)
    # ------------------------------------------------------------------

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    @torch.no_grad()
    def _update_cms(
        self,
        x: torch.Tensor,
        cms_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        if not self._passes_surprise(surprise_value):
            return

        # Self-modifier predicts the update direction.
        modifier = self.self_modifier(
            key=x.detach(),
            value=cms_out.detach(),
            error_signal=teach_signal.detach(),
        )

        # Update each CMS level if the level clock allows it.
        for spec in self.config.cms_levels:
            if not self.level_clock.should_update(spec.name):
                continue
            block = self.cms.blocks[spec.name]
            self._update_one_level(block, x.detach(), modifier.detach(), teach_signal.detach())
            self.level_clock.record_update(spec.name)

    def _update_one_level(
        self,
        block: nn.Module,
        x: torch.Tensor,
        modifier: torch.Tensor,
        teach_signal: torch.Tensor,
    ) -> None:
        """Gradient-based local update of a single CMS block.

        Uses the δ-shaped loss from the paper: construct a target such
        that ``∂loss/∂prediction ∝ δ`` (the teach signal), then apply
        one SGD step on the block parameters.
        """
        # Active mask: only update where teach signal is nonzero.
        active = teach_signal.abs().sum(dim=-1) > 0  # [B, T]
        if not active.any():
            return

        mask_f = active.unsqueeze(-1).float()  # [B, T, 1]

        # Enable grad temporarily for the local update.
        for p in block.parameters():
            p.requires_grad_(True)

        with torch.enable_grad():
            prediction = block(x)
            # δ-shaped loss: target = stopgrad(prediction - teach_signal)
            target = (prediction.detach() - teach_signal).detach()
            diff_sq = (prediction - target).pow(2)
            masked = diff_sq * mask_f
            loss = masked.sum() / mask_f.sum().clamp(min=1.0)

        # Manual SGD step on block parameters.
        grads = torch.autograd.grad(
            loss, list(block.parameters()),
            allow_unused=True,
        )
        lr = 1e-3
        for param, grad in zip(block.parameters(), grads):
            if grad is not None:
                param.data.sub_(lr * grad)
            param.requires_grad_(False)
