"""SelfModifier – learns parameter-update directions from error signals.

Localized from ``nested_learning.hope.self_mod.SelfModifier``.

Given ``(key, value, error_signal)`` triplets, the SelfModifier predicts
a modification delta that shapes the memory update direction.  This is
functionally equivalent to a learned optimization step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelfModifier(nn.Module):
    """Predict update deltas conditioned on key/value/error signals.

    Parameters
    ----------
    dim : int
        Feature dimension.
    hidden_multiplier : int
        MLP expansion ratio.
    """

    def __init__(self, dim: int, hidden_multiplier: int = 4) -> None:
        super().__init__()
        hidden = dim * hidden_multiplier
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        error_signal: torch.Tensor,
    ) -> torch.Tensor:
        concat = torch.cat([key, value, error_signal], dim=-1)
        return self.net(concat)
