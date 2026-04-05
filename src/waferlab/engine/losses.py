"""Loss functions for imbalanced wafer-map classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Focal loss down-weights well-classified examples so the model focuses
    on hard, misclassified samples.  Particularly effective for severe
    class imbalance.

    .. math::
        FL(p_t) = -\\alpha_t \\,(1 - p_t)^\\gamma \\,\\log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter.  ``gamma = 0`` reduces to standard CE.
    alpha : list[float] or None
        Per-class weight vector (length = num_classes).  If *None*,
        all classes are weighted equally.
    reduction : str
        ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : Tensor [B, C]
        targets : Tensor [B]  (class indices)
        """
        log_p = F.log_softmax(logits, dim=1)              # [B, C]
        p = log_p.exp()                                     # [B, C]

        # Gather the probability of the true class.
        targets_1d = targets.view(-1, 1)                    # [B, 1]
        log_pt = log_p.gather(1, targets_1d).squeeze(1)    # [B]
        pt = p.gather(1, targets_1d).squeeze(1)            # [B]

        focal_weight = (1.0 - pt) ** self.gamma             # [B]

        loss = -focal_weight * log_pt                       # [B]

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]  # [B]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
