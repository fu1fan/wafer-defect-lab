"""Prototype memory with surprise-gated updates.

Inspired by the Nested Learning / CMS multi-frequency memory mechanism:
- Per-class feature prototypes maintained as EMA running centroids.
- Surprise gating: prototypes are updated more aggressively for
  "surprising" samples (high per-sample loss), mirroring the teach-signal
  surprise threshold in Nested Learning.
- Auxiliary alignment loss: pulls features toward their class prototypes
  to encourage tighter, more discriminative feature clusters.

The memory operates **only during training**; validation/test are unaffected.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PrototypeMemory:
    """Per-class feature prototypes with surprise-gated EMA updates.

    Parameters
    ----------
    feat_dim : int
        Dimensionality of the feature vector (output of ``forward_features``).
    num_classes : int
        Number of classification targets.
    momentum : float
        EMA momentum for prototype updates (higher = slower drift).
    surprise_threshold : float
        Per-sample loss must exceed this value to trigger a prototype update.
        Set to 0 to disable gating (all samples contribute).
    aux_weight : float
        Weight of the auxiliary prototype alignment loss added to the
        main classification loss.  0 disables the auxiliary term.
    warmup_epochs : int
        Number of training epochs to run before activating the auxiliary
        loss.  Prototypes are still updated (passively) during warmup so
        they converge before being used as a loss target.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        momentum: float = 0.99,
        surprise_threshold: float = 0.0,
        aux_weight: float = 0.1,
        warmup_epochs: int = 3,
    ) -> None:
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.surprise_threshold = surprise_threshold
        self.aux_weight = aux_weight
        self.warmup_epochs = warmup_epochs

        # Prototypes and initialization flags (moved to device lazily).
        self.prototypes = torch.zeros(num_classes, feat_dim)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool)
        self._device_set = False

    def _ensure_device(self, device: torch.device) -> None:
        if not self._device_set or self.prototypes.device != device:
            self.prototypes = self.prototypes.to(device)
            self.initialized = self.initialized.to(device)
            self._device_set = True

    # ------------------------------------------------------------------
    # Surprise-gated update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        per_sample_loss: torch.Tensor,
    ) -> None:
        """Update prototypes using surprise-gated EMA.

        Parameters
        ----------
        features : Tensor [B, D]
            Detached feature vectors from ``model.forward_features``.
        labels : Tensor [B]
            Ground-truth class indices.
        per_sample_loss : Tensor [B]
            Un-reduced per-sample cross-entropy loss used as surprise signal.
        """
        self._ensure_device(features.device)

        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue

            c_feat = features[mask]
            c_loss = per_sample_loss[mask]

            # Surprise gating: keep only samples with loss above threshold.
            if self.surprise_threshold > 0:
                surprise_mask = c_loss > self.surprise_threshold
                if not surprise_mask.any():
                    continue
                c_feat = c_feat[surprise_mask]

            centroid = c_feat.mean(dim=0)

            if not self.initialized[c]:
                self.prototypes[c] = centroid
                self.initialized[c] = True
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c]
                    + (1 - self.momentum) * centroid
                )

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def alignment_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prototype alignment loss (MSE to class prototypes).

        Returns a scalar loss weighted by ``self.aux_weight``.  If no
        prototypes have been initialized yet the loss is zero.
        """
        if not self.initialized.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Only use classes that have an initialized prototype.
        valid = self.initialized[labels]
        if not valid.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        target_protos = self.prototypes[labels].detach()  # [B, D]
        loss = F.mse_loss(features[valid], target_protos[valid])
        return loss * self.aux_weight

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "prototypes": self.prototypes.cpu(),
            "initialized": self.initialized.cpu(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.prototypes = state["prototypes"]
        self.initialized = state["initialized"]
        self._device_set = False
