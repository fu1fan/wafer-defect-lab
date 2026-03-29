"""GradCAM implementation for wafer-level classification models.

These heatmaps are *interpretability aids* that show which spatial regions the
classifier attends to when making a prediction.  They are **not** equivalent to
true anomaly-localization maps (e.g. from PaDiM or PatchCore) which
characterise pixel-level deviation from a learned normal-feature distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.classifier import WaferClassifier


class GradCAM:
    """Gradient-weighted Class Activation Mapping for a WaferClassifier.

    Usage::

        cam = GradCAM(model)
        heatmaps = cam(images)            # shape [B, H, W], values in [0, 1]
        cam.remove_hooks()
    """

    def __init__(self, model: WaferClassifier, target_class: int | None = None) -> None:
        self.model = model
        self.target_class = target_class
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target_layer = model.get_cam_target_layer()
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: Any, input: Any, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module: Any, grad_input: Any, grad_output: Any) -> None:
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Compute GradCAM heatmaps.

        Parameters
        ----------
        x : Tensor [B, C, H, W]
            Already-normalised input.

        Returns
        -------
        heatmaps : ndarray [B, H, W]  in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)

        if self.target_class is not None:
            target = logits[:, self.target_class]
        else:
            target = logits.gather(1, logits.argmax(1, keepdim=True)).squeeze(1)

        self.model.zero_grad()
        target.sum().backward()

        assert self._activations is not None and self._gradients is not None

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [B, 1, h, w]
        cam = F.relu(cam)

        # Upsample to input spatial size.
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)  # [B, H, W]

        # Per-sample min-max normalisation.
        B = cam.shape[0]
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(B, x.shape[2], x.shape[3])

        return cam.cpu().numpy()

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def generate_cam_for_batch(
    model: WaferClassifier,
    batch: dict[str, Any],
    *,
    device: str = "cuda",
    target_class: int | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: run GradCAM on a data batch.

    Returns a dict with ``heatmaps`` (ndarray [B, H, W]),
    ``predictions``, ``labels``, ``sample_ids``.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    x = batch["image"].to(dev)
    if model.in_channels == 3 and x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    x = x / 2.0

    cam = GradCAM(model, target_class=target_class)
    heatmaps = cam(x)
    cam.remove_hooks()

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()

    result: dict[str, Any] = {
        "heatmaps": heatmaps,
        "predictions": preds,
        "input_images": batch["image"].numpy(),
    }
    if "label" in batch:
        result["labels"] = batch["label"].numpy()
    if "sample_id" in batch:
        result["sample_ids"] = batch["sample_id"].numpy()
    return result
