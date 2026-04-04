"""Shared transforms and input preprocessing for wafer map data."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch


# ── Input normalization ──────────────────────────────────────────────

# Wafer map discrete values: 0=background, 1=normal_die, 2=defect_die.
# Default normalization maps {0,1,2} -> [0, 1] by dividing by the max value.
DEFAULT_NORM_SCALE = 2.0


def prepare_input(
    batch: dict[str, Any],
    *,
    device: torch.device,
    target_channels: int = 1,
    norm_scale: float = DEFAULT_NORM_SCALE,
) -> torch.Tensor:
    """Normalize a wafer-map image batch for model consumption.

    Parameters
    ----------
    batch : dict
        Must contain ``"image"`` tensor of shape ``[B, 1, H, W]``.
    device : torch.device
        Target device.
    target_channels : int
        If 3, expands single-channel input to 3-channel via repeat.
    norm_scale : float
        Divide pixel values by this number (default 2.0 for {0,1,2} -> [0,1]).
    """
    x = batch["image"].to(device)
    if target_channels == 3 and x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    return x / norm_scale


# ── Spatial augmentations ────────────────────────────────────────────

class WaferAugmentation:
    """Simple spatial augmentations safe for wafer maps.

    Only applies transformations that preserve discrete wafer-map semantics:
    horizontal/vertical flips and 90-degree rotations.
    """

    def __init__(self, *, random_flip: bool = True, random_rotate90: bool = True) -> None:
        self.random_flip = random_flip
        self.random_rotate90 = random_rotate90

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]  # [1, H, W]

        do_hflip = self.random_flip and torch.rand(1).item() > 0.5
        do_vflip = self.random_flip and torch.rand(1).item() > 0.5
        k = int(torch.randint(0, 4, (1,)).item()) if self.random_rotate90 else 0

        img = self._apply(img, do_hflip, do_vflip, k)
        sample["image"] = img

        for mask_key in ("wafer_mask", "defect_mask"):
            if mask_key in sample:
                sample[mask_key] = self._apply(sample[mask_key], do_hflip, do_vflip, k)

        return sample

    @staticmethod
    def _apply(t: torch.Tensor, hflip: bool, vflip: bool, k: int) -> torch.Tensor:
        if hflip:
            t = t.flip(-1)
        if vflip:
            t = t.flip(-2)
        if k > 0:
            t = torch.rot90(t, k, dims=(-2, -1))
        return t


# ── Label injection ──────────────────────────────────────────────────

class InjectFailureTypeIdx:
    """Add ``failure_type_idx`` field for multi-class mode.

    Requires ``include_metadata=True`` on the dataset so that
    ``sample["metadata"]["failure_type"]`` is available.
    """

    def __init__(self, label_map: dict[str, int] | None = None) -> None:
        from ..models.resnet import FAILURE_TYPE_TO_IDX
        self.label_map = label_map or FAILURE_TYPE_TO_IDX

    def __call__(self, sample: dict) -> dict:
        meta = sample.get("metadata", {})
        ft = str(meta.get("failure_type", "none"))
        sample["failure_type_idx"] = self.label_map.get(ft, 0)
        return sample


# ── Compose utility ──────────────────────────────────────────────────

def compose(fns: Sequence[Callable]) -> Callable | None:
    """Chain a sequence of sample transforms, returning None if empty."""
    if not fns:
        return None

    def _apply(sample: dict) -> dict:
        for fn in fns:
            sample = fn(sample)
        return sample

    return _apply
