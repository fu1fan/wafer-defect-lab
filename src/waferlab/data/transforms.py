"""Shared transforms and input preprocessing for wafer map data."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


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
    horizontal/vertical flips, 90-degree rotations, and optional whole-wafer
    translation / mild scale jitter via nearest-neighbor affine transforms.
    """

    def __init__(
        self,
        *,
        random_flip: bool = True,
        random_rotate90: bool = True,
        random_translate_frac: float = 0.0,
        random_scale_min: float = 1.0,
        random_scale_max: float = 1.0,
    ) -> None:
        self.random_flip = random_flip
        self.random_rotate90 = random_rotate90
        self.random_translate_frac = max(float(random_translate_frac), 0.0)
        self.random_scale_min = float(random_scale_min)
        self.random_scale_max = float(random_scale_max)
        if self.random_scale_min <= 0 or self.random_scale_max <= 0:
            raise ValueError("random_scale_min/max must be positive.")
        if self.random_scale_min > self.random_scale_max:
            raise ValueError("random_scale_min cannot exceed random_scale_max.")

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]  # [1, H, W]

        do_hflip = self.random_flip and torch.rand(1).item() > 0.5
        do_vflip = self.random_flip and torch.rand(1).item() > 0.5
        k = int(torch.randint(0, 4, (1,)).item()) if self.random_rotate90 else 0
        tx, ty = self._sample_translate(img)
        scale = self._sample_scale()

        img = self._apply(img, do_hflip, do_vflip, k, tx=tx, ty=ty, scale=scale)
        sample["image"] = img

        for mask_key in ("wafer_mask", "defect_mask"):
            if mask_key in sample:
                sample[mask_key] = self._apply(
                    sample[mask_key], do_hflip, do_vflip, k, tx=tx, ty=ty, scale=scale,
                )

        return sample

    def _sample_translate(self, img: torch.Tensor) -> tuple[int, int]:
        if self.random_translate_frac <= 0:
            return 0, 0

        h, w = img.shape[-2:]
        max_tx = int(round(w * self.random_translate_frac))
        max_ty = int(round(h * self.random_translate_frac))
        tx = int(torch.randint(-max_tx, max_tx + 1, (1,)).item()) if max_tx > 0 else 0
        ty = int(torch.randint(-max_ty, max_ty + 1, (1,)).item()) if max_ty > 0 else 0
        return tx, ty

    def _sample_scale(self) -> float:
        if self.random_scale_min == self.random_scale_max == 1.0:
            return 1.0
        return float(
            torch.empty(1).uniform_(self.random_scale_min, self.random_scale_max).item(),
        )

    @staticmethod
    def _apply(
        t: torch.Tensor,
        hflip: bool,
        vflip: bool,
        k: int,
        *,
        tx: int = 0,
        ty: int = 0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if hflip:
            t = t.flip(-1)
        if vflip:
            t = t.flip(-2)
        if k > 0:
            t = torch.rot90(t, k, dims=(-2, -1))
        if tx != 0 or ty != 0 or abs(scale - 1.0) > 1e-6:
            t = TF.affine(
                t,
                angle=0.0,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )
        return t


# ── Wafer-safe Random Erasing ────────────────────────────────────────

class WaferRandomErasing:
    """Randomly erase a rectangular region of the wafer map.

    Fills the erased region with 0 (background).  Safe for wafer maps because
    0 is a valid discrete background value, effectively simulating partial
    observation or occluded regions.

    Parameters
    ----------
    p : float
        Probability of applying erasing per sample.
    scale_range : tuple[float, float]
        Range of erased area as a fraction of total image area.
    ratio_range : tuple[float, float]
        Range of aspect ratios for the erased rectangle.
    """

    def __init__(
        self,
        p: float = 0.5,
        scale_range: tuple[float, float] = (0.02, 0.15),
        ratio_range: tuple[float, float] = (0.3, 3.3),
    ) -> None:
        self.p = p
        self.scale_range = scale_range
        self.ratio_range = ratio_range

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() > self.p:
            return sample

        img = sample["image"]  # [C, H, W]
        h, w = img.shape[-2:]
        area = h * w

        for _ in range(10):  # retry until a valid rectangle is found
            target_area = area * float(
                torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
            )
            log_ratio = torch.empty(1).uniform_(
                math.log(self.ratio_range[0]), math.log(self.ratio_range[1]),
            )
            aspect = math.exp(float(log_ratio.item()))
            eh = int(round(math.sqrt(target_area * aspect)))
            ew = int(round(math.sqrt(target_area / aspect)))
            if 0 < eh < h and 0 < ew < w:
                top = int(torch.randint(0, h - eh, (1,)).item())
                left = int(torch.randint(0, w - ew, (1,)).item())
                img[..., top : top + eh, left : left + ew] = 0.0
                break

        sample["image"] = img
        return sample


# ── Class-aware augmentation ─────────────────────────────────────────

class ClassAwareAugmentation:
    """Apply stronger augmentation to minority classes.

    Reads ``sample["metadata"]["failure_type"]`` and routes to a *strong*
    augmentation pipeline for minority classes, or a *normal* pipeline for
    majority classes.  Minority samples additionally get random erasing.

    Requires ``include_metadata=True`` on the dataset.

    Parameters
    ----------
    minority_classes : set[str] or None
        Class names considered minority.  Defaults to the five weakest
        WM-811K defect types.
    random_flip, random_rotate90, random_translate_frac,
    random_scale_min, random_scale_max :
        Normal (majority) augmentation parameters.
    minority_translate_frac : float
        Translate fraction for minority augmentation (default 0.12).
    minority_scale_min, minority_scale_max : float
        Scale jitter range for minority augmentation.
    minority_erasing_p : float
        Random-erasing probability for minority samples.
    """

    _DEFAULT_MINORITY: set[str] = {"Donut", "Near-full", "Random", "Scratch", "Loc"}

    def __init__(
        self,
        *,
        minority_classes: set[str] | None = None,
        # normal (majority) augmentation params
        random_flip: bool = True,
        random_rotate90: bool = True,
        random_translate_frac: float = 0.0,
        random_scale_min: float = 1.0,
        random_scale_max: float = 1.0,
        # stronger minority augmentation params
        minority_translate_frac: float = 0.12,
        minority_scale_min: float = 0.90,
        minority_scale_max: float = 1.10,
        minority_erasing_p: float = 0.3,
    ) -> None:
        self.minority_classes = minority_classes or self._DEFAULT_MINORITY

        self.normal_aug = WaferAugmentation(
            random_flip=random_flip,
            random_rotate90=random_rotate90,
            random_translate_frac=random_translate_frac,
            random_scale_min=random_scale_min,
            random_scale_max=random_scale_max,
        )
        self.strong_aug = WaferAugmentation(
            random_flip=random_flip,
            random_rotate90=random_rotate90,
            random_translate_frac=minority_translate_frac,
            random_scale_min=minority_scale_min,
            random_scale_max=minority_scale_max,
        )
        self.erasing = (
            WaferRandomErasing(p=minority_erasing_p)
            if minority_erasing_p > 0
            else None
        )

    def __call__(self, sample: dict) -> dict:
        meta = sample.get("metadata", {})
        ft = str(meta.get("failure_type", "none"))

        if ft in self.minority_classes:
            sample = self.strong_aug(sample)
            if self.erasing is not None:
                sample = self.erasing(sample)
        else:
            sample = self.normal_aug(sample)

        return sample


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
