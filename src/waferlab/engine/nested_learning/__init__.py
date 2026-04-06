"""Nested Learning core modules (localized from nested_learning repo).

This package contains the core mechanisms from the Nested Learning / HOPE
framework, adapted for 2D wafer-map image classification:

- **LevelSpec / LevelClock**: Multi-frequency update scheduling.
- **CMSBlock / CMS**: Continuum Memory System with multi-frequency updates.
- **SelfModifier**: Teaches parameter updates conditioned on error signals.
- **NestedBlock**: Full HOPE-style block combining CMS + self-modifier +
  surprise gating for token sequences extracted from CNN feature maps.
"""

from .levels import LevelSpec, LevelClock
from .cms import CMSBlock, CMS
from .self_modifier import SelfModifier
from .nested_block import NestedBlock, NestedBlockConfig

__all__ = [
    "LevelSpec",
    "LevelClock",
    "CMSBlock",
    "CMS",
    "SelfModifier",
    "NestedBlock",
    "NestedBlockConfig",
]
