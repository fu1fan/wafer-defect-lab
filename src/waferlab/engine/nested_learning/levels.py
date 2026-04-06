"""Multi-frequency level scheduling for Nested Learning.

Localized from ``nested_learning.levels``.  Each ``LevelSpec`` defines
a named memory level with an independent ``update_period`` (in training
steps).  The ``LevelClock`` tracks when each level should be updated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping, Sequence


@dataclass(frozen=True)
class LevelSpec:
    """Configuration for a single nested-learning memory level."""

    name: str
    update_period: int
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if self.update_period <= 0:
            raise ValueError(
                f"update_period for level {self.name} must be positive"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps for level {self.name} must be non-negative"
            )


@dataclass
class LevelState:
    last_step: int = -1
    updates: int = 0


class LevelClock:
    """Deterministic scheduler for Nested Learning level updates."""

    def __init__(self, specs: Sequence[LevelSpec]) -> None:
        self._specs: Dict[str, LevelSpec] = {s.name: s for s in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate level names")
        self._state: MutableMapping[str, LevelState] = {
            name: LevelState() for name in self._specs
        }
        self._step: int = 0

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> None:
        self._step += 1

    def should_update(self, name: str) -> bool:
        spec = self._specs[name]
        state = self._state[name]
        if self._step < spec.warmup_steps:
            return False
        delta = self._step - state.last_step
        return state.last_step < 0 or delta >= spec.update_period

    def record_update(self, name: str) -> None:
        state = self._state[name]
        state.last_step = self._step
        state.updates += 1

    def stats(self) -> Dict[str, LevelState]:
        return {
            name: LevelState(s.last_step, s.updates)
            for name, s in self._state.items()
        }
