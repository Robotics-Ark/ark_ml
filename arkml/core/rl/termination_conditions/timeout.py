from __future__ import annotations

from arkml.core.rl.termination_conditions.base_termination_conditions import (
    FailureCondition,
)


class Timeout(FailureCondition):
    """
    Terminate episode after a maximum number of steps (counts as failure).
    """

    def __init__(self, max_steps: int = 500):
        self._max_steps = int(max_steps)
        self._steps = 0
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self._steps = 0

    def _step(self, obs) -> bool:  # obs unused; kept for interface compatibility
        self._steps += 1
        return self._steps >= self._max_steps
