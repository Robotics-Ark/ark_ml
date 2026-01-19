from __future__ import annotations

from abc import ABC, abstractmethod

from omegaconf import DictConfig

from arkml.core.rl.termination_conditions.base_termination_conditions import (
    BaseTerminationCondition,
)


class TerminationProvider(ABC):
    """
    Build a set of termination conditions from config.
    """

    @abstractmethod
    def build(self, cfg: DictConfig) -> dict[str, BaseTerminationCondition]:
        """
        Return a mapping of termination names to conditions.
        """
        ...
