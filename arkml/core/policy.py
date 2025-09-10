from abc import ABC, abstractmethod

from torch import nn

from .types import Observation, Action


class BasePolicy(ABC):
    """Minimal contract for all policies."""

    @abstractmethod
    def predict(self, obs: ..., **kwargs) -> Action: # TODO check type
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def to_device(self, device: str):
        raise NotImplementedError
