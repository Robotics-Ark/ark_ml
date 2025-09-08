from abc import ABC, abstractmethod


from .types import Observation, Action


class BasePolicy(ABC):
    """Minimal contract for all policies."""

    @abstractmethod
    def predict(self, obs: Observation, **kwargs) -> Action:
        ...

    def reset(self):
        ...

    def to_device(self, device: str):
        return self
