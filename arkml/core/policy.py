from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Minimal contract for all policies."""

    @abstractmethod
    def predict(self, obs: ..., **kwargs) -> ...:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def to_device(self, device: str):
        raise NotImplementedError

    def set_eval_mode(self):
        raise NotImplementedError

    def set_train_mode(self):
        raise NotImplementedError
