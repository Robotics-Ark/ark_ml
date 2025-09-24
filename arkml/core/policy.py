from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class BasePolicy(ABC, nn.Module):
    """
    Abstract base class defining the minimal interface for a policy.
    """

    @abstractmethod
    def predict(self, obs: dict[str, Any], **kwargs) -> Any:
        """
        Given the current observation (and optional keyword arguments),
        return the action(s) chosen by the policy.
        """
        ...

    @abstractmethod
    def reset(self):
        """
        Reset any internal state of the policy, typically called at the beginning
        of a new episode or sequence.
        """
        ...

    @abstractmethod
    def to_device(self, device: str):
        """
        Move the policy and its parameters to the specified device
        (e.g., "cpu", "cuda").
        """
        ...

    def set_eval_mode(self):
        """
        Switch the policy into evaluation mode, disabling behaviors such as dropout
        or exploration strategies used during training.
        """
        ...

    def set_train_mode(self):
        """
        Switch the policy into training mode, enabling behaviors such as dropout
        or exploration strategies used during learning.
        """
        ...
