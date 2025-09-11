from abc import abstractmethod, ABC

import numpy as np
from torch import nn


class PolicyNode(ABC):
    """Abstract base class for policy wrappers.

    Args:
      policy: Underlying policy module to be executed.
      device: Target device identifier (e.g., ``"cpu"``, ``"cuda"``).
    """

    def __init__(self, policy: nn.Module, device: str):
        self.policy = policy
        self.policy.to_device(device=device)
        self.policy.set_eval_mode()

    def reset(self) -> None:
        """Reset the internal state of the policy.

        This is a no-op for stateless policies, but sequence models or planners
        may override this behavior via the wrapped ``policy.reset()``.
        """

        self.policy.reset()

    def infer(self, obs_seq: dict[str, ...]) -> np.ndarray:
        """Backward-compatible alias for ``predict``.

        Args:
          obs_seq: Observation input for the policy. Subclasses must document
            the required structure and shapes.

        Returns:
          np.ndarray: Predicted action(s) as a NumPy array.
        """

        return self.predict(obs_seq)

    @abstractmethod
    def predict(self, obs_seq: dict[str, ...]) -> np.ndarray:
        """Compute the action(s) from observations.

        Args:
          obs_seq: Observation input for the policy. Subclasses must document
            the required structure and shapes.

        Returns:
          np.ndarray: Predicted action(s) as a NumPy array.
        """

        pass
