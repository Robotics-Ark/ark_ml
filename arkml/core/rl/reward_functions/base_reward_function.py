from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any


class BaseRewardFunction(metaclass=ABCMeta):
    """
    Base RewardFunction class
    """

    def __init__(self):
        """Initialize the reward function state."""
        self._reward = None
        self._info = None

    @abstractmethod
    def _step(self, obs) -> tuple[float, dict]:
        """
        Compute the reward and info dictionary for a single timestep.
        Args:
            obs: The current environment observation.

        Returns:
            reward : The scalar reward value for this timestep.
            info : Additional diagnostic information about the reward computation.
        """

        raise NotImplementedError()

    def step(self, obs) -> tuple[float, dict]:
        """
        Compute and record the reward and info for the current step.
        Args:
            obs: The current environment observation.

        Returns:
            reward : The computed reward.
            info : A deep-copied dictionary of auxiliary information about the reward.
        """

        self._reward, self._info = self._step(obs=obs)

        return self._reward, deepcopy(self._info)

    def reset(self) -> None:
        """Reset internal reward state."""
        # Reset internal vars
        self._reward = None
        self._info = None

    @property
    def reward(self) -> float:
        """
        Get the most recently computed reward.
        Returns:
            The last reward computed by `step()`.
        """
        assert (
            self._reward is not None
        ), "At least one step() must occur before reward can be calculated!"

        return self._reward

    @property
    def info(self) -> dict[str, Any]:
        """
        Get the most recently computed info dictionary.
        Returns:
            The info dictionary from the last `step()` call.
        """
        assert (
            self._info is not None
        ), "At least one step() must occur before info can be calculated!"

        return self._info
