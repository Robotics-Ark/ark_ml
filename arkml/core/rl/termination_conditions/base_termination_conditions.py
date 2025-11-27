from abc import ABCMeta, abstractmethod


class BaseTerminationCondition(metaclass=ABCMeta):
    """Base class for defining termination conditions in an environment."""

    def __init__(self):
        """Initialize the internal variables."""
        self._done = None

    @abstractmethod
    def _step(self, obs):
        """Compute whether the termination condition is met for the current step."""
        raise NotImplementedError()

    def step(self, obs):
        """
        Evaluate the termination condition and return `done` and `success`.
        Args:
            obs: Current environment observation.

        Returns:
            done : True if the episode should terminate.
            success : True if the termination counts as a successful episode.
        """
        self._done = self._step(obs=obs)

        success = self._done and self._terminate_is_success

        return self._done, success

    def reset(self):
        """Reset the internal termination state."""
        self._done = None

    @property
    def done(self) -> bool:
        """
        Get whether the termination condition has been triggered.
        Returns:
            True if the episode is done.

        """
        assert (
            self._done is not None
        ), "At least one step() must occur before done can be calculated!"
        return self._done

    @property
    def success(self) -> bool:
        """
        Get whether the episode is considered successful.
        Returns:
            True if the termination condition was met and counts as success.
        """
        assert (
            self._done is not None
        ), "At least one step() must occur before success can be calculated!"
        return self._done and self._terminate_is_success

    @property
    def _terminate_is_success(self):
        raise NotImplementedError()


class SuccessCondition(BaseTerminationCondition):
    """Termination condition that always counts as a success when triggered."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @property
    def _terminate_is_success(cls):
        return True


class FailureCondition(BaseTerminationCondition):
    """Termination condition that always counts as a failure when triggered."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @property
    def _terminate_is_success(self):
        return False
