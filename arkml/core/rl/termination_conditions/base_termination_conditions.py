from abc import ABCMeta, abstractmethod


class BaseTerminationCondition(metaclass=ABCMeta):
    """
    Base TerminationCondition class
    Condition-specific _step() method is implemented in subclasses
    """

    def __init__(self):
        # Initialize internal vars that will be filled in at runtime
        self._done = None

    @abstractmethod
    def _step(self, obs, action):
        """
        Step the termination condition and return whether the episode should terminate. Overwritten by subclasses.

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            bool: whether environment should terminate or not
        """
        raise NotImplementedError()

    def step(self, obs, action):
        """
        Step the termination condition and return whether the episode should terminate.

        Args:
            obs (dict): Observation at current timestep
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            2-tuple:
                - bool: whether environment should terminate or not
                - bool: whether a success was reached under this termination condition
        """
        # Step internally and store the done state internally as well
        self._done = self._step(obs=obs, action=action)

        # We are successful if done is True AND this is a success condition
        success = self._done and self._terminate_is_success

        return self._done, success

    def reset(self):
        """
        Termination condition-specific reset

        """
        # Reset internal vars
        self._done = None

    @property
    def done(self):
        """
        Returns:
            bool: Whether this termination condition has triggered or not
        """
        assert (
            self._done is not None
        ), "At least one step() must occur before done can be calculated!"
        return self._done

    @property
    def success(self):
        """
        Returns:
            bool: Whether this termination condition has been evaluated as a success or not
        """
        assert (
            self._done is not None
        ), "At least one step() must occur before success can be calculated!"
        return self._done and self._terminate_is_success

    @property
    def _terminate_is_success(self):
        """
        Returns:
            bool: Whether this termination condition corresponds to a success
        """
        raise NotImplementedError()


class SuccessCondition(BaseTerminationCondition):
    """
    Termination condition corresponding to a success
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of locomotion controllers
        super().__init_subclass__(**kwargs)

    @property
    def _terminate_is_success(cls):
        # Done --> success
        return True


class FailureCondition(BaseTerminationCondition):
    """
    Termination condition corresponding to a failure
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of locomotion controllers
        super().__init_subclass__(**kwargs)

    @property
    def _terminate_is_success(self):
        # Done --> not success
        return False
