from arkml.core.rl.reward_functions.base_reward_function import (
    BaseRewardFunction,
)


class PotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)

    Args:
        potential_fcn (method): function for calculating potential. Function signature should be:

            potential = potential_fcn(env)

            where @env is a Environment instance, and @potential is a float value representing the calculated potential

        r_potential (float): Reward weighting to give proportional to the potential difference calculated
            in between env timesteps
    """

    def __init__(self, potential_fcn, r_potential=1.0):
        # Store internal vars
        self._potential_fcn = potential_fcn
        self._r_potential = r_potential

        # Store internal vars that will be filled in at runtime
        self._potential = None

        # Run super
        super().__init__()

    def reset(self, initial_obs=None):
        """
        Compute the initial potential after episode reset

        :param initial_obs: optional initial observation
        """
        # Reset potential
        self._potential = (
            None if initial_obs is None else self._potential_fcn(initial_obs)
        )

    def _step(self, obs, action):
        # Reward is proportional to the potential difference between the current and previous timestep
        if self._potential is None:
            self._potential = self._potential_fcn(obs)
            return 0.0, {}

        new_potential = self._potential_fcn(obs)
        reward = (self._potential - new_potential) * self._r_potential

        # Update internal potential
        self._potential = new_potential

        return reward, {}
