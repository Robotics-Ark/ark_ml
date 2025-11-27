from arkml.core.rl.reward_functions.base_reward_function import BaseRewardFunction


class PointGoalReward(BaseRewardFunction):
    """Reward function that provides a fixed reward when a point-goal is reached."""

    def __init__(self, pointgoal, r_pointgoal=10.0):
        """Initialize the PointGoalReward function."""
        self._pointgoal = pointgoal
        self._r_pointgoal = r_pointgoal

        # Run super
        super().__init__()

    def _step(self, obs) -> tuple[float, dict]:
        """
        Compute the reward for the current observation.
        Args:
            obs: Current environment observation. Not used in this reward function.

        Returns:
            reward : The computed reward for this step (either `r_pointgoal` or 0.0).

        """
        reward = self._r_pointgoal if self._pointgoal.success else 0.0

        return reward, {}  # TODO check do we need info empty dict here
