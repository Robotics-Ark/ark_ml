from arkml.core.rl.reward_functions.base_reward_function import BaseRewardFunction


class PotentialReward(BaseRewardFunction):
    """Reward function based on the difference in a potential function over time."""

    def __init__(self, potential_fcn, r_potential=1.0):
        """Initialize the PotentialReward function."""
        self._potential_fcn = potential_fcn
        self._r_potential = r_potential
        self._potential = None

        super().__init__()

    def reset(self) -> None:
        """Reset the internal potential state."""
        self._potential = None

    def _step(self, obs) -> tuple[float, dict]:
        """
        Compute the reward based on the potential difference between timesteps.
        Args:
            obs: Current environment observation.

        Returns:
            reward : Reward proportional to the decrease in potential.
            info : Empty dictionary (included for compatibility with the base class).

        """
        if self._potential is None:
            self._potential = self._potential_fcn(obs)
            return 0.0, {}

        new_potential = self._potential_fcn(obs)
        reward = (self._potential - new_potential) * self._r_potential

        self._potential = new_potential

        return reward, {}
