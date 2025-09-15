import time
from typing import Any

import hydra
import numpy as np
from ark.env.ark_env import ArkEnv
from arktypes import flag_t
from arktypes import (
    task_space_command_t,
    pose_t,
    joint_state_t,
    rigid_body_state_t,
    rgbd_t,
)
from arktypes.utils import unpack
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def default_channels() -> dict[str, dict[str, type]]:
    """Default action/observation channels for the Franka + cube sim."""
    action_channels: dict[str, type] = {
        "franka/cartesian_command/sim": task_space_command_t,
    }
    observation_channels: dict[str, type] = {
        "franka/ee_state/sim": pose_t,
        "franka/joint_states/sim": joint_state_t,
        "cube/ground_truth/sim": rigid_body_state_t,
        "target/ground_truth/sim": rigid_body_state_t,
        "IntelRealSense/rgbd/sim": rgbd_t,
    }
    return {"actions": action_channels, "observations": observation_channels}


class RobotNode(ArkEnv):

    def __init__(self):
        config = (
            "ark_ml/arkml/examples/franka_pick_place/franka_config/global_config.yaml"
        )
        chans = default_channels()
        super().__init__(
            environment_name="diffusion_env",
            action_channels=chans["actions"],
            observation_channels=chans["observations"],
            global_config=config,
            sim=True,
        )

    def action_packing(self, action: Any) -> dict[str, Any]:
        """!Serialize an action.

        This method converts the high level action passed to :func:`step` into
        a dictionary that can be published over LCM.  The dictionary keys are
        channel names and the values are already packed LCM messages.

        @param action The high level action provided by the agent.
        @return A mapping from channel names to packed LCM messages.
        @rtype Dict[str, Any]
        """
        ...

    def observation_unpacking(self, observation_dict: dict[str, Any]) -> Any:
        """!Deserialize observations."""
        ...

    def terminated_truncated_info(
        self, state: Any, action: Any, next_state: Any
    ) -> tuple[bool, bool, Any]:
        """!Evaluate episode status."""
        return False, False, None

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Compute the reward for a transition"""
        ...

    def check_success(self, cube_pos, target_pos):
        """Compute termination and truncation from the current state.

        Supports both legacy dict states with ``cube``/``target`` entries and a
        flat vector layout ``[cube(3), target(3), gripper(1), ee(3)]``.

        Args:
          state: Current state (dict or flat vector).
          action: Applied action (unused).
          next_state: Next state (unused).

        Returns:
          tuple: ``(terminated, truncated, steps)`` where ``terminated`` is
          True if cube is within 0.1m of target; ``truncated`` is True if
          ``max_steps`` reached; and ``steps`` is current step count.
        """

        distance = np.linalg.norm(cube_pos - target_pos)
        terminated = bool(distance < 0.2)
        return (terminated,)

    def reset_objects(self):
        """Reset key scene objects and internal step counter."""
        self.steps = 0
        self.reset_component("cube")
        self.reset_component("target")
        self.reset_component("franka")

    def reset_scene(self) -> None:
        """Minimal scene bootstrap: reset objects and exit.

        This script no longer bridges observations/actions. The PolicyNode handles
        that directly. We only ensure the scene is reset and ready.
        """

        # Reset the policy node state at the beginning of the episode
        try:

            self.send_service_request(
                service_name="Policy/policy/reset",
                request=flag_t(),
                response_type=flag_t,
            )
            self.reset()
        except Exception as e:
            print(f"Warning: Failed to reset policy via service: {e}")
        # Give subsystems a moment to settle
        time.sleep(1.0)


@hydra.main(
    config_path="../../configs", config_name="defaults.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    """Run rollouts for a configured policy.

    Args:
      cfg: Hydra configuration composed of ``defaults.yaml`` and overrides.

    Returns:
      None. Prints progress and a final success summary to stdout.
    """
    print("Config:\n", OmegaConf.to_yaml(cfg))

    step_sleep = float(getattr(cfg, "step_sleep", 0.1))
    n_episodes = int(getattr(cfg, "n_episodes", 1))
    max_step = int(getattr(cfg, "max_steps", 1))

    robo_node = RobotNode()

    success_count = 0
    failure_count = 0

    for ep in tqdm(range(n_episodes), desc="Episodes", unit="ep"):
        robo_node.reset_scene()
        success = False
        time.sleep(1)
        for step_count in tqdm(
            range(max_step + 1), desc=f"Ep {ep}", unit="step", leave=False
        ):
            robo_node.observation_space.wait_until_observation_space_is_ready()
            obs_dict = robo_node.observation_space.get_observation()
            if obs_dict is None:
                time.sleep(step_sleep)
                continue

            cube_state = obs_dict["cube/ground_truth/sim"]
            target_state = obs_dict["target/ground_truth/sim"]

            if not obs_dict or any(v is None for v in obs_dict.values()):
                continue

            _, cube_position, _, _, _ = unpack.rigid_body_state(cube_state)
            _, target_position, _, _, _ = unpack.rigid_body_state(target_state)

            terminated = robo_node.check_success(cube_position, target_position)
            if terminated:
                success = True
                break

            step_count += 1
            time.sleep(step_sleep)

        if success:
            success_count += 1
        else:
            failure_count += 1

    # After all episodes
    success_rate = success_count / n_episodes * 100

    print("\n=== Results ===")
    print(f"Total Episodes : {n_episodes}")
    print(f"Successes      : {success_count}")
    print(f"Failures       : {failure_count}")
    print(f"Success Rate   : {success_rate:.2f}%")


if __name__ == "__main__":
    main()
