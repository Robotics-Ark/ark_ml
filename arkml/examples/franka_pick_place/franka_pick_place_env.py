import time
from typing import Any

import numpy as np
from ark.env.ark_env import ArkEnv
from arktypes.utils import pack, unpack
from omegaconf import DictConfig, OmegaConf

from arktypes import (
    task_space_command_t,
    pose_t,
    joint_state_t,
    rigid_body_state_t,
    rgbd_t,
)
from arktypes import flag_t

from abc import ABC, abstractmethod


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


# from ark.client.comm_infrastructure.base_node import BaseNode


class RobotEnv(ArkEnv):
    """Ark environment wrapper customized for Franka + cube simulation.

    Args:
      config: Global Ark configuration object.
      environment_name: Name/identifier of the Ark environment to load.
      action_channels: Mapping of action channels.
      observation_channels: Mapping of observation channels names.
      sim: Whether to run in simulation mode.
    """

    def __init__(
        self,
        config: str,
        environment_name: str,
        action_channels: dict[str, type],
        observation_channels: dict[str, type],
        max_steps: int,
        step_sleep: float,
        sim=True,
    ):
        super().__init__(
            environment_name=environment_name,
            action_channels=action_channels,
            observation_channels=observation_channels,
            global_config=config,
            sim=sim,
        )
        self.sim = sim
        self.max_steps = max_steps
        self.steps = 0
        self.step_sleep = step_sleep

    @staticmethod
    def action_packing(action: list) -> dict[str, Any]:
        """Pack action into Ark cartesian command message.

        Expects an 8D vector representing end-effector position, orientation
        quaternion, and gripper command in the following order:
        ``[x, y, z, qx, qy, qz, qw, gripper]``.

        Args:
          action: List describing the cartesian command.

        Returns:
          dict[str, ...]: Mapping with key pointing to a packed Ark message.
        """

        xyz_command = np.array(action[:3])
        quaternion_command = np.array(action[3:7])
        gripper_command = action[7]

        franka_cartesian_command = pack.task_space_command(
            "all", xyz_command, quaternion_command, gripper_command
        )
        return {"franka/cartesian_command/sim": franka_cartesian_command}

    @staticmethod
    def observation_unpacking(observation_dict):
        """Unpack raw Ark observations into structured components.

        Converts incoming channel messages into a dictionary with primitive
        types useful for policies.

        Returns a dictionary with keys:
          - ``cube``: np.ndarray (3,) cube position
          - ``target``: np.ndarray (3,) target position
          - ``gripper``: list[float] gripper opening
          - ``franka_ee``: tuple(np.ndarray (3,), np.ndarray (4,)) EE position and quaternion
          - ``images``: tuple(rgb, depth) from the RGBD sensor

        Args:
          observation_dict: Mapping from channel name to serialized Ark message.

        Returns:
          dict: Structured observation dictionary as described above.
        """
        cube_state = observation_dict["cube/ground_truth/sim"]
        target_state = observation_dict["target/ground_truth/sim"]
        joint_state = observation_dict["franka/joint_states/sim"]
        ee_state = observation_dict["franka/ee_state/sim"]
        images = observation_dict["IntelRealSense/rgbd/sim"]

        _, cube_position, _, _, _ = unpack.rigid_body_state(cube_state)
        _, target_position, _, _, _ = unpack.rigid_body_state(target_state)
        _, _, franka_joint_position, _, _ = unpack.joint_state(joint_state)
        franka_ee_position, franka_ee_orientation = unpack.pose(ee_state)
        rgb, depth = unpack.rgbd(images)

        gripper_position = franka_joint_position[
            -2
        ]  # Assuming last two joints are gripper

        return {
            "cube": cube_position,
            "target": target_position,
            "gripper": [gripper_position],
            "franka_ee": (franka_ee_position, franka_ee_orientation),
            "images": (rgb, depth),
        }

    def reset_objects(self):
        """Reset key scene objects and internal step counter."""
        self.steps = 0
        self.reset_component("cube")
        self.reset_component("target")
        self.reset_component("franka")

    @staticmethod
    def reward(state, action, next_state):
        """Compute a simple reward signal.


        Args:
          state: Current state.
          action: Applied action.
          next_state: Next state after applying the action.

        Returns:
          Reward value.
        """
        return 0.0

    def step(self, action):
        """Step the environment and increment internal counter.

        Args:
          action: Action to apply to the environment.

        Returns:
          Return ``(obs, reward, terminated, truncated, info)`` from base env.
        """
        self.steps += 1
        time.sleep(self.step_sleep)
        return super().step(action)

    def terminated_truncated_info(self, state, action, next_state):
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
        # Legacy dict path
        if isinstance(state, dict):
            cube_pos = np.asarray(state["cube"], dtype=np.float32)
            target_pos = np.asarray(state["target"], dtype=np.float32)
        else:
            # Flat vector: [cube(3), target(3), gripper(1), ee(3)]
            s = np.asarray(state, dtype=np.float32).reshape(-1)
            cube_pos = s[0:3]
            target_pos = s[3:6]

        distance = np.linalg.norm(cube_pos - target_pos)
        terminated = bool(distance < 0.2)

        if terminated:
            print("Cube is close enough to the target. Terminating episode.")

        truncated = bool(self.steps >= self.max_steps)
        if truncated:
            print("Max steps reached. Terminating episode.")
        truncated = False
        return terminated, truncated, self.steps


def reset_scene(env) -> None:
    """Minimal scene bootstrap: reset objects and exit.

    This script no longer bridges observations/actions. The PolicyNode handles
    that directly. We only ensure the scene is reset and ready.
    """
    env.reset()

    # Reset the policy node state at the beginning of the episode
    try:
        env.send_service_request(
            service_name="Policy/policy/reset",
            request=flag_t(),
            response_type=flag_t,
        )
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
        Expected keys include:
        - sim_config (str): Path to the Ark global sim config.
        - environment_name (str): Environment identifier.
        - algo (DictConfig): Algorithm group with ``name`` and ``model``.
        - n_episodes (int): Number of episodes to evaluate.
        - task_prompt (str, optional): Task string for language policies.

    Returns:
      None. Prints progress and a final success summary to stdout.
    """
    sim_config = "./sim_config/global_config.yaml"
    environment_name = "diffusion_env"

    print("Config:\n", OmegaConf.to_yaml(cfg))

    step_sleep = float(getattr(cfg, "step_sleep", 0.5))
    n_episodes = int(getattr(cfg, "n_episodes", 1))
    max_steps = 100

    chans = default_channels()
    env = RobotEnv(
        config=sim_config,
        environment_name=environment_name,
        action_channels=chans["actions"],
        observation_channels=chans["observations"],
        max_steps=max_steps,
        step_sleep=step_sleep,
        sim=True,
    )
    env.spin()


if __name__ == "__main__":
    main()
