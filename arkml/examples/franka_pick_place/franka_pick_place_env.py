from typing import Any

import numpy as np
from ark.env.ark_env import ArkEnv
from arktypes.utils import pack, unpack
from omegaconf import DictConfig


class RobotEnv(ArkEnv):
    """ARK environment wrapper customized for Franka + cube simulation.

    Args:
      config: Global ARK configuration object.
      environment_name: Name/identifier of the ARK environment to load.
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

    @staticmethod
    def action_packing(action: list) -> dict[str, Any]:
        """Pack action into ARK cartesian command message.

        Expects an 8D vector representing end-effector position, orientation
        quaternion, and gripper command in the following order:
        ``[x, y, z, qx, qy, qz, qw, gripper]``.

        Args:
          action: List describing the cartesian command.

        Returns:
          dict[str, ...]: Mapping with key pointing to a packed ARK message.
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
        """Unpack raw ARK observations into structured components.

        Converts incoming channel messages into a dictionary with primitive
        types useful for policies.

        Returns a dictionary with keys:
          - ``cube``: np.ndarray (3,) cube position
          - ``target``: np.ndarray (3,) target position
          - ``gripper``: list[float] gripper opening
          - ``franka_ee``: tuple(np.ndarray (3,), np.ndarray (4,)) EE position and quaternion
          - ``images``: tuple(rgb, depth) from the RGBD sensor

        Args:
          observation_dict: Mapping from channel name to serialized ARK message.

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
        terminated = bool(distance < 0.1)

        if terminated:
            print("Cube is close enough to the target. Terminating episode.")

        truncated = bool(self.steps >= self.max_steps)
        if truncated:
            print("Max steps reached. Terminating episode.")

        return terminated, truncated, self.steps
