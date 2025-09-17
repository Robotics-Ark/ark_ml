from typing import Any

import numpy as np
from arktypes.utils import unpack, pack


def observation_unpacking(observation_dict: dict[str:Any], obs_keys: list[str]):
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
      obs_keys: List of keys corresponding to the observations.

    Returns:
      dict: Structured observation dictionary as described above.
    """

    if not observation_dict or any(v is None for v in observation_dict.values()):
        return None
    source_state = observation_dict[obs_keys[0]]
    target_state = observation_dict[obs_keys[1]]
    joint_state = observation_dict[obs_keys[2]]
    ee_state = observation_dict[obs_keys[3]]
    images = observation_dict[obs_keys[4]]
    (
        source_name,
        source_position,
        source_orientation,
        source_lin_velocity,
        source_ang_velocity,
    ) = unpack.rigid_body_state(source_state)

    (
        target_name,
        target_position,
        target_orientation,
        target_lin_velocity,
        target_ang_velocity,
    ) = unpack.rigid_body_state(target_state)

    header, name, position, velocity, effort = unpack.joint_state(joint_state)
    ee_position, ee_orientation = unpack.pose(ee_state)
    rgb, depth = unpack.rgbd(images)

    gripper_position = position[-2]  # Assuming last two joints are gripper

    state = np.concatenate(
        [
            np.asarray(source_position),
            np.asarray(target_position),
            np.asarray([gripper_position]),
            np.asarray(ee_position),
        ]
    )

    return {
        "state": state,
        "images": [rgb],
    }


def action_packing(action: list[float], action_keys: list[str]) -> dict[str, Any]:
    """Pack action into Ark cartesian command message.

    Expects an 8D vector representing end-effector position, orientation
    quaternion, and gripper command in the following order:
    ``[x, y, z, qx, qy, qz, qw, gripper]``.

    Args:
      action: List describing the cartesian command.
      action_keys: List of keys corresponding to the actions.

    Returns:
      dict[str, ...]: Mapping with key pointing to a packed Ark message.
    """

    xyz_command = np.array(action[:3])
    quaternion_command = np.array(action[3:7])
    gripper_command = action[7]

    franka_cartesian_command = pack.task_space_command(
        "all", xyz_command, quaternion_command, gripper_command
    )
    return {action_keys[0]: franka_cartesian_command}
