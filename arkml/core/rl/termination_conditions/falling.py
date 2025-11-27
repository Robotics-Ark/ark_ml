from __future__ import annotations

import torch

from ark.utils.scene_status_utils import ObjectState, RobotState
from arkml.core.rl.termination_conditions.base_termination_conditions import (
    FailureCondition,
)
import arkml.utils.transform_utils as T


class Falling(FailureCondition):
    """
    Termination condition that triggers when the robot falls below a height
    threshold or topples beyond a tilt tolerance.
    """

    def __init__(
        self,
        fall_height: float = 0.03,
        topple: bool = True,
        tilt_tolerance: float = 0.75,
        floor_height: float | None = None,
    ):
        """
        Initialize the termination condition.
        Args:
            fall_height: Allowed distance below the floor height before the robot is considered to have fallen.
            topple: If True, also terminate when the robot tips over based on its orientation.
            tilt_tolerance: Minimum acceptable z-component of the robot's up vector. Values below
                            this threshold indicate that the robot has toppled.
            floor_height: If None, the floor height will be inferred from the observation on the
                            first step. Otherwise, the provided height is used.
        """
        self._fall_height = fall_height
        self._topple = topple
        self._tilt_tolerance = tilt_tolerance
        self._floor_height = floor_height
        self._violation_steps = 0

        super().__init__()

    def reset(self) -> None:
        """
        Reset internal state for a new episode.
        Returns:

        """
        super().reset()
        self._violation_steps = 0
        self._floor_height = None

    def _step(self, obs) -> bool:
        """
        Check whether the robot has fallen or toppled.
        Args:
            obs: Current environment observation, expected to contain robot pose information.
        Returns:
            True if a fall or topple event is detected, otherwise False.
        """
        robot = RobotState.from_observation(obs)
        if robot is None:
            return False

        if self._floor_height is None:
            self._floor_height = infer_floor_height(obs, default=0.0)

        robot_z = robot.position[2]
        if robot_z < (self._floor_height - self._fall_height):
            return True

        if self._topple:
            robot_up = T.quat_apply(
                torch.as_tensor(robot.orientation, dtype=torch.float32),
                torch.tensor([0, 0, 1], dtype=torch.float32),
            ).squeeze()
            if robot_up[2] < self._tilt_tolerance:
                return True

        return False


class ObjectFalling(FailureCondition):
    """
    Termination condition that triggers when a specified object (or objects)
    falls, topples, or leaves the workspace for a sustained number of steps.
    """

    def __init__(
        self,
        obj_name: str,
        fall_height: float = 0.03,
        topple: bool = True,
        tilt_tolerance: float = 0.75,
        sustain_steps: int = 5,
        only_when_supported: bool = True,
        ignore_when_grasped: bool = True,
    ):
        self._obj_names = obj_name if type(obj_name) is list else [obj_name]
        self._fall_height = fall_height
        self._topple = topple
        self._tilt_tolerance = tilt_tolerance
        self._sustain_steps = max(1, int(sustain_steps))
        self._only_when_supported = only_when_supported
        self._ignore_when_grasped = ignore_when_grasped
        self._floor_height = None
        self._violation_steps = 0
        super().__init__()

    def reset(self) -> None:
        """Reset the internal state for a new episode."""
        super().reset()
        self._violation_steps = 0
        self._floor_height = None

    def _step(self, obs):
        """
        Check monitored objects for falling or toppling events.
        Args:
            obs: Current environment observation

        Returns:
            True if any monitored object falls or topples for a sustained
            number of steps, otherwise False.
        """
        for obj_name in self._obj_names:
            object_state = ObjectState.from_observation(obs, obj_name)

            if object_state is None:
                continue

            if self._floor_height is None:
                self._floor_height = infer_floor_height(obs, default=0.0)

            # Terminate if the specified object is falling out of the scene
            obj_z = object_state.position[2]
            is_falling = obj_z < (self._floor_height - self._fall_height)

            if not is_falling and self._topple and object_state.orientation is not None:
                obj_up = T.quat_apply(
                    torch.as_tensor(object_state.orientation, dtype=torch.float32),
                    torch.tensor([0, 0, 1], dtype=torch.float32),
                ).squeeze()
                is_falling = obj_up[2] < self._tilt_tolerance

            if is_falling:
                self._violation_steps += 1
            else:
                self._violation_steps = 0

            if self._violation_steps >= self._sustain_steps:
                return True

        return False


def infer_floor_height(obs: dict, default: float = 0.0) -> float:
    """
    Infer the floor height from observation data.
    Args:
        obs: Observation dictionary containing positions of objects or robot parts.
        default: Returned if no suitable position entries are found.

    Returns:
        Estimated floor height.

    """
    candidates = []
    for key, value in obs.items():
        if key.endswith("::position") and hasattr(value, "__len__"):
            z_val = float(value[2])
            candidates.append(z_val)

    if candidates:
        return min(candidates)
    return default
