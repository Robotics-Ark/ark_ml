from __future__ import annotations

import torch

from ark.utils.scene_status_utils import ObjectState, RobotState
from arkml.core.rl.termination_conditions.base_termination_conditions import (
    FailureCondition,
)
import arkml.utils.transform_utils as T


class Falling(FailureCondition):
    """
    Falling (failure condition) used for any navigation-type tasks
    Episode terminates if the robot falls out of the world (i.e.: falls below the floor height by at least
    @fall_height

    Args:
        robot_idn (int): robot identifier to evaluate condition with. Default is 0, corresponding to the first
            robot added to the scene
        fall_height (float): distance (m) > 0 below the scene's floor height under which the the robot is considered
            to be falling out of the world
        topple (bool): whether to also consider the robot to be falling if it is toppling over (i.e.: if it is
            no longer upright
    """

    def __init__(
        self,
        fall_height: float = 0.03,
        topple: bool = True,
        tilt_tolerance: float = 0.75,
        floor_height: float | None = None,
    ):
        # Store internal vars
        self._fall_height = fall_height
        self._topple = topple
        self._tilt_tolerance = tilt_tolerance
        self._floor_height = floor_height
        self._violation_steps = 0

        # Run super init
        super().__init__()

    def reset(self):
        super().reset()
        self._violation_steps = 0
        # Recompute on next step to avoid carrying stale data between episodes
        self._floor_height = None

    def _step(self, obs, action):
        robot = RobotState.from_observation(obs)
        if robot is None:
            # Without pose information we cannot assess falling; stay alive.
            return False

        if self._floor_height is None:
            self._floor_height = infer_floor_height(obs, default=0.0)

        robot_z = robot.position[2]
        if robot_z < (self._floor_height - self._fall_height):
            return True

        # Terminate if the robot has toppled over
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
    Object falling (failure condition) for manipulation-type tasks.
    Episode terminates if the specified object falls below the floor height
    by at least @fall_height.

    Args:
        obj_name (str): Name of the target object in the scene registry.
        fall_height (float): Distance (m) > 0 below the scene's floor height
            under which the object is considered to have fallen out of the world.
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

    def reset(self):
        super().reset()
        self._violation_steps = 0
        self._floor_height = None

    def _step(self, obs, action):
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
    Estimate a floor height from available position observations.
    Uses the minimum z found to approximate the support surface.
    """
    candidates = []
    for key, value in obs.items():
        if key.endswith("::position") and hasattr(value, "__len__"):
            try:
                z_val = float(value[2])
                candidates.append(z_val)
            except Exception:
                continue
    if candidates:
        return min(candidates)
    return default
