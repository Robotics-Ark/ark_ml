from __future__ import annotations

from typing import Any

import numpy as np
from ark.env.ark_env import ArkEnv
from ark.utils.scene_status_utils import ObjectState, RobotState
from arkml.core.rl.reward_functions.point_goal_reward import PointGoalReward
from arkml.core.rl.reward_functions.potential_reward import PotentialReward
from arkml.core.rl.termination_conditions.base_termination_conditions import (
    SuccessCondition,
)
from arkml.core.rl.termination_conditions.falling import ObjectFalling
from arkml.core.rl.termination_conditions.timeout import Timeout


class PointGoalTermination(SuccessCondition):
    """
    Success condition triggered when the robot's end-effector reaches the cube.
    """

    def __init__(self, distance_tol: float = 1.0, distance_axes: str = "xyz"):
        super().__init__()
        self._distance_tol = distance_tol
        self._distance_axes = distance_axes

    def _step(self, obs):
        robot = RobotState.from_observation(obs)
        cube = ObjectState.from_observation(obs, "cube")

        if robot is None or cube is None:
            return False

        axes = _axes_indices(self._distance_axes)
        eef_pos = robot.position[axes]
        cube_pos = cube.position[axes]

        dist = float(np.linalg.norm(eef_pos - cube_pos))
        return dist <= self._distance_tol


class FrankaPickPlaceEnv(ArkEnv):
    def __init__(
        self,
        channel_schema,
        global_config,
        namespace: str,
        sim: bool = True,
        max_steps: int = 500,
    ):
        # Default reward configuration; can be overwritten externally if needed
        self._reward_config = {"r_potential": 0.1, "r_pointgoal": 10.0}
        self._max_steps = max_steps
        super().__init__(
            environment_name="rl_franka",
            channel_schema=channel_schema,
            global_config=global_config,
            namespace=namespace,
            sim=sim,
        )

    def reset_objects(self):
        self.reset_component("cube")
        self.reset_component("target")
        self.reset_component("franka")

    def _create_termination_conditions(self):
        terminations: dict[str, Any] = {}

        terminations["falling"] = ObjectFalling(obj_name="cube", fall_height=0.3)
        terminations["pointgoal"] = PointGoalTermination(
            distance_tol=0.05,
            distance_axes="xy",
        )
        terminations["timeout"] = Timeout(max_steps=self._max_steps)

        return terminations

    def _create_reward_functions(self):
        rewards: dict[str, Any] = {}

        rewards["potential"] = PotentialReward(
            potential_fcn=self.get_potential,
            r_potential=self._reward_config["r_potential"],
        )
        rewards["pointgoal"] = PointGoalReward(
            pointgoal=self._termination_conditions["pointgoal"],
            r_pointgoal=self._reward_config["r_pointgoal"],
        )
        return rewards

    def get_potential(self, obs: dict[str, Any]) -> float:
        """
        Potential based on the distance between end-effector and cube.
        Lower potential is better, so we simply return Euclidean distance.
        """
        cube = ObjectState.from_observation(obs, "cube")
        robot = RobotState.from_observation(obs)

        if cube is None or robot is None:
            return 0.0

        return float(np.linalg.norm(robot.position - cube.position))


def _axes_indices(axes: str) -> list[int]:
    """
    Map an axis string like "xyz" or "xy" to position indices.
    """
    mapping = {"x": 0, "y": 1, "z": 2}
    return [mapping[a] for a in axes if a in mapping]
