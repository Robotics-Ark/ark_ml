import math

import torch

from arkml.core.rl.reward_functions.base_reward_function import (
    BaseRewardFunction,
)
from ark.utils.scene_status_utils import ObjectState, RobotState

import arkml.utils.transform_utils as T


class GraspReward(BaseRewardFunction):
    """
    A composite reward function for grasping tasks. This reward function not only evaluates the success of object grasping
    but also considers various penalties and efficiencies.

    The reward is calculated based on several factors:
    - Grasping reward: A positive reward is given if the robot is currently grasping the specified object.
    - Distance reward: A reward based on the inverse exponential distance between the end-effector and the object.
    - Regularization penalty: Penalizes large magnitude actions to encourage smoother and more energy-efficient movements.
    - Position and orientation penalties: Discourages excessive movement of the end-effector.
    - Collision penalty: Penalizes collisions with the environment or other objects.

    Attributes:
        obj_name (str): Name of the object to grasp.
        dist_coeff (float): Coefficient for the distance reward calculation.
        grasp_reward (float): Reward given for successfully grasping the object.
        collision_penalty (float): Penalty incurred for any collision.
        eef_position_penalty_coef (float): Coefficient for the penalty based on end-effector's position change.
        eef_orientation_penalty_coef (float): Coefficient for the penalty based on end-effector's orientation change.
        regularization_coef (float): Coefficient for penalizing large actions.
    """

    def __init__(
        self,
        obj_name,
        dist_coeff,
        grasp_reward,
        collision_penalty,
        eef_position_penalty_coef,
        eef_orientation_penalty_coef,
        regularization_coef,
        grasp_distance_tol: float = 0.03,
    ):
        # Store internal vars
        self.prev_grasping = False
        self.prev_eef_pos = None
        self.prev_eef_quat = None
        self.obj_name = obj_name
        self.obj = None
        self.dist_coeff = dist_coeff
        self.grasp_reward = grasp_reward
        self.collision_penalty = collision_penalty
        self.eef_position_penalty_coef = eef_position_penalty_coef
        self.eef_orientation_penalty_coef = eef_orientation_penalty_coef
        self.regularization_coef = regularization_coef
        self.grasp_distance_tol = grasp_distance_tol

        # Run super
        super().__init__()

    def _step(self, obs, action):
        obj_state = ObjectState.from_observation(obs, self.obj_name)
        robot_state = RobotState.from_observation(obs)
        if obj_state is None or robot_state is None:
            return 0.0, {"grasp_success": False}

        eef_pos = torch.as_tensor(robot_state.position, dtype=torch.float32)
        eef_quat = torch.as_tensor(robot_state.orientation, dtype=torch.float32)
        obj_center = torch.as_tensor(obj_state.position, dtype=torch.float32)

        dist_to_obj = T.l2_distance(eef_pos, obj_center)
        current_grasping = dist_to_obj < self.grasp_distance_tol

        info = {"grasp_success": current_grasping}

        # Reward varying based on combination of whether the robot was previously grasping the desired object
        # and is currently grasping the desired object
        reward = 0.0

        # Penalize large actions
        action_tensor = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
        action_mag = torch.sum(torch.abs(action_tensor))
        action_mag_val = float(action_mag)
        regularization_penalty = -(action_mag_val * self.regularization_coef)
        reward += regularization_penalty
        info["regularization_penalty_factor"] = action_mag_val
        info["regularization_penalty"] = regularization_penalty

        # Penalize based on the magnitude of the action
        info["position_penalty_factor"] = 0.0
        info["position_penalty"] = 0.0
        if self.prev_eef_pos is not None:
            eef_pos_delta = float(T.l2_distance(self.prev_eef_pos, eef_pos))
            position_penalty = -eef_pos_delta * self.eef_position_penalty_coef
            reward += position_penalty
            info["position_penalty_factor"] = eef_pos_delta
            info["position_penalty"] = position_penalty
        self.prev_eef_pos = eef_pos

        info["rotation_penalty_factor"] = 0.0
        info["rotation_penalty"] = 0.0
        if self.prev_eef_quat is not None:
            delta_rot = float(
                T.get_orientation_diff_in_radian(self.prev_eef_quat, eef_quat)
            )
            rotation_penalty = -delta_rot * self.eef_orientation_penalty_coef
            reward += rotation_penalty
            info["rotation_penalty_factor"] = delta_rot
            info["rotation_penalty"] = rotation_penalty
        self.prev_eef_quat = eef_quat

        # Penalize robot for colliding with an object
        info["collision_penalty_factor"] = 0.0
        info["collision_penalty"] = 0.0
        # if detect_robot_collision_in_sim(robot, filter_objs=[self.obj]):
        #     reward += -self.collision_penalty
        #     info["collision_penalty_factor"] = 1.0
        #     info["collision_penalty"] = -self.collision_penalty

        # If we're not currently grasping
        info["grasp_reward_factor"] = 0.0
        info["grasp_reward"] = 0.0
        info["pregrasp_dist"] = 0.0
        info["pregrasp_dist_reward_factor"] = 0.0
        info["pregrasp_dist_reward"] = 0.0
        info["postgrasp_dist"] = 0.0
        info["postgrasp_dist_reward_factor"] = 0.0
        info["postgrasp_dist_reward"] = 0.0
        if not current_grasping:
            # TODO: If we dropped the object recently, penalize for that
            dist = float(dist_to_obj)
            dist_reward = math.exp(-dist) * self.dist_coeff
            reward += dist_reward
            info["pregrasp_dist"] = dist
            info["pregrasp_dist_reward_factor"] = math.exp(-dist)
            info["pregrasp_dist_reward"] = dist_reward
        else:
            # We are currently grasping - first apply a grasp reward
            reward += self.grasp_reward
            info["grasp_reward_factor"] = 1.0
            info["grasp_reward"] = self.grasp_reward

            # Then apply a distance reward to take us to a tucked position
            dist = float(dist_to_obj)
            dist_reward = math.exp(-dist) * self.dist_coeff
            reward += dist_reward
            info["postgrasp_dist"] = dist
            info["postgrasp_dist_reward_factor"] = math.exp(-dist)
            info["postgrasp_dist_reward"] = dist_reward

        self.prev_grasping = current_grasping

        return reward, info

    def reset(self, initial_obs=None):
        """
        Reward function-specific reset

        Args:
            initial_obs (dict | None): Optional initial observation
        """
        super().reset(initial_obs=initial_obs)
        self.prev_grasping = False
        self.prev_eef_pos = None
        self.prev_eef_quat = None
