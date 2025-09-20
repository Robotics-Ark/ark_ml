import argparse
import time
from typing import Any

import numpy as np
from ark.env.ark_env import ArkEnv
from ark.tools.log import log
from arktypes import flag_t, string_t
from arktypes import (
    task_space_command_t,
    pose_t,
    joint_state_t,
    rigid_body_state_t,
    rgbd_t,
)
from arktypes.utils import pack, unpack
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


import numpy as np


def convert_ndarray_to_list(obj):
    """
    Recursively convert any ndarray in a dict, list, or tuple to a list.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray_to_list(v) for v in obj)
    else:
        return obj


class RobotNode(ArkEnv):

    def __init__(self, max_steps):
        config = (
            "/Users/abhineetkumar/arkprojects/ark_diffusion_policies_on_franka/diffusion_policy/config/global_config.yaml"
        )
        chans = default_channels()
        super().__init__(
            environment_name="diffusion_env",
            action_channels=chans["actions"],
            observation_channels=chans["observations"],
            global_config=config,
            sim=True,
        )

        self.max_steps = max_steps

    def action_packing(self, action):
        """
        Packs the action into a joint_group_command_t format.

        List of:
        [EE X, EE_Y, EE_Z, EE_Quaternion_X, EE_Quaternion_Y, EE_Quaternion_Z, EE_Quaternion_W, Gripper]
        """
        xyz_command = np.array(action[:3])
        quaternion_command = np.array(action[3:7])
        gripper_command = action[7]

        franka_cartesian_command = pack.task_space_command(
            "all", xyz_command, quaternion_command, gripper_command
        )
        return {"franka/cartesian_command/sim": franka_cartesian_command}

    def observation_unpacking(self, observation_dict):
        """
        Unpacks the observation from the environment.

        Returns a dictionary with keys
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
            "position": [franka_joint_position],
            "franka_ee": (franka_ee_position, franka_ee_orientation),
            "images": (rgb, depth),
        }

    def terminated_truncated_info(self, state, action, next_state):
        cube_pos = np.array(state["cube"])
        target_pos = np.array(state["target"])

        # Terminate if cube is within 5 cm of target
        distance = np.linalg.norm(cube_pos - target_pos)
        terminated = distance < 0.1

        if terminated:
            print("Cube is close enough to the target. Terminating episode.")

        if self.steps >= self.max_steps:
            print("Max steps reached. Terminating episode.")
            truncated = True
        else:
            truncated = False

        return terminated, truncated, self.steps

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Compute the reward for a transition"""
        ...

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
        observation = None
        try:
            log.info("Resetting Policy Node ...")
            self.send_service_request(
                service_name="Policy/policy/reset",
                request=flag_t(),
                response_type=string_t,
            )
            log.info("Resetting Environment ...")
            observation, info = self.reset()

        except Exception as e:
            print(f"Warning: Failed to reset policy via service: {e}")
        # Give subsystems a moment to settle
        time.sleep(1.0)

def main() -> None:
    """Run rollouts for a configured policy.

    Args:
      cfg: Hydra configuration composed of ``defaults.yaml`` and overrides.

    Returns:
      None. Prints progress and a final success summary to stdout.
    """

    parser = argparse.ArgumentParser(
        description="Run rollouts for a configured policy."
    )
    parser.add_argument(
        "--step_sleep",
        type=float,
        default=0.1,
        help="Sleep time between steps (default: 0.1s)",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=500,
        help="Maximum number of steps per episode (default: 500)",
    )

    args = parser.parse_args()

    step_sleep = args.step_sleep
    n_episodes = args.n_episodes
    max_step = args.max_step

    robo_env = RobotNode(max_steps=max_step)

    success_count = 0
    failure_count = 0

    for ep in tqdm(range(n_episodes), desc="Episodes", unit="ep"):
        robo_env.reset_scene()
        success = False
        time.sleep(5)
        robo_env.observation_space.wait_until_observation_space_is_ready()
        for step_count in tqdm(
            range(max_step), desc=f"Ep {ep}", unit="step", leave=False
        ):
            response = robo_env.send_service_request(
                service_name="Policy/policy/predict",
                request=flag_t(),
                response_type=string_t,
            )
            if response is None:
                continue

            action = np.array(json.loads(response.data), dtype=np.float32)

            observation, reward, terminated, truncated, info = robo_env.step(action)

            if terminated or truncated:
                success = True
                break

            step_count += 1
            time.sleep(step_sleep)

        if success:
            success_count += 1
            print("EPISODE SUCCESS")
        else:
            failure_count += 1
            print("EPISODE FAILED")

    # After all episodes
    success_rate = success_count / n_episodes * 100

    print("\n=== Results ===")
    print(f"Total Episodes : {n_episodes}")
    print(f"Successes      : {success_count}")
    print(f"Failures       : {failure_count}")
    print(f"Success Rate   : {success_rate:.2f}%")


if __name__ == "__main__":
    main()
