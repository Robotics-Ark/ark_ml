import argparse
import time

import numpy as np
from ark.env.ark_env import ArkEnv
from ark.tools.log import log
from ark.utils.scene_status_utils import ObjectState, RobotState
from ark.utils.utils import ConfigPath
from arkml.core.rl.termination_conditions.base_termination_conditions import (
    SuccessCondition,
)
from arkml.core.rl.termination_conditions.timeout import Timeout
from arktypes import flag_t, string_t
from tqdm import tqdm


def _axes_indices(axes: str) -> list[int]:
    """
    Map an axis string like "xyz" or "xy" to position indices.
    """
    mapping = {"x": 0, "y": 1, "z": 2}
    return [mapping[a] for a in axes if a in mapping]


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


class RobotNode(ArkEnv):

    def __init__(self, max_steps: int, config_path: str, channel_schema: str):

        self.max_steps = max_steps
        self.config = ConfigPath(config_path).read_yaml()
        self.mode = self.config.get("policy_mode")

        super().__init__(
            environment_name="ark_franka_env",
            channel_schema=channel_schema,
            global_config=config_path,
            sim=True,
        )

    def _create_termination_conditions(self):
        terminations = dict()
        terminations["pointgoal"] = PointGoalTermination(
            distance_tol=0.1,
            distance_axes="xy",
        )
        terminations["timeout"] = Timeout(max_steps=self.max_steps)

        return {}

    def _create_reward_functions(self):

        return {}

    def reset_objects(self):
        """Reset key scene objects and internal step counter."""
        self.steps = 0
        self.reset_component("cube")
        self.reset_component("target")
        self.reset_component("franka")

    def reset_service(self) -> None:
        """Minimal scene bootstrap: reset objects and exit.

        This script no longer bridges observations/actions. The PolicyNode handles
        that directly. We only ensure the scene is reset and ready.
        """

        # Reset the policy node state at the beginning of the episode
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

    def terminated_truncated_info(self, state):
        cube_pos = np.array(state["objects::cube::position"])
        target_pos = np.array(state["objects::target::position"])

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


def stepper_based_episode(
    robo_env: RobotNode,
    n_episodes: int,
    max_step: int,
    policy_node: str,
    step_sleep: int,
):
    success_count = 0
    failure_count = 0

    for ep in tqdm(range(n_episodes), desc="Episodes", unit="ep"):
        robo_env.reset_service()
        success = False
        time.sleep(5)
        robo_env.send_service_request(
            service_name=f"{policy_node}/policy/start",
            request=flag_t(),
            response_type=flag_t,
        )
        for step_count in tqdm(
            range(max_step), desc=f"Ep {ep}", unit="step", leave=False
        ):
            obs = robo_env.ark_observation_space.get_observation()
            if obs is None:
                print("none")
                continue

            terminated, truncated, info = robo_env.terminated_truncated_info(obs)

            if terminated or truncated:
                success = True
                break

            step_count += 1
            time.sleep(step_sleep)
            if terminated or truncated:
                success = True
                break

            step_count += 1
            time.sleep(step_sleep)

        robo_env.send_service_request(
            service_name=f"{policy_node}/policy/stop",
            request=flag_t(),
            response_type=flag_t,
        )
        if success:
            success_count += 1
            print("EPISODE SUCCESS")
        else:
            failure_count += 1
            print("EPISODE FAILED")

    return success_count, failure_count


def service_based_episode(
    robo_env: RobotNode,
    n_episodes: int,
    max_step: int,
    policy_node: str,
    step_sleep: int,
):
    success_count = 0
    failure_count = 0

    for ep in tqdm(range(n_episodes), desc="Episodes", unit="ep"):
        robo_env.reset_service()
        success = False
        time.sleep(5)
        for step_count in tqdm(
            range(max_step), desc=f"Ep {ep}", unit="step", leave=False
        ):
            robo_env.send_service_request(
                service_name=f"{policy_node}/policy/predict",
                request=flag_t(),
                response_type=flag_t,
            )

            obs = robo_env.ark_observation_space.get_observation()
            if obs is None:
                continue
            terminated, truncated, info = robo_env.terminated_truncated_info(obs)

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

    return success_count, failure_count


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        Parsed arguments.
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
        default=1,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=500,
        help="Maximum number of steps per episode (default: 500)",
    )
    parser.add_argument(
        "--policy_node_name",
        type=str,
        default="Policy",
        help="Policy node name",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="ark_ml/arkml/examples/franka_pick_place/franka_config/global_config.yaml",
        help="Global config path",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="ark_framework/ark/configs/franka_panda.yaml",
        help="Global config path",
    )

    return parser.parse_args()


def main() -> None:
    """Run rollouts for a configured policy.

    Returns:
      None.
      Prints progress and a final success summary to stdout.
    """
    args = parse_args()
    step_sleep = args.step_sleep
    n_episodes = args.n_episodes
    max_step = args.max_step
    policy_node = args.policy_node_name
    config_path = args.config_path
    channel_schema = args.env_config_path

    robo_env = RobotNode(
        max_steps=max_step, config_path=config_path, channel_schema=channel_schema
    )
    policy_mode = robo_env.mode
    print(f"Policy Mode : {policy_mode}")

    if policy_mode == "service":
        success_count, failure_count = service_based_episode(
            robo_env=robo_env,
            n_episodes=n_episodes,
            max_step=max_step,
            policy_node=policy_node,
            step_sleep=step_sleep,
        )
    elif policy_mode == "stepper":
        success_count, failure_count = stepper_based_episode(
            robo_env=robo_env,
            n_episodes=n_episodes,
            max_step=max_step,
            policy_node=policy_node,
            step_sleep=step_sleep,
        )
    else:
        raise ValueError(f"Unknown policy mode: {policy_mode}")

    # After all episodes
    success_rate = success_count / n_episodes * 100

    print("\n=== Results ===")
    print(f"Total Episodes : {n_episodes}")
    print(f"Successes      : {success_count}")
    print(f"Failures       : {failure_count}")
    print(f"Success Rate   : {success_rate:.2f}%")


if __name__ == "__main__":
    main()
