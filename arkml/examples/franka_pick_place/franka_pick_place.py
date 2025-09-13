from typing import Any
import time
import json

import hydra
import torch

from arkml.nodes.policy_registry import get_policy_node
from arktypes import (
    task_space_command_t,
    pose_t,
    joint_state_t,
    rigid_body_state_t,
    rgbd_t,
    string_t,
)
from arktypes.utils import unpack
from omegaconf import DictConfig, OmegaConf

from ark_framework.ark.client.comm_infrastructure.instance_node import InstanceNode
from arkml.examples.franka_pick_place.franka_pick_place_env import RobotEnv


def default_channels() -> dict[str, dict[str, type]]:
    """Return default action/observation channel mappings for simulator.

    Returns:
      Dict[str, dict[str, type]]: A dictionary with two keys:
        - "actions": mapping from action channel name to its type.
        - "observations": mapping from observation channel name to its type.
    """
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


class RobotNode(InstanceNode):
    """Manages env-policy interactions.

    Args:
      env: Environment object.
      policy_node: Policy wrapper.
      obs_horizon: Number of most recent observations to stack for the policy.
    """

    def __init__(self, env, obs_horizon=8):
        super().__init__("Robot")
        self.env = env
        self.obs_horizon = obs_horizon
        self.obs_history = []
        self.truncated = None
        self.terminated = None
        self.next_command = None
        self.create_publisher("observation", string_t)
        self.create_subscriber("next_action", task_space_command_t, self.callback)
        self.create_stepper(10, self.step)

    def reset(self):
        """Reset environment and bootstrap observation history.

        Returns:
          ``(obs, info)`` as returned by ``env.reset()``
        """
        obs, info = self.env.reset()
        self.obs_history = [obs] * self.obs_horizon  # bootstrap with repeated obs
        return obs, info

    def callback(self, t, channel_name, msg):
        # Convert incoming task-space command into 8D action vector
        name, pos, quat, grip = unpack.task_space_command(msg)
        # breakpoint()
        self.next_command = [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
            float(grip),
        ]

    def step(self):
        """Step the environment once and update history.

        Args:
          action: Action to apply to the environment.

        Returns:
          ``(obs, reward, terminated, truncated, info)`` from the env.
        """
        if self.next_command is None:
            return
        obs, reward, self.terminated, self.truncated, info = self.env.step(
            self.next_command
        )
        self.obs_history.append(obs)
        if len(self.obs_history) > self.obs_horizon:
            self.obs_history.pop(0)  # keep window size fixe

    def get_obs_sequence(self, task_prompt: str):
        """Prepare a stacked observation batch for the policy.

        Args:
          task_prompt: Natural language instruction to include in the batch.

        Returns:
          Prepared observation batch, typically containing keys like
          ``image`` (Tensor ``[B,C,H,W]``), ``state`` (Tensor ``[B,D]``), and
          ``task`` (list of length ``B``).
        """
        obs_history = self.obs_history[-self.obs_horizon :]
        return self.prepare_observations(
            observations=obs_history, task_prompt=task_prompt
        )

    @staticmethod
    def prepare_observations(observations: list[dict[str, Any]], task_prompt: str):
        """Convert raw env observations into a batched policy input.

        Args:
          observations: List of observation dicts from the env. Each dict is
            expected to contain keys ``images`` (tuple with RGB as HxWxC),
            ``cube``, ``target``, ``gripper``, and ``franka_ee``.
          task_prompt: Natural language task description to include in the
            batch.

        Returns:
          A batch dictionary with:
            - ``image``: ``torch.FloatTensor`` of shape ``[B, C, H, W]``.
            - ``state``: ``torch.FloatTensor`` of shape ``[B, D]``.
            - ``task``: ``list[str]`` repeated across the batch (length ~= ``B``).
        """
        imgs = []
        states = []
        obs = {}
        for ob in observations:
            # ---- Image ----
            img = ob["images"][0]  # (H, W, C)
            img = torch.from_numpy(img.copy()).permute(2, 0, 1)  # (C, H, W)
            img = img.float() / 255.0
            imgs.append(img)

            # ---- State ----
            state = []
            state.extend(list(ob["cube"]))
            state.extend(list(ob["target"]))
            state.extend(list(ob["gripper"]))
            state.extend(list(ob["franka_ee"][0]))
            states.append(torch.tensor(state, dtype=torch.float32))

        obs["image"] = torch.stack(imgs, dim=0)
        obs["state"] = torch.stack(states, dim=0)
        obs["task"] = [task_prompt] * obs["state"].shape[0]
        return obs

    def run_episode(
        self,
        max_steps: int,
        obs_horizon: int,
        action_horizon: int,
        step_sleep: float,
        task_prompt: str,
    ):
        """Run one episode using the current policy and environment.

        Args:
          max_steps: Maximum number of env steps to execute.
          obs_horizon: TODO
          action_horizon: Number of actions to apply from a predicted sequence.
          step_sleep: Optional delay (seconds) between executed actions.
          start_offset: Index into the predicted sequence to start from.
          task_prompt: Natural language instruction to include in observations.

        Returns:
          ``(obs_history, terminated, truncated)`` where
          ``obs_history`` is the internal observation buffer, and the booleans
          indicate episode termination status.
        """
        _, _ = self.reset()
        for _ in range(max_steps):
            # Prepare current observation batch for the policy
            model_input = self.get_obs_sequence(task_prompt=task_prompt)

            # Serialize observation to JSON (lists) and publish on generic channel
            payload = {
                "image": (
                    model_input.get("image").tolist()
                    if "image" in model_input
                    else None
                ),
                "state": (
                    model_input.get("state").tolist()
                    if "state" in model_input
                    else None
                ),
                "task": model_input.get("task", []),
            }
            obs_msg = string_t()
            obs_msg.data = json.dumps(payload)
            self.obs_pub.publish(obs_msg)
            # breakpoint()

            # Stepper advances env asynchronously via next_action
            if self.truncated or self.terminated:
                break
            time.sleep(step_sleep if step_sleep is not None else 0.1)

        return self.terminated, self.truncated


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

    # Environment
    chans = default_channels()

    env: RobotEnv = RobotEnv(
        config=sim_config,
        environment_name=environment_name,
        action_channels=chans["actions"],
        observation_channels=chans["observations"],
        max_steps=int(getattr(cfg, "max_steps")),
        sim=True,
    )

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Robot node to manage history and interaction
    obs_horizon = cfg.algo.model.get("obs_horizon")
    action_horizon = cfg.algo.model.get("action_horizon")
    robot_node = RobotNode(env=env, obs_horizon=obs_horizon)

    n_episodes = int(getattr(cfg, "n_episodes"))

    success_count = 0
    for ep in range(n_episodes):
        print(f"\n=== Episode {ep} ===")
        # Reset policy state between episodes

        terminated, truncated = robot_node.run_episode(
            max_steps=int(getattr(cfg, "max_steps")),
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            step_sleep=float(getattr(cfg, "step_sleep", 0.0)),
            task_prompt=cfg.task_prompt,
        )
        if terminated:
            success_count += 1
            print(f"SUCCESS: Episode {ep}")
        else:
            print(f"FAILED: Episode {ep}")

    print(f"\nTotal successful episodes: {success_count}/{n_episodes}")


if __name__ == "__main__":
    main()
