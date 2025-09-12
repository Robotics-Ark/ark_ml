import hydra
import torch
from arktypes import (
    task_space_command_t,
    pose_t,
    joint_state_t,
    rigid_body_state_t,
    rgbd_t,
)
from omegaconf import DictConfig, OmegaConf

from arkml.examples.franka_pick_place_env import RobotEnv
from arkml.nodes.policy_registry import get_policy_node


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


class RobotNode:
    """Manages env-policy interactions.

    Args:
      env: Environment object.
      policy_node: Policy wrapper.
      obs_horizon: Number of most recent observations to stack for the policy.
    """

    def __init__(self, env, policy_node, obs_horizon=8):
        self.env = env
        self.policy_node = policy_node
        self.obs_horizon = obs_horizon
        self.obs_history = []

    def reset(self):
        """Reset environment and bootstrap observation history.

        Returns:
          ``(obs, info)`` as returned by ``env.reset()``
        """
        obs, info = self.env.reset()
        self.obs_history = [obs] * self.obs_horizon  # bootstrap with repeated obs
        return obs, info

    def step(self, action):
        """Step the environment once and update history.

        Args:
          action: Action to apply to the environment.

        Returns:
          ``(obs, reward, terminated, truncated, info)`` from the env.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_history.append(obs)
        if len(self.obs_history) > self.obs_horizon:
            self.obs_history.pop(0)  # keep window size fixed
        return obs, reward, terminated, truncated, info

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
        terminated = False
        truncated = False

        for _ in range(max_steps):
            model_input = self.get_obs_sequence(task_prompt=task_prompt)
            actions = self.policy_node.predict(model_input)

            # Normalize to a sequence of actions [T, action_dim]
            if actions is None:
                raise RuntimeError("Policy returned None for actions.")
            if actions.ndim == 1:
                actions_seq = actions[None, :]
            else:
                start = obs_horizon + 1
                end = start + action_horizon
                actions_seq = actions[start:end]

            for a in actions_seq:
                _, _, terminated, truncated, _ = self.step(a)
                if step_sleep:
                    import time as _t

                    _t.sleep(step_sleep)
                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        return self.obs_history, terminated, truncated


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

    policy_node = get_policy_node(cfg, device)

    # Robot node to manage history and interaction
    obs_horizon = cfg.algo.model.get("obs_horizon")
    action_horizon = cfg.algo.model.get("action_horizon")
    robot_node = RobotNode(env=env, policy_node=policy_node, obs_horizon=obs_horizon)

    n_episodes = int(getattr(cfg, "n_episodes"))

    success_count = 0
    for ep in range(n_episodes):
        print(f"\n=== Episode {ep} ===")
        # Reset policy state between episodes
        policy_node.reset()

        _, terminated, truncated = robot_node.run_episode(
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
