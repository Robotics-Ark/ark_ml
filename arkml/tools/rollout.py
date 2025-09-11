import hydra
import torch
from arktypes import task_space_command_t, pose_t, joint_state_t, rigid_body_state_t, rgbd_t
from omegaconf import DictConfig, OmegaConf

from arkml.nodes.env import RobotEnv
from arkml.nodes.policy_registry import get_policy_node
from arkml.nodes.robot_node import RobotNode


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


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run rollouts for a configured policy.

    Args:
      cfg: Hydra configuration composed of ``defaults.yaml`` and overrides.
        Expected keys include:
        - sim_config (str): Path to the ARK global sim config.
        - environment_name (str): Environment identifier.
        - algo (DictConfig): Algorithm group with ``name`` and ``model``.
        - n_episodes (int): Number of episodes to evaluate.
        - task_prompt (str, optional): Task string for language policies.

    Returns:
      None. Prints progress and a final success summary to stdout.
    """
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # Environment
    chans = default_channels()  # TODO cfg.channels
    env: RobotEnv = RobotEnv(
        config=getattr(cfg, "sim_config"),
        environment_name=getattr(cfg, "environment_name"),
        action_channels=chans["actions"],
        observation_channels=chans["observations"],
        max_steps=cfg.max_steps,
        sim=True,
    )

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_node = get_policy_node(cfg, device)


    # Robot node to manage history and interaction
    action_horizon = cfg.algo.model.get("action_horizon")
    obs_horizon=1
    robot_node = RobotNode(env=env, policy_node=policy_node, obs_horizon=1)

    n_episodes = int(getattr(cfg, "n_episodes"))

    success_count = 0
    for ep in range(n_episodes):
        print(f"\n=== Episode {ep} ===")
        # Reset policy state between episodes
        policy_node.reset()

        # For diffusion policies, offset start to ensure history is filled
        slice_start = (obs_horizon + 1) if cfg.algo.name == "diffusion_policy" else 0  # TODO: verify
        _, terminated, truncated = robot_node.run_episode(
            max_steps=int(getattr(cfg, "max_steps", 500)),
            action_horizon=action_horizon,
            step_sleep=float(getattr(cfg, "step_sleep", 0.0)),
            start_offset=int(getattr(cfg, "start_offset", slice_start)),
            task_prompt=cfg.task_prompt
        )
        if terminated:
            success_count += 1
            print(f"SUCCESS: Episode {ep}")
        else:
            print(f"FAILED: Episode {ep}")

    print(f"\nTotal successful episodes: {success_count}/{n_episodes}")


if __name__ == "__main__":
    # Example overrides:
    # python tools/rollout.py checkpoint=/path/to/weights.pth algo=diffusion data=diffusion_dataset \
    #   algo.model.obs_horizon=8 algo.model.pred_horizon=16 algo.model.action_horizon=8
    main()
