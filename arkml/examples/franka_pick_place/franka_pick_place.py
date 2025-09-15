import time
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from arkml.examples.franka_pick_place.franka_pick_place_env import RobotEnv
from arktypes import (
    task_space_command_t,
    pose_t,
    joint_state_t,
    rigid_body_state_t,
    rgbd_t,
)
from arktypes import flag_t


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


def reset_scene(
    sim_config: str, environment_name: str, max_steps: int, step_sleep: float
) -> None:
    """Minimal scene bootstrap: reset objects and exit.

    This script no longer bridges observations/actions. The PolicyNode handles
    that directly. We only ensure the scene is reset and ready.
    """
    chans = default_channels()
    env = RobotEnv(
        config=sim_config,
        environment_name=environment_name,
        action_channels=chans["actions"],
        observation_channels=chans["observations"],
        max_steps=max_steps,
        step_sleep=step_sleep,
        sim=True,
    )
    env.reset()

    # Reset the policy node state at the beginning of the episode
    try:
        env.send_service_request(
            service_name="Policy/policy/reset",
            request=flag_t(),
            response_type=flag_t,
        )
    except Exception as e:
        print(f"Warning: Failed to reset policy via service: {e}")
    # Give subsystems a moment to settle
    time.sleep(1.0)


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

    step_sleep = float(getattr(cfg, "step_sleep", 0.5))
    n_episodes = int(getattr(cfg, "n_episodes", 1))

    for ep in range(n_episodes):
        print(f"\n=== Episode {ep+1}/{n_episodes} ===")
        reset_scene(
            sim_config=sim_config,
            environment_name=environment_name,
            max_steps=int(getattr(cfg, "max_steps")) * 2,
            step_sleep=step_sleep,
        )
        # Optional pause between episodes
        time.sleep(step_sleep)


if __name__ == "__main__":
    main()
