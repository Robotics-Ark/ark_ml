from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from ark.env.vector_env import make_vector_env
from ark.utils.config_utils import resolve_class
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.registry import ALGOS
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from ark.utils.scene_status_utils import task_space_action_from_obs
from tqdm import tqdm

# Supported SB3 algorithms
_SB3_ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


class TensorboardRewardCallback(BaseCallback):
    """
    Callback to log per-episode rewards to TensorBoard
    """

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_r = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                self.logger.record("rollout/ep_rew", ep_r)
                self.logger.record("rollout/ep_len", ep_len)
        return True


class TrainingProgressCallback(BaseCallback):
    """TQDM-like progress bar for SB3 training."""

    def __init__(self, total_timesteps: int, log_interval: int = 10_000):
        """
        Initialize internals
        Args:
            total_timesteps: Total number of timesteps that training is expected to run.
            log_interval: Interval (in timesteps) at which progress is printed.
        """
        super().__init__()
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self._last = 0
        self._tqdm_bar = None

    def _on_training_start(self) -> None:
        """Initialize the TQDM progress bar if the `tqdm` package is available."""
        self._tqdm_bar = tqdm(total=self.total_timesteps, desc="SB3 training")

    def _on_step(self) -> bool:
        """
        Update the progress bar or log progress at the current training step.
        Returns:
            Always returns True to allow training to continue.
        """
        delta = self.num_timesteps - self._last
        if self._tqdm_bar:
            self._tqdm_bar.update(delta)
        elif delta >= self.log_interval:
            print(f"[SB3] {self.num_timesteps}/{self.total_timesteps} steps")
        self._last = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        """Finalize and close the progress bar after training completes."""
        if self._tqdm_bar:
            remaining = self.total_timesteps - self._last
            if remaining > 0:
                self._tqdm_bar.update(remaining)
            self._tqdm_bar.close()


class ObservationWrapper(gym.vector.VectorEnvWrapper):
    """
    Vectorized observation wrapper for Ark environment.
    """

    def __init__(self, venv: gym.vector.VectorEnv):
        """
        Wrapper initializer.
        Args:
            venv: The vectorized Ark environment to wrap.
        """
        super().__init__(venv)
        base_space = venv.single_observation_space
        if base_space is None or not isinstance(base_space, spaces.Dict):
            raise ValueError("ObservationWrapper requires Dict observation space.")

        rgb_space = base_space.spaces.get("sensors::top_camera::rgb")
        h, w, _ = rgb_space.shape

        proprio_dim = sum(
            np.prod(s.shape)
            for k, s in base_space.spaces.items()
            if k.startswith("proprio")
        )

        single_space = spaces.Dict(
            {
                "rgb": spaces.Box(low=0.0, high=1.0, shape=(3, h, w), dtype=np.float32),
                "proprio": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32
                ),
            }
        )
        self.single_observation_space = single_space
        self.observation_space = batch_space(single_space, n=venv.num_envs)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment and return transformed observations.
        Args:
            seed: Random seed.
            options: Reset parameters.

        Returns:
            tuple:
                - dict: Transformed initial observations in standardized format.
                - dict: Additional reset information from the environment.

        """
        obs, info = self.env.reset(seed=seed, options=options)
        return self._transform(obs), info

    def step(self, actions):
        """
        Take a step in the environment using the provided actions.
        Args:
            actions: Batched actions to forward to the underlying vector env.

        Returns:
            tuple:
                - dict: Transformed observations after stepping.
                - np.ndarray: Step rewards.
                - np.ndarray: Episode termination flags.
                - np.ndarray: Episode truncation flags.
                - dict: Additional step information.

        """
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        return self._transform(obs), rewards, terminated, truncated, info

    @staticmethod
    def _transform(obs: dict[str, Any]) -> dict[str, Any]:
        """
        Convert raw Ark environment observations into standardized multi-input format.
        Args:
            obs: Raw observation dictionary from the vector env.

        Returns:
            dict[str, Any]:
                A dictionary with:
                - ``"rgb"``: Float32 image tensor (N, 3, H, W).
                - ``"proprio"``: Float32 proprioception vector (N, D).

        """
        rgb = np.asarray(obs["sensors::top_camera::rgb"], dtype=np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0
        rgb = np.transpose(rgb, (0, 3, 1, 2))  # NCHW

        proprio = (
            np.concatenate(
                [
                    np.asarray(obs[k], dtype=np.float32).reshape(obs[k].shape[0], -1)
                    for k in obs
                    if k.startswith("proprio")
                ],
                axis=1,
            )
            if any(k.startswith("proprio") for k in obs)
            else np.zeros((len(next(iter(obs.values()))), 0), dtype=np.float32)
        )

        proprio = np.nan_to_num(proprio, nan=0.0, posinf=0.0, neginf=0.0)

        return {"rgb": rgb, "proprio": proprio}

    def observation(self, obs) -> dict[str, Any]:
        """
        Transform a single environment observation (non-batched).
        Args:
            obs: Raw single-environment observation.

        Returns:
            dict[str, Any]:
                Dictionary with:
                - ``"rgb"``: RGB array from the top camera.
                - ``"proprio"``: Flattened proprioceptive vector.

        """
        rgb = obs["sensors::top_camera::rgb"]

        proprio = np.concatenate(
            [obs[k].ravel() for k in obs if k.startswith("proprio")], axis=0
        )
        return {"rgb": rgb, "proprio": proprio}


class ActionSmoothingWrapper(gym.vector.VectorEnvWrapper):
    """
    A vectorized action-smoothing wrapper.
    """

    def __init__(
        self,
        venv: gym.vector.VectorEnv,
        alpha: float = 0.3,
        clip_delta: float | None = None,
        warmup_steps: int = 0,
        warmup_clip_delta: float | None = None,
    ):
        """
        Initialize the action smoother.
        Args:
            venv: The vectorized environment to wrap. Must have a Box action space.
            alpha: Exponential smoothing factor. `alpha=1.0` means no smoothing.
            clip_delta: Maximum allowed per-step change in each action.
            warmup_steps: Number of initial steps per environment where a
                warm-up clipping value is applied.
            warmup_clip_delta:  clip value used during the warm-up period.
        """
        super().__init__(venv)
        if not isinstance(venv.single_action_space, spaces.Box):
            raise ValueError("Action smoothing only supports Box action spaces.")
        self.alpha = float(alpha)
        self.clip_delta = None if clip_delta is None else float(clip_delta)
        self.warmup_steps = int(warmup_steps)
        # Warmup fall back to either the dedicated warmup clip or the default clip
        self.warmup_clip_delta = (
            float(warmup_clip_delta)
            if warmup_clip_delta is not None
            else (self.clip_delta if self.clip_delta is not None else None)
        )
        self._prev_actions = None
        self._warmup_remaining = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reset the underlying vectorized environment and initialize the smoothing state.
        Args:
            seed: RNG seed for the environment reset.
            options: Environment-specific reset options.

        Returns:
            obs: Batched reset observations.
            info: Additional reset information from the environment.

        """
        obs, info = self.env.reset(seed=seed, options=options)
        # Use the robot's reset pose as the initial action baseline
        action_dim = int(np.prod(self.env.single_action_space.shape))
        self._prev_actions = task_space_action_from_obs(
            obs=obs, action_dim=action_dim, num_envs=self.num_envs
        )
        self._warmup_remaining = np.full(self.num_envs, self.warmup_steps, dtype=int)
        return obs, info

    def step(self, actions):
        """
        Apply smoothing to the provided actions and step the environment.
        Args:
            actions: Raw environment actions before smoothing.

        Returns:
            A tuple `(obs, rewards, terminated, truncated, info)` exactly as
            returned by the underlying vectorized environment.

        """
        smoothed = self._apply_filter(actions)
        obs, rewards, terminated, truncated, info = self.env.step(smoothed)
        dones = np.logical_or(terminated, truncated)
        if (
            self._warmup_remaining is None
            or self._warmup_remaining.shape[0] != dones.shape[0]
        ):
            self._warmup_remaining = np.full(
                dones.shape[0], self.warmup_steps, dtype=int
            )
        self._warmup_remaining[dones] = self.warmup_steps
        return obs, rewards, terminated, truncated, info

    def _apply_filter(self, actions: Any):
        """
        Apply EMA smoothing and optional delta clipping to a batch of actions.
        Args:
            actions: A batch of raw actions with shape `[num_envs, ...]`.

        Returns:
            The smoothed and clipped action batch with the same shape as input.
        """
        if (
            self.alpha <= 0.0
            and self.clip_delta is None
            and self.warmup_clip_delta is None
        ):
            return actions

        arr = np.asarray(actions, dtype=np.float32)

        if (
            self._warmup_remaining is None
            or self._warmup_remaining.shape[0] != arr.shape[0]
        ):
            self._warmup_remaining = np.zeros(arr.shape[0], dtype=int)

        prev = np.array(self._prev_actions, copy=True)

        alpha = max(0.0, min(self.alpha, 1.0))
        smoothed = alpha * arr + (1.0 - alpha) * prev

        active_clip = np.array(
            [
                (
                    self.warmup_clip_delta
                    if self._warmup_remaining[i] > 0
                    else self.clip_delta
                )
                for i in range(arr.shape[0])
            ],
            dtype=np.float32,
        )
        for i in range(arr.shape[0]):
            clip_val = active_clip[i]
            if clip_val is None or np.isnan(clip_val):
                continue
            delta = smoothed[i] - prev[i]
            delta = np.clip(delta, -clip_val, clip_val)
            smoothed[i] = prev[i] + delta
            if self._warmup_remaining[i] > 0:
                self._warmup_remaining[i] -= 1

        self._prev_actions = smoothed
        return smoothed


class SB3GymVectorAdapter(VecEnv):
    """
    Adapter to make a Gymnasium VectorEnv compatible with SB3.
    """

    def __init__(self, vec_env: gym.vector.VectorEnv):
        """
        Initialize the SB3 adapter.
        Args:
            vec_env: The underlying Gymnasium VectorEnv.
        """
        obs_space = vec_env.single_observation_space
        act_space = vec_env.single_action_space
        super().__init__(vec_env.num_envs, obs_space, act_space)
        self._vec_env = vec_env
        self._actions = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reset all sub-environments.
        Args:
            seed: Random seed.
            options: Optional reset options.

        Returns:

        """
        obs, info = self._vec_env.reset(seed=seed, options=options)
        return obs

    def step_async(self, actions: Any) -> None:
        """
        Queue actions for the next environment step.
        Args:
            actions: The actions to apply to each environment instance.

        Returns:
            None

        """
        self._actions = actions

    def step_wait(self):
        """
        Execute the environment step using previously provided actions.
        Returns:
            tuple:
                - obs (Any): Batched observations.
                - rewards (np.ndarray): Vector of rewards for each environment.
                - dones (np.ndarray): Boolean flags indicating episode completion.
                - infos (list[dict]): Per-environment info dictionaries.

        """
        obs, rewards, terminated, truncated, infos = self._vec_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)
        # ensure infos is a list
        if isinstance(infos, dict):
            infos = [infos for _ in range(self.num_envs)]
        return obs, rewards, dones, infos

    def close(self) -> None:
        """
        Close the underlying vector environment.
        Returns:
            None

        """
        return self._vec_env.close()

    def render(self, mode: str = "human"):
        """
        Render the environment.
        Args:
            mode: Render mode.

        Returns:
            Any: Rendered frame or display output.

        """
        return getattr(self._vec_env, "render", lambda mode=None: None)(mode)

    # ---- implement all abstract methods ----
    def get_attr(self, attr_name: str, indices: list[int] | None = None):
        """
        Retrieve attribute values from individual sub-environments.
        Args:
            attr_name: Name of the attribute to fetch.
            indices: List of environment indices.

        Returns:
            A list of attribute values from the selected environments.

        """
        envs = getattr(self._vec_env, "envs", None)
        if envs is None:
            raise AttributeError(
                "Underlying VectorEnv does not expose per-env instances."
            )
        idxs = list(range(self.num_envs)) if indices is None else indices
        return [getattr(envs[i], attr_name) for i in idxs]

    def set_attr(
        self, attr_name: str, value: Any, indices: list[int] | None = None
    ) -> None:
        """
        Set attribute values on individual sub-environments.
        Args:
            attr_name: The attribute to modify.
            value: The value to assign to the attribute.
            indices: Indices of environments to modify.

        Returns:
            None

        """
        envs = getattr(self._vec_env, "envs", None)
        if envs is None:
            raise AttributeError(
                "Underlying VectorEnv does not expose per-env instances."
            )
        idxs = list(range(self.num_envs)) if indices is None else indices
        for i in idxs:
            setattr(envs[i], attr_name, value)

    def env_method(
        self, method_name: str, *args, indices: list[int] | None = None, **kwargs
    ):
        """
        Call a method on sub-environments.
        Args:
            method_name: The method name to invoke.
            *args: Positional arguments passed to the method.
            indices: Indices of environments to call.
            **kwargs: Keyword arguments passed to the method.

        Returns:
            Results of the method calls for each environment selected.

        """
        envs = getattr(self._vec_env, "envs", None)
        if envs is None:
            raise AttributeError(
                "Underlying VectorEnv does not expose per-env instances."
            )
        idxs = list(range(self.num_envs)) if indices is None else indices
        return [getattr(envs[i], method_name)(*args, **kwargs) for i in idxs]

    def env_is_wrapped(self, wrapper_class, indices: list[int] | None = None):
        """
        Check whether each selected environment is wrapped with a given wrapper.
        Args:
            wrapper_class: The wrapper class to check for.
            indices: Environment indices.

        Returns:
            Boolean flags indicating wrapper presence per environment.

        """
        envs = getattr(self._vec_env, "envs", None)
        if envs is None:
            return [False] * self.num_envs
        idxs = list(range(self.num_envs)) if indices is None else indices
        return [isinstance(envs[i], wrapper_class) for i in idxs]


def build_vector_env(
    env_cls: type,
    channel_schema: str,
    global_config: str,
    num_envs: int,
    asynchronous: bool,
    sim: bool,
    action_smoothing: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> VecEnv:
    """
    Construct a fully wrapped, SB3-compatible vectorized environment.
    Args:
        env_cls: The environment class to instantiate for each vectorized worker.
        channel_schema: Path to the YAML schema describing observation and action channels.
        global_config: Path to the global configuration file for the simulator or environment.
        num_envs: Number of parallel environments to create.
        asynchronous: Whether to run environments in parallel using `AsyncVectorEnv` `SyncVectorEnv`.
        sim: Whether to launch external simulator processes for each environment.
        action_smoothing: Configuration for action smoothing.
        env_kwargs: Additional kwargs forwarded to the environment constructor.

    Returns:
        A monitored, SB3-compatible vectorized environment ready for training.
    """
    init_kwargs = dict(env_kwargs or {})
    vec_env = make_vector_env(
        env_cls,
        num_envs=num_envs,
        channel_schema=channel_schema,
        global_config=global_config,
        sim=sim,
        asynchronous=asynchronous,
        env_kwargs=init_kwargs,
    )

    if action_smoothing:
        vec_env = ActionSmoothingWrapper(
            vec_env,
            alpha=float(action_smoothing.get("alpha", 0.3)),
            clip_delta=action_smoothing.get("clip_delta"),
            warmup_steps=int(action_smoothing.get("warmup_steps", 0)),
            warmup_clip_delta=action_smoothing.get("warmup_clip_delta"),
        )

    vec_env = ObservationWrapper(vec_env)
    vec_env.reset()
    adapted = SB3GymVectorAdapter(vec_env)
    return VecMonitor(adapted)


@ALGOS.register("sb3rl")
class StableBaselinesRLAlgorithm(BaseAlgorithm):
    """Train Stable-Baselines3 algorithms on Ark vectorized environments."""

    def __init__(self, policy: Any, device: str, cfg: DictConfig):
        """
        Initialize the Stable-Baselines3 algorithm.
        Args:
            policy: Placeholder for API compatibility
            device: Device identifier passed to SB3 (e.g., "cpu", "cuda:0").
            cfg: Configurations
        """
        super().__init__()
        self.device = device
        self.cfg = cfg

        sb3_cfg = cfg.algo.sb3

        algo_name = sb3_cfg.get("algo_name", "ppo").lower()
        algo_cls = _SB3_ALGOS.get(algo_name)
        if algo_cls is None:
            raise ValueError(f"Unsupported SB3 algo '{algo_name}'")

        self._sb3_algo_class = algo_cls
        self._policy_name = sb3_cfg.get("policy", "MultiInputPolicy")
        self._total_timesteps = sb3_cfg.get("total_timesteps", 100_000)
        self._eval_episodes = sb3_cfg.get("eval_episodes", 1)
        self._sb3_kwargs = dict(sb3_cfg.get("kwargs", {}) or {})
        self._sb3_kwargs.setdefault("device", device)
        self._model = None
        self.output_dir = (
            Path(self.cfg.output_dir)
            / "sb3_rl"
            / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        env_cfg = cfg.algo.env
        env_cls = resolve_class(env_cfg.class_path)
        env_kwargs = env_cfg.kwargs
        channel_schema = env_kwargs["channel_schema"]
        global_config = env_kwargs["config_path"]
        asynchronous = env_kwargs.get("asynchronous", False)
        sim = env_kwargs.get("sim", True)
        num_envs = env_cfg.get("num_envs", 2)
        action_smoothing = sb3_cfg.get("action_smoothing", None)
        env_init_kwargs = {
            k: v
            for k, v in env_kwargs.items()
            if k not in {"channel_schema", "config_path", "asynchronous", "sim"}
        }

        self._env = build_vector_env(
            env_cls=env_cls,
            channel_schema=channel_schema,
            global_config=global_config,
            num_envs=num_envs,
            asynchronous=asynchronous,
            sim=sim,
            action_smoothing=action_smoothing,
            env_kwargs=env_init_kwargs,
        )

    def train(self) -> dict[str, Any]:
        """
        Train the configured Stable-Baselines3 algorithm.
        Returns:
            Dictionary containing:
            - "output_dir": Path to the run directory.
            - "model_path": File path for the saved model.
            - "total_timesteps": Number of training timesteps performed.
        """

        model = self._sb3_algo_class(
            policy=self._policy_name,
            env=self._env,
            tensorboard_log=str(self.output_dir / "tensorboard"),
            **self._sb3_kwargs,
        )
        self._model = model

        callbacks = [
            TrainingProgressCallback(total_timesteps=self._total_timesteps),
            TensorboardRewardCallback(),
        ]

        model.learn(total_timesteps=self._total_timesteps, callback=callbacks)
        save_path = self.output_dir / "sb3_model"
        model.save(str(save_path))

        return {
            "output_dir": str(self.output_dir),
            "model_path": f"{save_path}.zip",
            "total_timesteps": self._total_timesteps,
        }

    def eval(self) -> dict[str, Any]:
        """
        Evaluate the trained SB3 model over several episodes.
        Returns:
            Dictionary containing:
            - "n_eval_episodes": Number of episodes evaluated.
            - "episode_returns": Per-environment returns for each evaluation episode.
            - "avg_return": Mean of all collected episode returns.



        """
        if self._model is None:
            candidate = self.output_dir / "sb3_model.zip"
            if not candidate.exists():
                raise RuntimeError("No trained SB3 model found for evaluation.")
            self._model = self._sb3_algo_class.load(str(candidate), env=self._env)

        episode_returns = []

        for _ in range(self._eval_episodes):
            obs, _ = self._env.reset()
            done = np.zeros(self._env.num_envs, dtype=bool)
            ep_ret = np.zeros(self._env.num_envs, dtype=np.float32)

            while not done.all():
                actions, _ = self._model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated, _ = self._env.step(actions)
                done = np.logical_or(terminated, truncated)
                ep_ret += rewards

            episode_returns.extend(ep_ret.tolist())

        avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        return {
            "n_eval_episodes": self._eval_episodes,
            "episode_returns": episode_returns,
            "avg_return": avg_return,
        }


if __name__ == "__main__":
    from arkml.examples.rl.franka_env import FrankaPickPlaceEnv
    from ark.utils.video_recorder import VideoRecorder

    channel_schema = "ark_framework/ark/configs/franka_panda.yaml"
    global_config = (
        "ark_diffusion_policies_on_franka/diffusion_policy/config/global_config.yaml"
    )
    action_smoothing = {
        "alpha": 0.4,
        "clip_delta": 0.05,
        "warmup_steps": 3,
        "warmup_clip_delta": 0.02,
    }

    env = build_vector_env(
        env_cls=FrankaPickPlaceEnv,
        channel_schema=channel_schema,
        global_config=global_config,
        num_envs=1,
        asynchronous=False,
        sim=True,
        action_smoothing=action_smoothing,
    )
    obs = env.reset()

    model = PPO.load("outputs/sb3_rl/2025-11-26-23-14-10/sb3_model.zip")
    rec = VideoRecorder("outputs/sb3_rl/2025-11-26-23-14-10/rollout.mp4", fps=20)

    with tqdm(total=500) as pbar:
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rec.add_frame(obs)
            pbar.update(1)
            if done:
                print("Episode finished")
                break

    rec.close()
    env.close()
