from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
    when using custom vectorized adapters.
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
        super().__init__()
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self._last = 0
        self._tqdm_bar = None

    def _on_training_start(self) -> None:
        try:
            from tqdm import tqdm

            self._tqdm_bar = tqdm(total=self.total_timesteps, desc="SB3 training")
        except ImportError:
            self._tqdm_bar = None

    def _on_step(self) -> bool:
        delta = self.num_timesteps - self._last
        if self._tqdm_bar:
            self._tqdm_bar.update(delta)
        elif delta >= self.log_interval:
            print(f"[SB3] {self.num_timesteps}/{self.total_timesteps} steps")
        self._last = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self._tqdm_bar:
            remaining = self.total_timesteps - self._last
            if remaining > 0:
                self._tqdm_bar.update(remaining)
            self._tqdm_bar.close()


class ObservationWrapper(gym.vector.VectorEnvWrapper):
    """
    Vectorized observation wrapper: converts Ark dict observations
    to {"rgb": (N,3,H,W), "proprio": (N,D)} format.
    """

    def __init__(self, venv: gym.vector.VectorEnv):
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._transform(obs), info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        return self._transform(obs), rewards, terminated, truncated, info

    @staticmethod
    def _transform(obs: dict[str, Any]) -> dict[str, Any]:
        rgb = np.asarray(obs["sensors::top_camera::rgb"], dtype=np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0
        rgb = np.transpose(rgb, (0, 3, 1, 2))  # NCHW
        # rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

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

    def observation(self, obs):
        rgb = obs["sensors::top_camera::rgb"]

        proprio = np.concatenate(
            [obs[k].ravel() for k in obs if k.startswith("proprio")], axis=0
        )
        return {"rgb": rgb, "proprio": proprio}


class ActionSmoothingWrapper(gym.vector.VectorEnvWrapper):
    """
    Simple exponential moving average smoother for continuous actions.
    Keeps per-env state so resets don't bleed through.
    """

    def __init__(
        self,
        venv: gym.vector.VectorEnv,
        alpha: float = 0.3,
        clip_delta: Optional[float] = None,
    ):
        super().__init__(venv)
        if not isinstance(venv.single_action_space, spaces.Box):
            raise ValueError("Action smoothing only supports Box action spaces.")
        self.alpha = float(alpha)
        self.clip_delta = None if clip_delta is None else float(clip_delta)
        self._prev_actions: Optional[np.ndarray] = None
        self._reset_mask: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._prev_actions = None
        self._reset_mask = None
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        smoothed = self._apply_filter(actions)
        obs, rewards, terminated, truncated, info = self.env.step(smoothed)
        dones = np.logical_or(terminated, truncated)
        if self._reset_mask is None:
            self._reset_mask = dones.copy()
        else:
            self._reset_mask = dones
        return obs, rewards, terminated, truncated, info

    def _apply_filter(self, actions: Any) -> Any:
        if self.alpha <= 0.0:
            return actions

        arr = np.asarray(actions, dtype=np.float32)
        if self._prev_actions is None:
            self._prev_actions = arr
            self._reset_mask = np.zeros(arr.shape[0], dtype=bool)
            return arr

        if self._reset_mask is None or self._reset_mask.shape[0] != arr.shape[0]:
            self._reset_mask = np.zeros(arr.shape[0], dtype=bool)

        smoothed = np.array(self._prev_actions, copy=True)
        reset_mask = self._reset_mask

        # Do not smooth freshly reset envs
        smoothed[reset_mask] = arr[reset_mask]

        keep_mask = ~reset_mask
        if keep_mask.any():
            smoothed[keep_mask] = (
                self.alpha * arr[keep_mask]
                + (1.0 - self.alpha) * self._prev_actions[keep_mask]
            )
            if self.clip_delta is not None:
                delta = smoothed[keep_mask] - self._prev_actions[keep_mask]
                delta = np.clip(delta, -self.clip_delta, self.clip_delta)
                smoothed[keep_mask] = self._prev_actions[keep_mask] + delta

        self._prev_actions = smoothed
        self._reset_mask = np.zeros_like(reset_mask)
        return smoothed


class SB3GymVectorAdapter(VecEnv):
    """
    Adapter to make a Gymnasium VectorEnv compatible with SB3.
    """

    def __init__(self, vec_env: gym.vector.VectorEnv):
        obs_space = vec_env.single_observation_space
        act_space = vec_env.single_action_space
        super().__init__(vec_env.num_envs, obs_space, act_space)
        self._vec_env = vec_env
        self._actions = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._vec_env.reset(seed=seed, options=options)
        return obs

    def step_async(self, actions: Any) -> None:
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self._vec_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)
        # ensure infos is a list
        if isinstance(infos, dict):
            infos = [infos for _ in range(self.num_envs)]
        return obs, rewards, dones, infos

    def close(self) -> None:
        return self._vec_env.close()

    def render(self, mode: str = "human"):
        return getattr(self._vec_env, "render", lambda mode=None: None)(mode)

    # ---- implement all abstract methods correctly ----
    def get_attr(self, attr_name: str, indices: Optional[list[int]] = None):
        envs = getattr(self._vec_env, "envs", None)
        if envs is None:
            raise AttributeError(
                "Underlying VectorEnv does not expose per-env instances."
            )
        idxs = list(range(self.num_envs)) if indices is None else indices
        return [getattr(envs[i], attr_name) for i in idxs]

    def set_attr(
        self, attr_name: str, value: Any, indices: Optional[list[int]] = None
    ) -> None:
        envs = getattr(self._vec_env, "envs", None)
        if envs is None:
            raise AttributeError(
                "Underlying VectorEnv does not expose per-env instances."
            )
        idxs = list(range(self.num_envs)) if indices is None else indices
        for i in idxs:
            setattr(envs[i], attr_name, value)

    def env_method(
        self, method_name: str, *args, indices: Optional[list[int]] = None, **kwargs
    ):
        envs = getattr(self._vec_env, "envs", None)
        if envs is None:
            raise AttributeError(
                "Underlying VectorEnv does not expose per-env instances."
            )
        idxs = list(range(self.num_envs)) if indices is None else indices
        return [getattr(envs[i], method_name)(*args, **kwargs) for i in idxs]

    def env_is_wrapped(self, wrapper_class, indices: Optional[list[int]] = None):
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
    action_smoothing: Optional[dict[str, Any]] = None,
) -> VecEnv:
    """Construct a monitored SB3 vector environment with the multi-input wrapper."""
    vec_env = make_vector_env(
        env_cls,
        num_envs=num_envs,
        channel_schema=channel_schema,
        global_config=global_config,
        sim=sim,
        asynchronous=asynchronous,
    )

    if action_smoothing:
        vec_env = ActionSmoothingWrapper(
            vec_env,
            alpha=float(action_smoothing.get("alpha", 0.3)),
            clip_delta=action_smoothing.get("clip_delta"),
        )

    vec_env = ObservationWrapper(vec_env)
    vec_env.reset()
    adapted = SB3GymVectorAdapter(vec_env)
    return VecMonitor(adapted)


@ALGOS.register("sb3_rl")
class StableBaselinesRLAlgorithm(BaseAlgorithm):
    """Train Stable-Baselines3 algorithms on Ark vectorized environments."""

    def __init__(self, policy: Any, device: str, cfg: DictConfig):
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
        self._sb3_kwargs: dict[str, Any] = dict(sb3_cfg.get("kwargs", {}) or {})
        self._sb3_kwargs.setdefault("device", device)
        self._model = None
        self.output_dir = (
            Path(self.cfg.output_dir) / "sb3_rl" / datetime.now().strftime("%Y-%m-%d")
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

        self._env = build_vector_env(
            env_cls=env_cls,
            channel_schema=channel_schema,
            global_config=global_config,
            num_envs=num_envs,
            asynchronous=asynchronous,
            sim=sim,
            action_smoothing=action_smoothing,
        )

    def train(self) -> dict[str, Any]:
        """Train the selected SB3 algorithm with TensorBoard reward logging."""

        model = self._sb3_algo_class(
            policy=self._policy_name,
            env=self._env,
            tensorboard_log=str(self.output_dir / "tensorboard"),
            **self._sb3_kwargs,
        )
        self._model = model

        # Log both progress and episode rewards
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
        """Evaluate the trained model over a number of episodes."""
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
    from arkml.core.rl.franka_env import FrankaPickPlaceEnv
    from ark.utils.video_recorder import VideoRecorder
    from tqdm import tqdm

    vec_env = make_vector_env(
        FrankaPickPlaceEnv,
        num_envs=1,
        channel_schema="ark_framework/ark/configs/franka_panda.yaml",
        global_config="ark_diffusion_policies_on_franka/diffusion_policy/config/global_config.yaml",
        sim=True,
        asynchronous=False,
    )
    env = ObservationWrapper(vec_env)

    model = PPO.load("outputs/sb3_rl/sb3_model_2.zip")

    obs, _ = env.reset()
    rec = VideoRecorder("outputs/sb3_rl/rollout.mp4", fps=20)

    horizon = 500

    with tqdm(total=horizon) as pbar:
        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # print(reward, done)
            rec.add_frame(obs)

            pbar.update(1)

            if done or truncated:
                print("Episode finished")
                break

    rec.close()
    env.close()
