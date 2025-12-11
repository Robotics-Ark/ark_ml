# Reinforcement Learning (SB3)

This module wraps Ark vectorized environments with Stable-Baselines3 (SB3) to train PPO/SAC/TD3/DDPG policies on simulated or real hardware. It provides standardized observation/action adapters, optional action smoothing, and Hydra-driven configuration for reproducible runs.

## 1. Introduction
- `sb3_algorithm.py` builds SB3-compatible vector environments via `make_vector_env`, applies observation flattening (`ObservationWrapper`), optional action smoothing, and a gym adapter (`SB3GymVectorAdapter` + `VecMonitor`).
- Supported algorithms are `ppo`, `sac`, `td3`, `ddpg` with multi-input policies (image + proprioception).
- Training is orchestrated through the generic `arkml.tools.train` entrypoint and Hydra configs under `arkml/configs/algo`.
- Rollout is orchestrated through the generic `arkml.tools.policy_service` entrypoint.

## 2. Sample config
Example: `arkml/configs/algo/sb3rl.yaml`

```yaml
name: sb3rl
model:
  name: sb3_dummy        # placeholder model; SB3 owns the policy internally

env:
  class_path: "arkml.examples.rl.franka_env.FrankaPickPlaceEnv"
  num_envs: 2
  kwargs:
    max_steps: 200
    config_path: "ark_diffusion_policies_on_franka/diffusion_policy/config/global_config.yaml"
    channel_schema: "ark_framework/ark/configs/franka_panda.yaml"
    asynchronous: True     # use AsyncVectorEnv when True, SyncVectorEnv otherwise
    sim: True              # launch simulator processes; set False for real robot

sb3:
  algo_name: "ppo"         # one of: ppo, sac, td3, ddpg
  policy: "MultiInputPolicy"
  total_timesteps: 1000000
  eval_episodes: 5
  action_smoothing:        # optional EMA + per-dimension clipping
    alpha: 0.4
    clip_delta: 0.05
    warmup_steps: 3
    warmup_clip_delta: 0.02
  kwargs:                  # forwarded directly to the SB3 constructor
    learning_rate: 3.0e-4
    gamma: 0.99
```

Key fields:
- `env.class_path` and `env.kwargs`: points to the Ark `ArkEnv` subclass plus simulator/robot config.
- `env.num_envs`: number of parallel workers for data collection.
- `sb3.algo_name`/`policy`: which SB3 learner to instantiate (multi-input handles dict obs).
- `sb3.action_smoothing`: exponential moving average + delta clipping applied before actions reach the robot.
- `sb3.kwargs`: SB3 hyperparameters (learning rate, discount, etc.).

## 3. How to make a vectorized Env
Use the helper to wrap Ark environments for SB3:

```python
from ark.env.vector_env import make_vector_env
from arkml.examples.rl.franka_env import FrankaPickPlaceEnv

vec_env = make_vector_env(
    env_cls=FrankaPickPlaceEnv,
    channel_schema="ark_framework/ark/configs/franka_panda.yaml",
    global_config="ark_diffusion_policies_on_franka/diffusion_policy/config/global_config.yaml",
    num_envs=4,
    asynchronous=True,
    sim=True,
    env_kwargs=kwargs,
)
```

`make_vector_env` constructs the vector env, applies optional action smoothing, converts observations to `{"rgb": NCHW, "proprio": flat}` for `MultiInputPolicy`.

## 4. Training
Run the Hydra entrypoint and override the defaults as needed:

```bash
HYDRA_FULL_ERROR=1 python -m arkml.tools.train \
  algo=sb3rl \
  algo.env.num_envs=4 \
  algo.env.kwargs.namespace=rl_franka \
  algo.env.kwargs.config_path=ark_diffusion_policies_on_franka/diffusion_policy/config/global_config.yaml \
  algo.env.kwargs.channel_schema=ark_framework/ark/configs/franka_panda.yaml \
  algo.sb3.algo_name=ppo \
  algo.sb3.total_timesteps=200000 \
  output_dir=outputs/rl/franka
```

Outputs are stored under `outputs/sb3_rl/<timestamp>/` and include the saved SB3 model (`sb3_model.zip`) plus TensorBoard logs at `tensorboard/`. Adjust `sb3.kwargs` for hyperparameters or switch `algo_name` to `sac/td3/ddpg`.

## 5. How to do policy rollout
Run the common policy service entry point with `algo=`

```bash
HYDRA_FULL_ERROR=1 python -m ark_ml.arkml.tools.policy_service  algo=sbr3rl algo.model.model_path=path/to/trained/policy.zip
```

For visual rollouts, wrap the loop with `ark.utils.video_recorder.VideoRecorder` (see the example in `sb3_algorithm.py`) or stream frames directly from the vector env.
