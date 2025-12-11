# Diffusion Policy

Diffusion Policy trains a conditional denoising diffusion model on demonstration trajectories (images + low‑dim state) and deploys it via an online sampler. The Ark wrapper provides dataset preprocessing to Parquet, Hydra configs, trainers/evaluators, and a policy node for rollout.

## 1. Introduction
- `algorithm.py` wires dataset loading, train/val split, and hands off to `DiffusionTrainer` for optimization plus `DiffusionPolicyEvaluator` for validation.
- `dataset.py` converts raw `.pkl` trajectories into a cached `processed.parquet`, slicing windows with configurable observation, past-action, and prediction horizons.
- `models.py` implements the diffusion UNet backbone and scheduler helpers; `trainer.py` handles EMA, gradient clipping, and cosine LR scheduling.
- `nodes/diffusion_node.py` is a policy service wrapper that loads a saved checkpoint and streams actions in the Ark runtime.

## 2. Sample config and explanation
Config: `arkml/configs/algo/diffusion_policy.yaml`

```yaml
name: diffusion_policy
model:
  name: DiffusionPolicyModel
  state_dim: 8
  action_dim: 8
  obs_horizon: 8       # how many frames of context
  pred_horizon: 16     # how many future actions to predict
  action_horizon: 8    # how many past actions to condition on
  image_backbone: resnet18
  image_emb_dim: 256
  state_emb_dim: 128
  past_action_emb_dim: 128
  crop_shape: [224, 224]
  diffusion_steps: 100
  num_inference_steps: 16
  beta_schedule: squaredcos_cap_v2
  model_path: null     # set to a checkpoint for rollout
trainer:
  max_epochs: 100
  lr: 1.0e-4
  weight_decay: 1e-6
  ema: true
  ema_power: 0.75
  grad_clip: 1.0
  scheduler: cosine
  batch_size: 128
  log_every_n_steps: 50
  val_every_n_epochs: 10
  num_workers: 8
  prefetch_factor: 8
  max_steps: 100000    # optional step budget; maps to epochs automatically
```

Notes:
- `model.*`: observation/action dimensions, horizons, encoder backbones, diffusion schedule, and optional checkpoint for inference.
- `trainer.*`: optimizer + EMA + scheduler knobs. If `max_steps` is set, epochs are derived from steps per epoch.
- Data path is provided separately via `data.dataset_path` (see training command).

## 3. Dataset expectations
- Directory of trajectory `.pkl` files; each contains a list of timesteps with `state`, `action`, and camera images.
- On first load, `dataset.py` builds `processed.parquet` alongside the data for fast slicing.
- Windows use `obs_horizon`, `action_horizon`, and `pred_horizon` with optional temporal subsampling (`subsample` arg).

## 4. How to do training

```bash
HYDRA_FULL_ERROR=1 python -m arkml.tools.train \
  algo=diffusion_policy \
  data.dataset_path=/path/to/demos \
  output_dir=outputs/diffusion_policy \
  algo.model.state_dim=8 \
  algo.model.action_dim=8 \
  algo.trainer.batch_size=64
```

Artifacts land in `outputs/diffusion_policy/` with checkpoints, logs, and EMA weights (if enabled).

## 5. How to do policy rollout
Run the policy node as a service (uses `nodes/diffusion_node.py`):

```bash
HYDRA_FULL_ERROR=1 python -m arkml.tools.policy_service \
  algo=diffusion_policy \
  algo.model.model_path=/path/to/checkpoint.pt \
  channel_schema=ark_framework/ark/configs/franka_panda.yaml \
  global_config=ark_ml/arkml/examples/franka_pick_place/franka_config/global_config.yaml \
  node_name=diffusion_policy_node
```

The node will:
- Load the checkpoint, build the scheduler, and keep rolling horizon buffers for images/state/actions.
- Subscribe to observations per `channel_schema` and publish actions to the Ark runtime.
- For offline evaluation, call `algo.eval()` (uses `DiffusionPolicyEvaluator`) after training.
