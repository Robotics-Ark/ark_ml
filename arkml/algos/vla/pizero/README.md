# PiZero (Pi0)

PiZero fine-tunes an open VLA-style policy on task demonstrations with text-conditioned actions. The Ark wrapper covers dataset stats, train/val splits, Hydra configs, and a policy node that streams actions in the Ark runtime.

## 1. Introduction
- `algorithm.py` loads the PiZero dataset, computes dataset stats (`pizero_stats.json`), performs an 80/20 split, and hands training to `PiZeroTrainer` with validation via `PiZeroEvaluator`.
- `models.py` contains the PiZero network variants; `models_with_lora.py` adds LoRA options if needed.
- `nodes/pizero_node.py` wraps the trained policy for online rollout; it listens to a text prompt channel, builds state/image tensors, and outputs the next action sequence.
- Dataset utilities live in `dataset.py` and `compute_stats.py`, which normalize shapes and cache statistics for consistent preprocessing.

## 2. Sample config and explanation
Config: `arkml/configs/algo/pizero.yaml`

```yaml
name: pizero
model:
  type: PiZeroNet
  name: PiZeroNet
  policy_type: pi0             # registry key used by policy_service
  model_path: lerobot/pi0      # HF/locally cached base weights; override with fine-tuned ckpt
  obs_dim: 9
  action_dim: 8
  obs_horizon: 1
  pred_horizon: 1
  action_horizon: 1
  image_dim: (3, 480, 640)

trainer:
  lr: 2e-4
  batch_size: 8
  max_epochs: 10
  num_workers: 4
  use_bf16: true
  weight_decay: 0.0
```

Notes:
- `model_path`: base checkpoint to start from (local path or Hub id). For rollout, set to your fine-tuned weights.
- `obs_dim`/`action_dim`/`image_dim`: used to validate dataset shapes and compute normalization stats.
- `policy_type` selects the correct policy node via `policy_registry`.

## 3. Dataset expectations
- Directory with trajectory files used by `PiZeroDataset`. Each sample provides an image, state vector, and action.
- On first run, `algorithm.calculate_dataset_stats` writes `pizero_stats.json` into the dataset root (mean/std for image/state/action).
- Train/val split is 80/20 with optional worker prefetch; adjust `trainer.num_workers` for throughput.

## 4. How to do training

```bash
HYDRA_FULL_ERROR=1 python -m arkml.tools.train \
  algo=pizero \
  data.dataset_path=/path/to/pizero_dataset \
  output_dir=outputs/pizero \
  algo.model.model_path=lerobot/pi0 \
  algo.trainer.batch_size=8 \
  algo.trainer.max_epochs=20
```

Artifacts (checkpoints, logs) are saved under `outputs/pizero/`.

## 5. How to do policy rollout
Launch the policy node as a service (listens for a text prompt on the `user_input` channel by default):

```bash
HYDRA_FULL_ERROR=1 python -m arkml.tools.policy_service \
  algo=pizero \
  algo.model.policy_type=pi0 \
  algo.model.model_path=/path/to/fine_tuned_pi0.pt \
  channel_schema=ark_framework/ark/configs/franka_panda.yaml \
  global_config=ark_ml/arkml/examples/franka_pick_place/franka_config/global_config.yaml \
  node_name=pi0_policy
```

Flow:
- `policy_service` builds `PiZeroPolicyNode`, loads the checkpoint, and resets internal queues.
- The node reads camera + proprio observations from the configured channels, pairs them with the latest text prompt, and publishes actions back to the Ark runtime.
- For offline evaluation after training, call `algo.eval()` (uses `PiZeroEvaluator`) to compute validation metrics.
