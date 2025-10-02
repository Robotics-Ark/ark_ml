import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from arkml.core.policy import BasePolicy
from arkml.core.registry import MODELS
from arkml.utils.utils import print_trainable_summary
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from torch import tensor

from arkml.core.app_context import ArkMLContext


@MODELS.register("PiZeroNet")
class PiZeroNet(BasePolicy):
    """
    VLA PiZero policy wrapper that uses explicit lerobot policies with a switchable type models of that kind.

    - policy_type: 'pi0'
    - pretrained_model: HF hub id or local path. If None, uses a sensible default per type.
    - Numeric state only is supported out-of-the-box (passed as 'observation.state').
      To use image-based policies like Pi0, pass a full observation dict with
      the required image tensors and task string.
    """

    def __init__(
        self,
        policy_type: str,
        model_path: str,
        obs_dim: int,
        action_dim: int,
        image_dim: tuple,
        pred_horizon: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_dim = image_dim
        self.device = None

        kind = policy_type.lower()
        if kind != "pi0":
            raise ValueError(f"Unsupported policy_type '{policy_type}'. Use 'pi0'.")

        policy_class = PI0Policy

        self._policy = policy_class.from_pretrained(model_path)

        self._policy.config.n_action_steps = pred_horizon
        self._load_input_output_features()

    def to_device(self, device: str) -> Any:
        """
        Move the underlying policy to a device and return self.
        Args:
            device: Target device identifier (e.g., "cuda", "cpu").

        Returns:
            PiZeroNet: This instance, for method chaining.

        """
        self.device = device
        self._policy.to(device)
        return self

    def set_eval_mode(self) -> None:
        """
        Set the underlying policy to evaluation mode.
        """
        self._policy.eval()

    def set_train_mode(self) -> None:
        """
        Set the underlying policy to training mode.
        """
        self._policy.train()

    def reset(self) -> None:
        """
        Reset internal policy state.
        """
        self._policy.reset()

    def prepare_input(self, observation: dict) -> dict[str, Any]:
        """
        Convert an observation dict into the policy’s expected input format.

        Expected keys in `observation`:
            - "image": torch.Tensor of shape (B, C, H, W)
            - "state": torch.Tensor of shape (B, state_dim)
            - "task": str task prompt or instruction
            - "action" (optional): torch.Tensor of shape (B, action_dim)

        Args:
            observation: Raw observation dictionary.

        Returns:
            Processed observation with keys:
                - "observation.images.image": torch.Tensor on `self.device`
                - "observation.state": torch.Tensor on `self.device`
                - "task": str (unchanged)
                - "action": torch.Tensor on `self.device` (if present)
        """
        obs = {}
        for k, v in observation.items():
            if k == "state":
                obs["observation.state"] = v.to(self.device)
            elif k == "task":
                obs["task"] = v
            elif k in {"action", "action_is_pad"}:
                obs[k] = v.to(self.device)
            elif k in ArkMLContext.visual_input_features:
                obs[f"observation.images.{k}"] = v.to(self.device)
        return obs

    def predict(self, obs: dict[str, Any], **kwargs) -> tensor:
        """
        Select an action for a single observation.
        Args:
            obs: Observation dictionary
            **kwargs: Additional keyword arguments forwarded to `select_action`.

        Returns:
            Predicted action
        """
        obs = self.prepare_input(observation=obs)
        return self._policy.select_action(obs)

    def predict_n_actions(self, obs: dict[str, Any], n_actions: int = 10) -> tensor:
        """
        Generate and return a sequence of `n_actions` actions.

        Uses the policy's internal action queue. If the queue is empty, the
        underlying policy will generate a chunk of size `config.n_action_steps`
        (default 50) and subsequent calls pop from that chunk.

        Args:
            obs: Observation dictionary.
            n_actions: Number of actions to return from the model.

        Returns:
            Tensor of shape (n_actions, action_dim) on the model device.
        """
        obs_prep = self.prepare_input(observation=obs)
        actions = []
        for _ in range(n_actions):
            actions.append(self._policy.select_action(obs_prep))
        # Stack to (n, action_dim). select_action returns (batch=1, action_dim) or (action_dim)

        actions = [
            a.squeeze(0) if a.dim() == 2 and a.size(0) == 1 else a for a in actions
        ]
        return torch.stack(actions, dim=0)

    def get_trainable_params(self) -> list[nn.parameter]:
        """
        Return the parameters that should be optimized during training.

        Returns:
            List of parameters to optimize.
        """
        print_trainable_summary(self._policy)
        params = [p for p in self._policy.parameters()]
        return params

    def forward(self, observation) -> tensor:
        """
        Compute the training loss for a batch.
        Prepares the observation into the policy’s expected format and delegates
        to the wrapped policy’s `forward`.
        Assumes the policy returns a
        `(loss, loss_dict)` tuple and this method returns the loss only.

        Args:
            observation: Batch observation (see `prepare_input`).

        Returns:
            Scalar loss tensor for the batch.
        """
        batch = self.prepare_input(observation=observation)
        loss, _ = self._policy.forward(batch)

        return loss

    def save_policy(self, out_dir: str) -> None:
        """
        Save the full fine-tuned model via the underlying policy’s  `save_pretrained`.

        Args:
            out_dir: Output directory to write model artifacts.

        """
        os.makedirs(out_dir, exist_ok=True)

        self._policy.save_pretrained(out_dir)
        print(f"[Model] Saved full model state_dict to {out_dir}")

    def load_dataset_stats(self, dataset_stats_path: str) -> None:
        """
        Load dataset stats from JSON and (re)initialize normalization modules.

        Args:
            dataset_stats_path: Path to a JSON file containing LeRobot-compatible stats
                for keys like 'observation.state', 'observation.images.image', 'action'.
        """

        stats_path = Path(dataset_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Dataset stats file not found: {stats_path}")

        with open(stats_path, "r") as f:
            raw = json.load(f)
        loaded_stats = {
            k: {kk: np.array(vv) for kk, vv in d.items()} for k, d in raw.items()
        }

        norm_map = getattr(self._policy.config, "normalization_mapping", None)
        if norm_map is None:
            return

        # Refresh buffers with current feature schemas
        self._policy.normalize_inputs = Normalize(
            self._policy.config.input_features, norm_map, loaded_stats
        )
        if hasattr(self._policy, "normalize_targets"):
            self._policy.normalize_targets = Normalize(
                self._policy.config.output_features, norm_map, loaded_stats
            )
        self._policy.unnormalize_outputs = Unnormalize(
            self._policy.config.output_features, norm_map, loaded_stats
        )

    def _load_input_output_features(self) -> None:
        input_features = {
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(self.obs_dim,)
            )
        }
        for cam_name in ArkMLContext.visual_input_features:
            input_features[f"observation.images.{cam_name}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=self.image_dim
            )
        self._policy.config.input_features = input_features

        self._policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        }
