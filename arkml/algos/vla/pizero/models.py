import os
from typing import Any

import torch.nn as nn
from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from torch import tensor

from arkml.core.policy import BasePolicy
from arkml.core.registry import MODELS


@MODELS.register("PiZeroNet")
class PiZeroNet(BasePolicy, nn.Module):
    """
    VLA PiZero policy wrapper that uses explicit lerobot policies with a switchable type models of that kind.

    - policy_type: 'pi0' or 'smolvla'
    - pretrained_model: HF hub id or local path. If None, uses a sensible default per type.
    - Numeric state only is supported out-of-the-box (passed as 'observation.state').
      To use image-based policies like SmolVLA, pass a full observation dict with
      the required image tensors and task string.
    """

    def __init__(
        self,
        policy_type: str,
        model_path: str,
        obs_dim: int,
        action_dim: int,
        image_dim: tuple,
        # LoRA config
        enable_lora: bool = False,
        lora_modules: list = None,
        # Config
        pred_horizon: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_dim = image_dim
        self._peft_module = None
        self.device = None

        self._peft_attached = False

        # LoRA Config
        self.lora_modules = lora_modules or []
        self.lora_params = []
        self.is_lora_enabled = enable_lora
        self._peft_attached = False

        kind = policy_type.lower()
        if kind not in {"pi0", "smolvla"}:
            raise ValueError(
                f"Unsupported policy_type '{policy_type}'. Use 'pi0' or 'smolvla'."
            )

        policy_class = PI0Policy if kind == "pi0" else SmolVLAPolicy

        try:
            self._policy = policy_class.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained {kind} policy '{model_path}': {e}"
            )

        # TODO need to investigate the policy config
        self._policy.config.norm_map = {
            FeatureType.STATE: NormalizationMode.IDENTITY,
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
            # FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
            # FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }

        self._policy.config.input_features = {
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=self.image_dim
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(self.obs_dim,)
            ),
        }
        self._policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
        }

        self._policy.normalize_inputs = Normalize(
            self._policy.config.input_features,
            self._policy.config.norm_map,
        )

        self._policy.unnormalize_outputs = Unnormalize(
            self._policy.config.output_features,
            self._policy.config.norm_map,
        )

        if self.is_lora_enabled:
            raise NotImplementedError("Lora policies not implemented yet to VLA.")
        else:
            for p in self._policy.parameters():
                p.requires_grad = True

        self._policy.config.n_action_steps = pred_horizon

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
        Moves tensors to `self.device` and maps keys to the feature schema
        expected by the underlying policy.

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
        obs = {
            "observation.images.image": observation["image"].to(self.device),
            "observation.state": observation["state"].to(self.device),
            "task": observation["task"],
        }
        if "action" in observation:
            obs["action"] = observation["action"].to(self.device)
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

    def get_trainable_params(self) -> list[nn.parameter]:
        """
        Return the parameters that should be optimized during training.
        If LoRA is enabled, returns LoRA parameters; otherwise returns all
        parameters of the underlying policy (and ensures they are trainable).

        Returns:
            List of parameters to optimize.
        """
        if self.is_lora_enabled:
            return self.lora_params
        else:
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
        Save LoRA adapters for the policy  If LoRA is enabled and attached, delegates to `save_lora`.
        Otherwise, save the full fine-tuned model via the underlying policy’s  `save_pretrained`.

        Args:
            out_dir: Output directory to write model artifacts.

        """
        os.makedirs(out_dir, exist_ok=True)

        if self.is_lora_enabled and self._peft_attached:
            self.save_lora(out_dir)
        else:
            self._policy.save_pretrained(out_dir)
            print(f"[Model] Saved full model state_dict to {out_dir}")

    def save_lora(self, out_dir: str) -> None:
        """
        Args:
            out_dir: Output directory to write LoRA adapter weights.

        Raises:
            NotImplementedError: Always raised; LoRA saving is not implemented yet.

        """
        raise NotImplementedError
