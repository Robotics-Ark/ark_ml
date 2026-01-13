from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
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
from lerobot.processor.normalize_processor import NormalizerProcessorStep as Normalize, \
    UnnormalizerProcessorStep as Unnormalize
from torch import tensor

from arkml.core.app_context import ArkMLContext


@MODELS.register("smolVLAnet")
class smolVLAnet(BasePolicy):
    """
    VLA SmolVLAPolicy policy wrapper that uses explicit lerobot policies with a switchable type models of that kind.

    - policy_type: 'SmolVLAPolicy'
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
            pred_horizon: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_dim = image_dim  # Should be (C, H, W) e.g., (3, 224, 224)
        self.device = None
        self._tokenizer = None

        kind = policy_type.lower()
        if kind != "smolvla":
            raise ValueError(f"Unsupported policy_type '{policy_type}'. Use 'smolvla'.")

        policy_class = SmolVLAPolicy

        self._policy = policy_class.from_pretrained(model_path)

        self._policy.config.n_action_steps = pred_horizon
        self._load_input_output_features()

    def _get_tokenizer(self):
        """
        Load and cache the tokenizer for SmolVLM2.
        Uses the tokenizer associated with the model specified in config.vlm_model_name.

        Returns:
            Tokenizer instance or None if loading fails.
        """
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return None

        # Get the VLM model name from config
        vlm_model_name = getattr(self._policy.config, "vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        self._tokenizer = AutoTokenizer.from_pretrained(vlm_model_name)
        return self._tokenizer

    def _infer_batch_size(self, observation: dict) -> int:
        """Infer batch size from observation tensors."""
        for value in observation.values():
            if torch.is_tensor(value) and value.dim() > 0:
                return value.shape[0]
        return 1

    def to_device(self, device: str) -> Any:
        """
        Move the underlying policy to a device and return self.
        Args:
            device: Target device identifier (e.g., "cuda", "cpu").

        Returns:
            smolVLAnet: This instance, for method chaining.

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

    def _tokenize_task(self, task: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize task description(s) using the SmolVLM2 tokenizer.
        Ensures task ends with newline character as expected by SmolVLM.
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None, None

        # Convert task to list of strings
        if isinstance(task, str):
            texts = [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            texts = task
        else:
            texts = [str(task)]

        # CRITICAL: Ensure each task ends with newline (required by SmolVLM/PaliGemma)
        texts = [t if t.endswith("\n") else f"{t}\n" for t in texts]

        # Get tokenizer config
        cfg = self._policy.config
        max_len = cfg.tokenizer_max_length  # 48
        pad_strategy = cfg.pad_language_to  # "max_length"

        # Tokenize with proper padding
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=pad_strategy,
            max_length=max_len,
            return_tensors="pt",
        )

        return tokenized["input_ids"], tokenized["attention_mask"]

    def prepare_input(self, observation: dict) -> dict[str, Any]:
        """
        Convert an observation dict into the policy's expected input format.

        Expected observation keys:
        - "state": Robot state tensor (B, state_dim) or (state_dim,)
        - "task": Task description string or list of strings
        - Camera names from ArkMLContext.visual_input_features: Image tensors (B, C, H, W) or (C, H, W)
        - "action" (optional, for training): Action tensor (B, action_horizon, action_dim)
        - "action_is_pad" (optional, for training): Boolean mask (B, action_horizon)
        """
        obs = {}
        cfg = self._policy.config

        # Use the configured tokenizer max length
        token_seq_length = cfg.tokenizer_max_length  # 48

        # Infer batch size first
        batch_size = self._infer_batch_size(observation)

        # Handle task tokenization
        if "task" in observation:
            task = observation["task"]
            input_ids, attention_mask = self._tokenize_task(task)

            if input_ids is not None:
                # Verify the tokenized length matches expected
                actual_len = input_ids.shape[1]
                if actual_len != token_seq_length:
                    print(f"WARNING: Tokenized length {actual_len} != expected {token_seq_length}")
                    # Adjust to match expected length
                    if actual_len < token_seq_length:
                        # Pad to token_seq_length
                        pad_len = token_seq_length - actual_len
                        input_ids = torch.cat([
                            input_ids,
                            torch.zeros(batch_size, pad_len, dtype=torch.long, device=input_ids.device)
                        ], dim=1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.zeros(batch_size, pad_len, dtype=torch.bool, device=attention_mask.device)
                            # FIXED: bool dtype
                        ], dim=1)
                    else:
                        # Truncate
                        input_ids = input_ids[:, :token_seq_length]
                        attention_mask = attention_mask[:, :token_seq_length]

                # CRITICAL: Convert attention mask to boolean
                obs["observation.language.tokens"] = input_ids.to(self.device)
                obs["observation.language.attention_mask"] = attention_mask.bool().to(self.device)  # FIXED: .bool()
            else:
                # Fallback if tokenization fails
                obs["observation.language.tokens"] = torch.zeros(
                    batch_size, token_seq_length, dtype=torch.long, device=self.device
                )
                obs["observation.language.attention_mask"] = torch.zeros(
                    batch_size, token_seq_length, dtype=torch.bool, device=self.device  # FIXED: bool dtype
                )
        else:
            # Provide default empty tokens with exact length
            obs["observation.language.tokens"] = torch.zeros(
                batch_size, token_seq_length, dtype=torch.long, device=self.device
            )
            obs["observation.language.attention_mask"] = torch.zeros(
                batch_size, token_seq_length, dtype=torch.bool, device=self.device  # FIXED: bool dtype
            )

        # Process state
        if "state" in observation:
            state = observation["state"]
            # Ensure state has batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)
            obs["observation.state"] = state.to(self.device)
        else:
            # Provide zero state if missing
            obs["observation.state"] = torch.zeros(
                batch_size, self.obs_dim, dtype=torch.float32, device=self.device
            )

        # Process images - CRITICAL: Images must be in range [0, 1]
        # Track which cameras are present
        cameras_present = []
        for cam_name in ArkMLContext.visual_input_features:
            if cam_name in observation:
                img = observation[cam_name]

                # Ensure batch dimension
                if img.dim() == 3:  # (C, H, W)
                    img = img.unsqueeze(0)  # (1, C, H, W)

                # Ensure correct batch size
                if img.shape[0] != batch_size:
                    if img.shape[0] == 1 and batch_size > 1:
                        img = img.repeat(batch_size, 1, 1, 1)
                    else:
                        raise ValueError(f"Image batch size {img.shape[0]} doesn't match expected {batch_size}")

                # Ensure images are in [0, 1] range
                if img.max() > 1.0:
                    print(f"WARNING: Image '{cam_name}' has values > 1.0, normalizing to [0, 1]")
                    img = img / 255.0

                obs[f"observation.images.{cam_name}"] = img.to(self.device)
                cameras_present.append(cam_name)
            # Note: Don't add padding masks - let the model handle missing cameras

        # Verify we have at least one camera
        if len(cameras_present) == 0:
            raise ValueError(
                f"At least one camera image is required. "
                f"Expected cameras: {ArkMLContext.visual_input_features}, "
                f"Provided keys: {observation.keys()}"
            )

        # Process actions (for training)
        if "action" in observation:
            action = observation["action"]
            # Ensure batch dimension
            if action.dim() == 2:  # (action_horizon, action_dim)
                action = action.unsqueeze(0)  # (1, action_horizon, action_dim)

            # Verify action shape matches config
            expected_chunk_size = cfg.chunk_size
            actual_chunk_size = action.shape[1]

            if actual_chunk_size != expected_chunk_size:
                print(f"WARNING: Action chunk size {actual_chunk_size} != expected {expected_chunk_size}")
                # Adjust action sequence length
                if actual_chunk_size < expected_chunk_size:
                    # Pad with zeros
                    pad_len = expected_chunk_size - actual_chunk_size
                    padding = torch.zeros(
                        action.shape[0], pad_len, action.shape[2],
                        dtype=action.dtype, device=action.device
                    )
                    action = torch.cat([action, padding], dim=1)
                else:
                    # Truncate
                    action = action[:, :expected_chunk_size, :]

            obs["action"] = action.to(self.device)

        if "action_is_pad" in observation:
            action_is_pad = observation["action_is_pad"]
            # Ensure batch dimension
            if action_is_pad.dim() == 1:
                action_is_pad = action_is_pad.unsqueeze(0)

            # Verify shape
            expected_chunk_size = cfg.chunk_size
            actual_chunk_size = action_is_pad.shape[1]

            if actual_chunk_size != expected_chunk_size:
                # Adjust padding mask
                if actual_chunk_size < expected_chunk_size:
                    # Pad with True (these are padding positions)
                    pad_len = expected_chunk_size - actual_chunk_size
                    padding = torch.ones(
                        action_is_pad.shape[0], pad_len,
                        dtype=torch.bool, device=action_is_pad.device  # FIXED: bool dtype
                    )
                    action_is_pad = torch.cat([action_is_pad, padding], dim=1)
                else:
                    # Truncate
                    action_is_pad = action_is_pad[:, :expected_chunk_size]

            # CRITICAL: Ensure boolean dtype
            obs["actions_id_pad"] = action_is_pad.bool().to(self.device)  # FIXED: .bool()

        return obs

    def predict(self, obs: dict[str, Any], **kwargs) -> tensor:
        """
        Select an action for a single observation.
        Args:
            obs: Observation dictionary with keys:
                - "state": (state_dim,) or (1, state_dim)
                - "task": string
                - Camera names: (C, H, W) or (1, C, H, W) in range [0, 1]
            **kwargs: Additional keyword arguments forwarded to `select_action`.

        Returns:
            Predicted action of shape (action_dim,) or (1, action_dim)
        """
        obs_prep = self.prepare_input(observation=obs)
        action = self._policy.select_action(obs_prep, **kwargs)
        return action

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
            action = self._policy.select_action(obs_prep)
            # Remove batch dimension if present
            if action.dim() == 2 and action.size(0) == 1:
                action = action.squeeze(0)
            actions.append(action)

        return torch.stack(actions, dim=0)

    def get_trainable_params(self) -> list[nn.parameter.Parameter]:
        """
        Return the parameters that should be optimized during training.

        Returns:
            List of parameters to optimize.
        """
        print_trainable_summary(self._policy)
        params = [p for p in self._policy.parameters() if p.requires_grad]
        return params

    def forward(self, observation) -> tensor:
        """
        Compute the training loss for a batch.
        Prepares the observation into the policy's expected format and delegates
        to the wrapped policy's `forward`.

        Args:
            observation: Batch observation dictionary with keys:
                - "state": (B, state_dim)
                - "task": list of B strings or single string
                - Camera names: (B, C, H, W) in range [0, 1]
                - "action": (B, action_horizon, action_dim)
                - "action_is_pad" (optional): (B, action_horizon) boolean mask

        Returns:
            Scalar loss tensor for the batch.
        """
        batch = self.prepare_input(observation=observation)
        loss, loss_dict = self._policy.forward(batch)

        # Optionally log loss_dict for debugging
        # print(f"Loss breakdown: {loss_dict}")

        return loss

    def save_policy(self, out_dir: str) -> None:
        """
        Save the full fine-tuned model via the underlying policy's `save_pretrained`.

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
                for keys like 'observation.state', 'observation.images.{cam}', 'action'.
        """
        stats_path = Path(dataset_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Dataset stats file not found: {stats_path}")

        with open(stats_path, "r") as f:
            raw = json.load(f)

        # Convert to torch tensors
        loaded_stats = {}
        for k, d in raw.items():
            loaded_stats[k] = {}
            for kk, vv in d.items():
                if isinstance(vv, list):
                    loaded_stats[k][kk] = torch.tensor(vv)
                else:
                    loaded_stats[k][kk] = torch.tensor(np.array(vv))

        # Update policy config with stats
        norm_map = getattr(self._policy.config, "normalization_mapping", None)
        if norm_map is None:
            print("WARNING: No normalization_mapping found in config")
            return

        # Recreate normalizers with new stats
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

        print(f"[Model] Loaded dataset stats from {stats_path}")

    def _load_input_output_features(self) -> None:
        """
        Configure input and output features for the policy.
        """
        input_features = {
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(self.obs_dim,)
            )
        }

        # Add image features
        for cam_name in ArkMLContext.visual_input_features:
            input_features[f"observation.images.{cam_name}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=self.image_dim  # (C, H, W)
            )

        self._policy.config.input_features = input_features

        self._policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        }