from collections import deque
from typing import Any

import numpy as np
import torch
from arkml.algos.vla.pizero.models import PiZeroNet
from arkml.algos.vla.pizero.config_utils import resolve_visual_feature_names
from arkml.core.policy_node import PolicyNode
from arkml.utils.schema_io import (
    load_schema,
    make_observation_unpacker,
)


class PiZeroPolicyNode(PolicyNode):
    """Wrapper node for PiZero"""

    def __init__(self, cfg, device: str):
        model_cfg = cfg.algo.model
        self.visual_input_features = resolve_visual_feature_names(
            getattr(model_cfg, "visual_input_features", None)
        )

        policy = PiZeroNet(
            policy_type=model_cfg.policy_type,
            model_path=model_cfg.model_path,
            obs_dim=model_cfg.obs_dim,
            action_dim=model_cfg.action_dim,
            image_dim=model_cfg.image_dim,
            visual_input_features=self.visual_input_features,
        )

        io_schema_path = getattr(cfg, "io_schema", "default_io_schema.yaml")
        schema = load_schema(io_schema_path)
        obs_unpacker = make_observation_unpacker(schema)

        super().__init__(
            policy=policy,
            device=device,
            policy_name=cfg.policy_node_name,
            observation_unpacking=obs_unpacker,
            global_config=cfg.global_config,
        )

        self.policy.to_device(device)
        self.policy.reset()
        self.policy.set_eval_mode()
        self.task_prompt = model_cfg.task_prompt or ""

        self.n_infer_actions = getattr(model_cfg, "pred_horizon", 10)
        self._action_queue: deque[np.ndarray] = deque()

    def _on_reset(self):
        self._action_queue.clear()

    def prepare_observation(self, ob: dict[str, Any], task_prompt: str):
        """Convert a single raw env observation into a batched policy input.

        Notes:
          - Produces per-camera keys matching ``self.visual_input_features``.
          - Avoids redundant tensor copies; normalizes images to float32 in [0, 1].

        Args:
          ob: Single observation dict from the env. Expected keys include
            ``state`` and any camera names listed in ``visual_input_features``.
          task_prompt: Natural language task description to include in the batch.

        Returns:
          A batch dictionary with:
            - per-camera image tensors: ``torch.FloatTensor`` of shape ``[1, C, H, W]``.
            - ``state``: ``torch.FloatTensor`` of shape ``[1, D]`` if present.
            - ``task``: ``list[str]`` of length 1.
        """
        obs: dict[str, Any] = {"task": [task_prompt]}

        # State: accept numpy array or tensor, ensure [1, D] float32
        state_value = ob.get("state")
        if state_value is not None:
            if isinstance(state_value, torch.Tensor):
                state_t = state_value
            else:
                state_t = torch.from_numpy(state_value)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            obs["state"] = state_t.to(dtype=torch.float32, copy=False)

        # Images: accept HWC (numpy/tensor) or CHW tensor, ensure [1, C, H, W] float32 in [0,1]
        for cam_name in self.visual_input_features:
            value = ob.get(cam_name)
            if value is None:
                raise KeyError(f"Missing visual input '{cam_name}' in observation")

            if isinstance(value, torch.Tensor):
                img_t = value
                # If HWC, convert to CHW
                if img_t.dim() == 3 and img_t.shape[0] in (1, 3) and img_t.shape[-1] in (1, 3):
                    # Ambiguous; prefer assuming HWC if last dim is channels
                    if img_t.shape[-1] in (1, 3):
                        img_t = img_t.permute(2, 0, 1)
                elif img_t.dim() == 3 and img_t.shape[-1] not in (1, 3):
                    # Likely HWC
                    img_t = img_t.permute(2, 0, 1)
            else:
                img_np = value
                # Ensure contiguous numpy array
                if hasattr(img_np, "copy"):
                    img_np = img_np.copy()
                img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # (C, H, W)

            # Normalize to float32 in [0, 1]
            if img_t.dtype == torch.uint8:
                img_t = img_t.to(torch.float32).div(255.0)
            else:
                img_t = img_t.to(torch.float32)
                # If values appear in [0,255], normalize
                if torch.isfinite(img_t).all() and img_t.max() > 1.0:
                    img_t = img_t.div(255.0)

            if img_t.dim() == 3:
                img_t = img_t.unsqueeze(0)  # (1, C, H, W)

            obs[cam_name] = img_t

        return obs

    def predict(self, obs_seq):
        """Compute the action for the given observation batch.

        The expected structure of ``obs_seq`` is dictated by the underlying VLA
        policy (typically a dict with batched tensors for images and state, and
        a list[str] for the task prompt).

        Args:
          obs_seq: Observation input to the policy (dict or tensor as required
            by the wrapped model).

        Returns:
          numpy.ndarray: Action vector for the first batch element.
        """
        obs = self.prepare_observation(obs_seq, self.task_prompt)

        if len(self._action_queue) == 0:
            with torch.no_grad():
                actions = self.policy.predict_n_actions(
                    obs, n_actions=self.n_infer_actions
                )
            actions = actions.detach().cpu().numpy()
            for action in actions:
                self._action_queue.append(action)

        return self._action_queue.popleft()
