from collections import deque
from typing import Any

import numpy as np
import torch
from arkml.algos.vla.pizero.models import PiZeroNet
from arkml.algos.vla.pizero.config_utils import resolve_visual_feature_names
from arkml.core.policy_node import PolicyNode
from arkml.utils.schema_io import (
    load_schema,
    make_action_packer,
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
            image_dim=tuple(model_cfg.image_dim),
            visual_input_features=self.visual_input_features,
        )

        io_schema_path = getattr(cfg, "io_schema", "default_io_schema.yaml")
        schema = load_schema(io_schema_path)
        obs_unpacker = make_observation_unpacker(schema)
        act_packer = make_action_packer(schema)

        super().__init__(
            policy=policy,
            device=device,
            policy_name=cfg.policy_node_name,
            observation_unpacking=obs_unpacker,
            action_packing=act_packer,
            stepper_frequency=cfg.stepper_frequency,
            global_config=cfg.global_config,
        )

        self.policy.to_device(device)
        self.policy.reset()
        self.policy.set_eval_mode()
        self.create_stepper(10, self.step)
        self.task_prompt = model_cfg.task_prompt or ""

        self.n_infer_actions = getattr(model_cfg, "pred_horizon", 10)
        self._action_queue: deque[np.ndarray] = deque()

    def _on_reset(self):
        self._action_queue.clear()

    def prepare_observation(self, ob: dict[str, Any], task_prompt: str):
        """Convert a single raw env observation into a batched policy input.

        Args:
          ob: Single observation dict from the env. Expected to contain keys:
            ``images`` (tuple with RGB as HxWxC),
            ``cube``, ``target``, ``gripper``, and ``franka_ee``.
          task_prompt: Natural language task description to include in the batch.

        Returns:
          A batch dictionary with:
            - ``image``: ``torch.FloatTensor`` of shape ``[1, C, H, W]``.
            - ``state``: ``torch.FloatTensor`` of shape ``[1, D]``.
            - ``task``: ``list[str]`` of length 1.
        """
        obs: dict[str, Any] = {"task": [task_prompt]}

        state_value = ob.get("state")
        if state_value is not None:
            obs["state"] = self._to_tensor(state_value, add_batch=True)

        for cam_name in self.visual_input_features:
            value = ob.get(cam_name)
            if value is None:
                continue
            obs[cam_name] = self._to_tensor(value, is_image=True)

        return obs

    @staticmethod
    def _to_tensor(value: Any, *, is_image: bool = False, add_batch: bool = False) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.clone().detach().float()
        else:
            array = np.asarray(value)
            tensor = torch.from_numpy(array).float()

        if is_image:
            if tensor.dim() == 3:
                if tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
                    tensor = tensor.permute(2, 0, 1)
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 4:
                if tensor.shape[1] not in {1, 3} and tensor.shape[-1] in {1, 3}:
                    tensor = tensor.permute(0, 3, 1, 2)
            return tensor

        if add_batch and tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.float()

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
