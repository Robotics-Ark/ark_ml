from collections import deque
from typing import Any

import numpy as np
import torch
from arkml.algos.vla.pizero.models import PiZeroNet
from arkml.core.policy_node import PolicyNode
from arkml.utils.schema_io import (
    load_schema,
    make_observation_unpacker,
    make_action_packer,
)


class PiZeroPolicyNode(PolicyNode):
    """Wrapper node for PiZero

    Args:
      model_cfg: Model configurations.
      device: Target device string (e.g., ``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        cfg,
        device: str,
    ):
        model_cfg = cfg.algo.model
        policy = PiZeroNet(
            policy_type=model_cfg.policy_type,
            model_path=model_cfg.model_path,
            obs_dim=model_cfg.obs_dim,
            action_dim=model_cfg.action_dim,
            image_dim=model_cfg.image_dim,
        )
        # Choose pack/unpack strategy: schema-based (robot-agnostic) or model-specific defaults
        if getattr(cfg, "io_schema", None):
            io_schema_path = cfg.io_schema
        else:
            io_schema_path = "default_io_schema.yaml"

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

        # if getattr(cfg, "channel_config", None):
        #     schema = load_schema(cfg.channel_config.observation)
        #     obs_only = getattr(cfg, "obs_only", None)
        #     obs_unpacker = make_observation_unpacker(schema)
        #     act_packer = make_action_packer(schema)
        #
        #     # Bind optional filter via partial by wrapping in a small closure
        #     def _obs_unpack_wrapper(obs_dict, obs_keys=None, only=obs_only):
        #         return obs_unpacker(obs_dict, obs_keys=obs_keys, only=only)
        #
        #     observation_unpack_fn = _obs_unpack_wrapper
        #     action_pack_fn = act_packer
        # else:
        #     # Fall back to existing Franka-specific helpers
        #     observation_unpack_fn = franka_observation_unpacking
        #     action_pack_fn = franka_action_packing

        self.policy.to_device(device)
        self.policy.reset()
        self.policy.set_eval_mode()
        self.create_stepper(10, self.step)
        self.task_prompt = model_cfg.task_prompt

        # Inference chunking: number of actions to prefetch from the model when queue is empty
        self.n_infer_actions = getattr(model_cfg, "pred_horizon", 10)
        self._action_queue: deque[np.ndarray] = deque()

    def _on_reset(self):
        """Clear any prefetched actions when an episode ends."""
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

        obs = {}
        for k, v in ob.items():
            if k == "state":
                state = torch.from_numpy(ob["state"]).float().unsqueeze(0)  # (1, D)
                obs["state"] = torch.tensor(state, dtype=torch.float32)
            elif "image" in k:
                img = torch.from_numpy(ob[k][0].copy()).permute(2, 0, 1)  # (C, H, W)
                img = img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)
                obs[k] = torch.tensor(img.clone(), dtype=torch.float32)

        obs["task"] = [task_prompt]
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

        # Convert to tensors in the expected shapes
        obs = self.prepare_observation(obs_seq, self.task_prompt)

        # Serve one action per call. If queue is empty, prefetch n actions.
        if len(self._action_queue) == 0:
            with torch.no_grad():
                actions = self.policy.predict_n_actions(
                    obs, n_actions=self.n_infer_actions
                )
            actions = actions.detach().cpu().numpy()  # (n, action_dim)
            for i in range(actions.shape[0]):
                self._action_queue.append(actions[i])

        return self._action_queue.popleft()
