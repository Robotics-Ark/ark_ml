from collections import deque
from typing import Any

import numpy as np
import torch
from arkml.algos.vla.pizero.models import PiZeroNet
from arkml.core.policy_node import PolicyNode

from arkml.utils.franka_utils import observation_unpacking, action_packing


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
        super().__init__(
            policy=policy,
            device=device,
            policy_name=cfg.policy_node_name,
            observation_unpacking=observation_unpacking,
            action_packing=action_packing,
            stepper_frequency=cfg.stepper_frequency,
            global_config=cfg.global_config,
        )

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

    def prepare_observation(self, ob: dict[str, Any]):
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
        # ---- Image ----
        img = torch.from_numpy(ob["images"][0].copy()).permute(2, 0, 1)  # (C, H, W)
        img = img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)

        # ---- State ----
        state = torch.from_numpy(ob["state"]).float().unsqueeze(0)  # (1, D)

        return {
            "image": img,
            "state": state,
        }

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
        obs_seq = self.prepare_observation(obs_seq)
        obs = {
            "image": torch.tensor(obs_seq["image"], dtype=torch.float32),
            "state": torch.tensor(obs_seq["state"], dtype=torch.float32),
            "task": [self.task_prompt],
        }

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
