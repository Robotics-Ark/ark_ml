from collections import deque
from typing import Any

import numpy as np
import torch
from arkml.algos.vla.smolvla.models import smolVLAnet
from arkml.core.app_context import ArkMLContext
from arkml.core.policy_node import PolicyNode
from arkml.utils.utils import _image_to_tensor
from arktypes import string_t


class smolVLApolicynode(PolicyNode):
    """Wrapper node for smolvla"""

    def __init__(self, device: str):
        """
        Initialize smolVLA
        Args:
            device: Device to use
        """
        model_cfg = ArkMLContext.cfg.get("algo").get("model")

        policy = smolVLAnet(
            policy_type=model_cfg.get("policy_type"),
            model_path=model_cfg.get("model_path"),
            obs_dim=model_cfg.get("obs_dim"),
            action_dim=model_cfg.get("action_dim"),
            image_dim=model_cfg.get("image_dim"),
        )

        super().__init__(
            policy=policy,
            device=device,
            policy_name=ArkMLContext.cfg.get("node_name"),
        )

        # Listen to text prompt channel
        channel_name = ArkMLContext.global_config.get("channel", "user_input")
        self.text_input = None
        self.sub = self.create_subscriber(
            channel_name, string_t, self._callback_text_input
        )

        self.policy.to_device(device)
        self.policy.reset()
        self.policy.set_eval_mode()

        self.n_infer_actions = getattr(model_cfg, "pred_horizon", 1)
        self._action_queue: deque[np.ndarray] = deque()

    def _on_reset(self) -> None:
        """
        Policy specific reset function.

        Returns:
            None
        """
        self.policy.reset()

    def _callback_text_input(
        self, time_stamp: int, channel_name: str, msg: string_t
    ) -> None:
        """
        Service callback to read text prompt.
        Args:
            time_stamp: Callback time
            channel_name: Service channel id.
            msg: Message

        Returns:
            None
        """
        self.text_input = msg.data

    def prepare_observation(self, ob: dict[str, Any]):
        """Convert a single raw env observation into a batched policy input.

        Args:
          ob: Single observation dict from the env. Expected keys include
            ``state`` and any camera names listed in ``visual_input_features``.

        Returns:
          A batch dictionary with:
            - per-camera image tensors: ``torch.FloatTensor`` of shape ``[1, C, H, W]``.
            - ``state``: ``torch.FloatTensor`` of shape ``[1, D]`` if present.
            - ``task``: ``list[str]`` of length 1.
        """
        if self.text_input is None:
            raise ValueError("Prompt input is empty")
        obs = {"task": [self.text_input]}

        state = np.concatenate(
            [
                np.ravel(ob["proprio::pose::position"]),
                np.ravel(ob["proprio::pose::orientation"]),
                np.ravel([ob["proprio::joint_state::position"][-2:]]),
            ]
        )
        state = torch.from_numpy(state).float().unsqueeze(0)  # (1, D)
        img = torch.from_numpy(ob["sensors::image_top::rgb"].copy()).permute(
            2, 0, 1
        )  # (C, H, W)
        img = img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)

        obs["state"] = state
        #
        # # State: tensor, ensure [1, D] float32
        # state_value = ob.get("state")
        # if state_value is not None:
        #     if isinstance(state_value, torch.Tensor):
        #         state_t = state_value
        #     else:
        #         state_t = torch.from_numpy(state_value)
        #     if state_t.dim() == 1:
        #         state_t = state_t.unsqueeze(0)
        #     obs["state"] = state_t.to(dtype=torch.float32, copy=False)

        # Images:  tensor, ensure [1, C, H, W]
        for cam_name in ArkMLContext.visual_input_features:
            # value = ob.get(cam_name)
            # if value is None:
            #     raise KeyError(f"Missing visual input '{cam_name}' in observation")
            obs[cam_name] = img  # _image_to_tensor(value).unsqueeze(0)
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

        obs = self.prepare_observation(obs_seq)

        with torch.no_grad():
            actions = self.policy.predict(obs, n_actions=self.n_infer_actions)
            actions = actions.detach().cpu().numpy()

        return actions[0]
