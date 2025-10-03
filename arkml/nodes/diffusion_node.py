from collections import deque
from typing import Any, Optional

import numpy as np
import torch
from arkml.core.factory import build_model
from arkml.core.policy_node import PolicyNode
from arkml.utils.utils import _image_to_tensor

from arkml.core.app_context import ArkMLContext


class DiffusionPolicyNode(PolicyNode):
    def __init__(self, device: str):
        # Create Policy
        model_cfg = ArkMLContext.cfg.get("algo").get("model")
        policy = build_model(ArkMLContext.cfg.get("algo"))
        state_dict = torch.load(model_cfg.model_path, map_location="cpu")
        policy.load_state_dict(state_dict)
        policy.eval()
        super().__init__(
            policy=policy,
            policy_name=ArkMLContext.cfg.get("node_name"),
            device=device,
        )

        self.device = device
        self.num_diffusion_iters = model_cfg.get("diffusion_steps")
        self.pred_horizon = model_cfg.get("pred_horizon")
        self.action_dim = model_cfg.get("action_dim")
        self.obs_horizon = model_cfg.get("obs_horizon")
        self.action_horizon = model_cfg.get("action_horizon")

        self.state_key = "state"

        self.scheduler = policy.build_scheduler()

        # Observation and action history queue
        self._state_history: deque[torch.Tensor] = deque(maxlen=self.obs_horizon)
        self._image_history: deque[torch.Tensor] = deque(maxlen=self.obs_horizon)
        self._action_history: deque[torch.Tensor] = deque(maxlen=self.action_horizon)

    def _on_reset(self) -> None:
        """
        Policy specific reset function.
        Returns:
            None
        """
        self._state_history.clear()
        self._image_history.clear()
        self._action_history.clear()

    def _extract_state(self, obs: dict[str, Any]) -> torch.Tensor:
        """
        Extract state from observation.
        Args:
            obs: Observation dictionary.

        Returns:
            Extracted state.
        """
        if self.state_key in obs:
            state = np.asarray(obs[self.state_key], dtype=np.float32)
        else:
            raise ValueError(f"Observation {self.state_key} not found")
        state = np.asarray(state, dtype=np.float32)[: self.action_dim]
        return torch.from_numpy(state).to(self.device)

    @staticmethod
    def _extract_image(obs: dict[str, Any]) -> torch.Tensor:
        """
        Extract image from observation.
        Args:
            obs: Observation dictionary.

        Returns:
            Extracted image.
        """
        if ArkMLContext.visual_input_features[0] not in obs:
            raise ValueError(
                f"Observation {ArkMLContext.visual_input_features[0]} not found"
            )

        image = obs[ArkMLContext.visual_input_features[0]]
        return _image_to_tensor(image)

    @staticmethod
    def _stack_history(
        history: deque[torch.Tensor], state: torch.Tensor, obs_horizon: int
    ) -> torch.Tensor:
        """
        Stack history of states.
        Args:
            history: History of states.
            state: Next state to stack.

        Returns:
            Stacked state.
        """
        if not history:
            history.append(state)
        while len(history) < obs_horizon:
            history.appendleft(history[0].clone())
        return torch.stack(list(history)[-obs_horizon:], dim=0)

    def predict(self, obs: dict[str, Any]):
        """
        Predict the next action.
        Args:
            obs: Observation dictionary.

        Returns:
            Next action.
        """
        # extract latest obs
        state_tensor = self._extract_state(obs)
        image_tensor = self._extract_image(obs)

        # build stacked histories
        state_history = self._stack_history(
            self._state_history, state_tensor, self.obs_horizon
        )
        image_history = self._stack_history(
            self._image_history, image_tensor, self.obs_horizon
        )

        # build past action history
        if self._action_history:
            past_actions = self._stack_history(
                self._action_history,
                self._action_history[-1],
                self.action_horizon,
            )
        else:
            # pad with zeros on first rollout
            past_actions = torch.zeros(
                self.action_horizon, self.action_dim, device=self.device
            )

        # assemble obs dict for policy
        obs_dict = {
            self.state_key: state_history.unsqueeze(0),  # (1, obs_horizon, state_dim)
            ArkMLContext.visual_input_features[0]: image_history.unsqueeze(
                0
            ),  # (1, obs_horizon, C,H,W)
            "past_actions": past_actions.unsqueeze(
                0
            ),  # (1, action_horizon, action_dim)
        }

        # predict
        actions = self.policy.predict(obs=obs_dict, scheduler=self.scheduler)

        action = torch.as_tensor(actions, device=self.device)

        # update action history with executed action
        self._action_history.append(action)

        return action.cpu().numpy()
