from collections import deque
from typing import Any, Optional

import numpy as np
import torch
from arkml.core.policy_node import PolicyNode

from arkml.core.factory import build_model
from arkml.utils.utils import _image_to_tensor

from ark.utils.utils import ConfigPath

from arkml.utils.schema_io import get_visual_features


class DiffusionPolicyNode(PolicyNode):
    def __init__(self, cfg, device: str):

        # Read global config
        global_config = ConfigPath(cfg.global_config).read_yaml()

        # Get camera names
        io_schema = ConfigPath(global_config["channel_config"]).read_yaml()
        self.visual_input_features = get_visual_features(
            schema=io_schema["observation"]
        )
        if len(self.visual_input_features) > 1:
            raise NotImplementedError(
                f"Diffusion policy only support one visual feature"
            )

        # Create Policy
        model_cfg = cfg.algo.model
        policy = build_model(cfg.algo)
        state_dict = torch.load(model_cfg.model_path, map_location="cpu")
        policy.load_state_dict(state_dict)
        policy.eval()
        policy.visual_input_features = self.visual_input_features

        super().__init__(
            policy=policy,
            policy_name=cfg.node_name,
            device=device,
            global_config=cfg.global_config,
        )

        self.device = device
        self.num_diffusion_iters = model_cfg.diffusion_steps
        self.pred_horizon = model_cfg.pred_horizon
        self.action_im = model_cfg.action_dim
        self.obs_horizon = model_cfg.obs_horizon

        image_cfg = getattr(model_cfg, "image_encoder", {})
        self.use_images = image_cfg.get("enabled", True)

        self.state_key = "state"

        self.scheduler = policy.build_scheduler()

        # Observation history queue
        self._state_history: deque[torch.Tensor] = deque(maxlen=self.obs_horizon)
        self._image_history: deque[torch.Tensor] = deque(maxlen=self.obs_horizon)

    def _on_reset(self):
        self._state_history.clear()
        self._image_history.clear()

    def _extract_state(self, obs: dict[str, Any]) -> Optional[torch.Tensor]:
        if self.state_key in obs:
            state = np.asarray(obs[self.state_key], dtype=np.float32)
        else:
            raise ValueError(f"Observation {self.state_key} not found")
        state = np.asarray(state, dtype=np.float32)
        return torch.from_numpy(state).to(self.device)

    def _extract_image(self, obs: dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.use_images:
            return None
        if self.visual_input_features[0] not in obs:
            return None
        image = obs[self.visual_input_features[0]]
        return _image_to_tensor(image)

    def _stack_history(
        self, history: deque[torch.Tensor], example: torch.Tensor
    ) -> torch.Tensor:
        if not history:
            history.append(example)
        while len(history) < self.obs_horizon:
            history.appendleft(history[0].clone())
        return torch.stack(list(history)[-self.obs_horizon :], dim=0)

    def predict(self, obs: dict[str, Any]):
        state_tensor = self._extract_state(obs)
        if state_tensor is not None:
            state_history = self._stack_history(self._state_history, state_tensor)
        else:
            state_history = None

        image_tensor = self._extract_image(obs)
        if image_tensor is not None:
            image_history = self._stack_history(self._image_history, image_tensor)
        else:
            image_history = None

        obs_dict: dict[str, torch.Tensor] = {}
        if state_history is not None:
            obs_dict[self.state_key] = state_history.unsqueeze(0)
        if image_history is not None:
            obs_dict[self.visual_input_features[0]] = image_history.unsqueeze(0)
        if not obs_dict:
            raise ValueError(
                "No observations were collected for diffusion policy input."
            )

        return self.policy.predict(obs=obs_dict, scheduler=self.scheduler)
