from collections import deque
from typing import Any, Optional

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from arkml.nodes.policy_node import PolicyNode


class DiffusionPolicyNode(PolicyNode):
    def __init__(self, policy, cfg, device: str = "cuda"):
        model_cfg = cfg.algo.model
        self.scheduler = DDPMScheduler(
            num_train_timesteps=model_cfg.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        super().__init__(
            policy=policy,
            policy_name=cfg.node_name,
            device=device,
            global_config=cfg.global_config,
        )
        self.device = torch.device(device)
        self.num_diffusion_iters = model_cfg.diffusion_steps
        self.pred_horizon = model_cfg.pred_horizon
        self.action_dim = model_cfg.action_dim
        self.obs_horizon = model_cfg.obs_horizon

        image_cfg = getattr(model_cfg, "image_encoder", {})
        self.use_images = image_cfg.get("enabled", True)
        self.image_key = image_cfg.get("key", "images")

        proprio_cfg = getattr(model_cfg, "proprio_encoder", {})
        self.use_state = proprio_cfg.get("enabled", True)
        self.state_key = proprio_cfg.get("key", "state")

        self._state_history: deque[torch.Tensor] = deque(maxlen=self.obs_horizon)
        self._image_history: deque[torch.Tensor] = deque(maxlen=self.obs_horizon)

    def _on_reset(self):
        self._state_history.clear()
        self._image_history.clear()

    def _legacy_flatten_state(self, obs: dict[str, Any]) -> np.ndarray:
        cube = np.asarray(obs.get("cube", [0, 0, 0]), dtype=np.float32).reshape(-1)[:3]
        target = np.asarray(obs.get("target", [0, 0, 0]), dtype=np.float32).reshape(-1)[:3]
        grip = np.asarray(obs.get("gripper", [0.0]), dtype=np.float32).reshape(-1)[:1]
        ee_pos, _ee_quat = obs.get("franka_ee", ([0, 0, 0], [0, 0, 0, 1]))
        ee = np.asarray(ee_pos, dtype=np.float32).reshape(-1)[:3]
        vec = np.concatenate([cube, target, grip, ee], axis=0)
        if vec.shape[0] < 10:
            vec = np.pad(vec, (0, 10 - vec.shape[0]))
        return vec[:10]

    def _extract_state(self, obs: dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.use_state:
            return None
        if self.state_key in obs:
            state = np.asarray(obs[self.state_key], dtype=np.float32)
        else:
            state = self._legacy_flatten_state(obs)
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 2:
            # assume (T, D), take last
            state = state[-1]
        return torch.from_numpy(state).to(self.device)

    def _extract_image(self, obs: dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.use_images:
            return None
        if self.image_key not in obs:
            return None
        image = obs[self.image_key]
        if isinstance(image, dict):
            image = image.get("rgb") or next(iter(image.values()))
        image = np.asarray(image)
        if image.ndim == 4:
            # Assume (T, H, W, C) and take last frame
            image = image[-1]
        if image.ndim == 3 and image.shape[0] not in {1, 3} and image.shape[-1] in {1, 3}:
            image = np.transpose(image, (2, 0, 1))
        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape {image.shape}")
        image_tensor = torch.from_numpy(image).float().to(self.device)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        return image_tensor

    def _stack_history(self, history: deque[torch.Tensor], example: torch.Tensor) -> torch.Tensor:
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
            obs_dict["state"] = state_history.unsqueeze(0)
        if image_history is not None:
            obs_dict["images"] = image_history.unsqueeze(0)
        if not obs_dict:
            raise ValueError("No observations were collected for diffusion policy input.")

        self.scheduler.set_timesteps(self.num_diffusion_iters)
        actions = self.policy.sample_actions(
            obs_dict,
            scheduler=self.scheduler,
            num_inference_steps=self.num_diffusion_iters,
        )
        return actions.squeeze(0).detach().cpu().numpy()
