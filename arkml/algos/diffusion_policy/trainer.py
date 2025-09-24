import copy
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from arkml.core.algorithm import Trainer


class DiffusionTrainer(Trainer):
    STATE_KEYS = {
        "state",
        "states",
        "proprio",
        "proprioception",
        "agent_pos",
        "lowdim",
        "low_dim",
    }

    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda",
        num_epochs: int = 20000,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        num_diffusion_iters: int = 100,
        obs_horizon: int = 8,
        pred_horizon: int = 16,
        use_ema: bool = True,
        ema_power: float = 0.75,
        grad_clip: float | None = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_ema = use_ema
        self.grad_clip = grad_clip

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.ema = (
            EMAModel(parameters=self.model.parameters(), power=ema_power)
            if use_ema
            else None
        )

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_loss = float("inf")

    def _move_to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, Mapping):
            return {k: self._move_to_device(v, device) for k, v in data.items()}
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return type(data)(self._move_to_device(v, device) for v in data)
        return data

    def _reshape_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            numel = tensor.numel()
            if numel % self.obs_horizon != 0:
                raise ValueError(
                    f"Cannot reshape tensor of shape {tuple(tensor.shape)} to obs_horizon {self.obs_horizon}"
                )
            step_dim = numel // self.obs_horizon
            tensor = tensor.view(1, self.obs_horizon, step_dim)
        elif tensor.dim() == 2:
            if tensor.size(0) == self.obs_horizon and tensor.size(1) != self.obs_horizon:
                tensor = tensor.unsqueeze(0)
            elif tensor.size(1) == self.obs_horizon and tensor.size(0) != self.obs_horizon:
                tensor = tensor.unsqueeze(0).transpose(1, 2)
            elif tensor.size(1) % self.obs_horizon == 0:
                step_dim = tensor.size(1) // self.obs_horizon
                tensor = tensor.view(tensor.size(0), self.obs_horizon, step_dim)
            else:
                raise ValueError(
                    f"Cannot reshape tensor of shape {tuple(tensor.shape)} to obs_horizon {self.obs_horizon}"
                )
        elif tensor.dim() == 3:
            if tensor.size(1) == self.obs_horizon:
                pass
            elif tensor.size(2) == self.obs_horizon:
                tensor = tensor.transpose(1, 2)
            elif tensor.size(0) == self.obs_horizon and tensor.size(1) != self.obs_horizon:
                tensor = tensor.transpose(0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"Cannot reshape tensor of shape {tuple(tensor.shape)} to obs_horizon {self.obs_horizon}"
                )
        else:
            raise ValueError(
                f"Cannot reshape tensor of shape {tuple(tensor.shape)} to obs_horizon {self.obs_horizon}"
            )
        return tensor.contiguous().float()

    def _normalize_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        name = key.lower()
        if name in self.STATE_KEYS:
            return self._reshape_sequence(value.float())
        return value.float()

    def _normalize_obs_value(self, key: str, value, device):
        if isinstance(value, torch.Tensor):
            return self._normalize_tensor(key, value.to(device))
        if isinstance(value, Mapping):
            return {
                sub_key: self._normalize_obs_value(sub_key, sub_val, device)
                for sub_key, sub_val in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            return self._normalize_tensor(key, tensor)
        return value

    def _format_observation(self, obs, device):
        if isinstance(obs, torch.Tensor):
            tensor = obs.to(device, dtype=torch.float32)
            return {"state": self._reshape_sequence(tensor)}
        if isinstance(obs, Mapping):
            formatted = {
                key: self._normalize_obs_value(key, value, device)
                for key, value in obs.items()
            }
            if "state" not in formatted:
                for key, value in formatted.items():
                    if key.lower() in self.STATE_KEYS:
                        formatted["state"] = value
                        break
            return formatted
        if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            return {"state": self._reshape_sequence(tensor)}
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return {"state": self._reshape_sequence(tensor)}

    def fit(self) -> dict:
        device = torch.device(self.device)
        self.model.to(device)
        self.model.set_train_mode()
        if self.ema is not None:
            self.ema.to(device)

        history = {"loss": []}

        with tqdm(range(self.num_epochs), desc="Epoch") as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = []

                for batch in tqdm(self.dataloader, desc="Batch", leave=False):
                    actions = batch["action"].to(device)
                    raw_obs = batch.get("obs", batch.get("state"))
                    if raw_obs is None:
                        raise KeyError("Batch must contain 'obs' or 'state' entries.")
                    obs = self._format_observation(raw_obs, device)
                    B = actions.size(0)

                    noise = torch.randn_like(actions)
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                        dtype=torch.long,
                    )

                    noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

                    noise_pred = self.model(
                        noisy_actions,
                        timesteps,
                        obs=obs,
                    )
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.ema is not None:
                        self.ema.step(self.model.parameters())

                    epoch_loss.append(loss.item())

                mean_loss = float(np.mean(epoch_loss)) if epoch_loss else float("inf")
                history["loss"].append(mean_loss)
                tglobal.set_postfix({"loss": mean_loss})

                torch.save(self.model.state_dict(), "noise_pred_net_latest.pth")

                if self.ema is not None:
                    ema_model = copy.deepcopy(self.model)
                    self.ema.copy_to(ema_model.parameters())
                    torch.save(ema_model.state_dict(), "ema_noise_pred_net_latest.pth")

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    torch.save(self.model.state_dict(), "noise_pred_net_best.pth")
                    if self.ema is not None:
                        ema_model = copy.deepcopy(self.model)
                        self.ema.copy_to(ema_model.parameters())
                        torch.save(ema_model.state_dict(), "ema_noise_pred_net_best.pth")

        return history
