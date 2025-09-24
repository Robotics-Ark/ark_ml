from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from arkml.core.algorithm import Evaluator


class DiffusionEvaluator(Evaluator):
    STATE_KEYS = {
        "state",
        "states",
        "proprio",
        "proprioception",
        "agent_pos",
        "lowdim",
        "low_dim",
    }

    def __init__(self, model, dataloader, device: str = "cpu"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.model.set_eval_mode()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=model.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def _move_to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, Mapping):
            return {k: self._move_to_device(v, device) for k, v in data.items()}
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return type(data)(self._move_to_device(v, device) for v in data)
        return data

    def _reshape_sequence(self, tensor: torch.Tensor, obs_horizon: int) -> torch.Tensor:
        if tensor.dim() == 1:
            numel = tensor.numel()
            if numel % obs_horizon != 0:
                raise ValueError(
                    f"Cannot reshape tensor of shape {tuple(tensor.shape)} to obs_horizon {obs_horizon}"
                )
            step_dim = numel // obs_horizon
            tensor = tensor.view(1, obs_horizon, step_dim)
        elif tensor.dim() == 2:
            if tensor.size(0) == obs_horizon and tensor.size(1) != obs_horizon:
                tensor = tensor.unsqueeze(0)
            elif tensor.size(1) == obs_horizon and tensor.size(0) != obs_horizon:
                tensor = tensor.unsqueeze(0).transpose(1, 2)
            elif tensor.size(1) % obs_horizon == 0:
                step_dim = tensor.size(1) // obs_horizon
                tensor = tensor.view(tensor.size(0), obs_horizon, step_dim)
            else:
                raise ValueError(
                    f"Cannot reshape tensor of shape {tuple(tensor.shape)} to obs_horizon {obs_horizon}"
                )
        elif tensor.dim() == 3:
            if tensor.size(1) == obs_horizon:
                pass
            elif tensor.size(2) == obs_horizon:
                tensor = tensor.transpose(1, 2)
            elif tensor.size(0) == obs_horizon and tensor.size(1) != obs_horizon:
                tensor = tensor.transpose(0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"Cannot reshape tensor of shape {tuple(tensor.shape)} to obs_horizon {obs_horizon}"
                )
        else:
            raise ValueError(
                f"Cannot reshape tensor of shape {tuple(tensor.shape)} to obs_horizon {obs_horizon}"
            )
        return tensor.contiguous().float()

    def _normalize_tensor(self, key: str, value: torch.Tensor, obs_horizon: int) -> torch.Tensor:
        if key.lower() in self.STATE_KEYS:
            return self._reshape_sequence(value.float(), obs_horizon)
        return value.float()

    def _normalize_obs_value(self, key: str, value, device, obs_horizon: int):
        if isinstance(value, torch.Tensor):
            return self._normalize_tensor(key, value.to(device), obs_horizon)
        if isinstance(value, Mapping):
            return {
                sub_key: self._normalize_obs_value(sub_key, sub_val, device, obs_horizon)
                for sub_key, sub_val in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            return self._normalize_tensor(key, tensor, obs_horizon)
        return value

    def _format_observation(self, obs, device, obs_horizon: int):
        if isinstance(obs, torch.Tensor):
            tensor = obs.to(device, dtype=torch.float32)
            return {"state": self._reshape_sequence(tensor, obs_horizon)}
        if isinstance(obs, Mapping):
            formatted = {
                key: self._normalize_obs_value(key, value, device, obs_horizon)
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
            return {"state": self._reshape_sequence(tensor, obs_horizon)}
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return {"state": self._reshape_sequence(tensor, obs_horizon)}

    def evaluate(self, dataloader):
        total_loss = 0.0
        n = 0
        device = torch.device(self.device)
        obs_horizon = getattr(self.model, "obs_horizon", 1)
        with torch.no_grad():
            for batch in dataloader:
                actions = batch["action"].to(device)
                raw_obs = batch.get("obs", batch.get("state"))
                if raw_obs is None:
                    raise KeyError("Batch must contain 'obs' or 'state' entries.")
                obs = self._format_observation(raw_obs, device, obs_horizon)
                noise = torch.randn_like(actions)
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (actions.size(0),),
                    device=device,
                    dtype=torch.long,
                )
                noisy_action = self.noise_scheduler.add_noise(actions, noise, timesteps)
                pred_noise = self.model(noisy_action, timesteps, obs=obs)
                loss = F.mse_loss(pred_noise, noise, reduction="sum")
                total_loss += loss.item()
                n += actions.shape[0]
        return {"mse_loss": total_loss / max(n, 1)}
