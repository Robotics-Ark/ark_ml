import copy

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from arkml.core.algorithm import Trainer


class DiffusionTrainer(Trainer):
    def __init__(
            self,
            model,
            dataloader,
            device="cuda",
            num_epochs=20000,
            lr=1e-3,
            weight_decay=1e-6,
            num_diffusion_iters=100,
            obs_horizon=8,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.obs_horizon = obs_horizon

        # Diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # EMA wrapper
        self.ema = EMAModel(parameters=self.model.parameters(), power=0.75)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_loss = float("inf")

    def fit(self) -> dict:
        device = torch.device(self.device)
        self.model.to(device)
        self.ema.to(device)

        history = {"loss": []}

        with tqdm(range(self.num_epochs), desc="Epoch") as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = []

                for batch in tqdm(self.dataloader, desc="Batch", leave=False):
                    nobs = batch["obs"].to(device)  # (B, obs_horizon, obs_dim)
                    naction = batch["action"].to(device)  # (B, pred_horizon, action_dim)
                    B = nobs.shape[0]

                    obs_cond = nobs[:, : self.obs_horizon, :].flatten(start_dim=1)

                    noise = torch.randn_like(naction)
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
                    ).long()

                    noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = self.model(noisy_actions, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # EMA update
                    self.ema.step(self.model.parameters())

                    loss_val = loss.item()
                    epoch_loss.append(loss_val)

                mean_loss = np.mean(epoch_loss)
                history["loss"].append(mean_loss)

                # checkpoints
                # always save the latest raw model
                torch.save(self.model.state_dict(), "noise_pred_net_latest.pth")

                # always save the latest EMA-smoothed weights
                ema_model = copy.deepcopy(self.model)
                self.ema.copy_to(ema_model.parameters())
                torch.save(ema_model.state_dict(), "ema_noise_pred_net_latest.pth")

                # save best checkpoints
                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss

                    # raw model best
                    torch.save(self.model.state_dict(), "noise_pred_net_best.pth")

                    # EMA best
                    ema_model = copy.deepcopy(self.model)
                    self.ema.copy_to(ema_model.parameters())
                    torch.save(ema_model.state_dict(), "ema_noise_pred_net_best.pth")

        return history
