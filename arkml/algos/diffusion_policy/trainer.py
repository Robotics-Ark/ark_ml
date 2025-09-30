import copy
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from arkml.core.algorithm import Trainer
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm


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
        output_dir: str,
        device: str = "cuda",
        num_epochs: int = 20000,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        obs_horizon: int = 9,
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
        self.output_dir = output_dir

        self.noise_scheduler = self.model.build_scheduler()

        self.ema = (
            EMAModel(parameters=self.model.parameters(), power=ema_power)
            if use_ema
            else None
        )

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_loss = float("inf")

    def save_checkpoint(self, ckpt_dir, epoch_idx, mean_loss):
        os.makedirs(ckpt_dir, exist_ok=True)  # ensure directory exists

        def save_model(model, prefix):
            filename = f"{prefix}{epoch_idx + 1}.pth"
            path = os.path.join(ckpt_dir, filename)
            torch.save(model.state_dict(), path)

        # Always save "latest"
        save_model(self.model, "noise_pred_net_latest")

        ema_model = None
        if self.ema is not None:
            ema_model = copy.deepcopy(self.model)
            self.ema.copy_to(ema_model.parameters())
            save_model(ema_model, "ema_noise_pred_net_latest")

        # Save "best" if loss improved
        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            save_model(self.model, "noise_pred_net_best")

            if ema_model is not None:  # reuse already built ema_model
                save_model(ema_model, "ema_noise_pred_net_best")

    def fit(self) -> dict:
        self.model.to(self.device)
        self.model.set_train_mode()
        if self.ema is not None:
            self.ema.to(self.device)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_dir = os.path.join(self.output_dir, date_str)
        os.makedirs(ckpt_dir, exist_ok=True)

        history = {"loss": []}

        with tqdm(range(self.num_epochs), desc="Epoch") as epoch_bar:
            for epoch_idx in epoch_bar:
                batch_losses = []

                for batch in tqdm(self.dataloader, desc="Batch", leave=False):
                    actions = batch["action"].to(self.device)
                    batch_size = actions.size(0)

                    # Sample noise and time steps
                    noise = torch.randn_like(actions)
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (batch_size,),
                        device=self.device,
                        dtype=torch.long,
                    )

                    # Forward pass
                    noisy_actions = self.noise_scheduler.add_noise(
                        actions, noise, timesteps
                    )
                    noise_pred = self.model(noisy_actions, timesteps, obs=batch)
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # Backward pass
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )

                    self.optimizer.step()

                    if self.ema is not None:
                        self.ema.step(self.model.parameters())

                    batch_losses.append(loss.item())

                mean_loss = (
                    float(np.mean(batch_losses)) if batch_losses else float("inf")
                )
                history["loss"].append(mean_loss)
                epoch_bar.set_postfix({"loss": mean_loss})

                # Save checkpoints
                self.save_checkpoint(
                    ckpt_dir=ckpt_dir, epoch_idx=epoch_idx, mean_loss=mean_loss
                )

        return history
