"""Trainer for diffusion policies with validation-aware checkpointing."""

import copy
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from arkml.core.algorithm import Trainer
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .evaluator import DiffusionPolicyEvaluator


class DiffusionTrainer(Trainer):

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
        *,
        val_dataloader: DataLoader | None = None,
        eval_every: int = 1,
    ):
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.eval_every = max(1, int(eval_every))
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_ema = use_ema
        self.grad_clip = grad_clip
        self.output_dir = output_dir

        self.ema = (
            EMAModel(parameters=self.model.parameters(), power=ema_power)
            if use_ema
            else None
        )

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_metric = float("inf")
        self.best_metric_name = "train_loss"
        self.best_ckpt_path: str | None = None

    def save_checkpoint(
        self,
        ckpt_dir: str,
        epoch_idx: int,
        metric_value: float,
        metric_name: str,
    ) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)

        def save_model(model, prefix: str) -> str:
            filename = f"{prefix}{epoch_idx + 1}.pth"
            path = os.path.join(ckpt_dir, filename)
            torch.save(model.state_dict(), path)
            return path

        save_model(self.model, "noise_pred_net_latest")

        ema_model = None
        if self.ema is not None:
            ema_model = copy.deepcopy(self.model)
            self.ema.copy_to(ema_model.parameters())
            save_model(ema_model, "ema_noise_pred_net_latest")

        if metric_value < self.best_metric:
            best_prev = self.best_metric
            self.best_metric = metric_value
            self.best_metric_name = metric_name
            self.best_ckpt_path = save_model(self.model, "noise_pred_net_best")
            if ema_model is not None:
                save_model(ema_model, "ema_noise_pred_net_best")
            print(
                f"[checkpoint] New best {metric_name}: {metric_value:.6f} (prev {best_prev:.6f})"
            )

    def fit(self) -> dict[str, Any]:
        """
        Run the training loop and return summary metrics.

        Returns:
            Training summary.
        """
        # Enable cuDNN benchmark to speed up convs for fixed shapes
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        self.model.to(self.device)
        self.model.set_train_mode()
        if self.ema is not None:
            self.ema.to(self.device)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_dir = os.path.join(self.output_dir, date_str)
        os.makedirs(ckpt_dir, exist_ok=True)

        scheduler = self.model.build_scheduler()

        with tqdm(range(self.num_epochs), desc="Epoch") as epoch_bar:
            for epoch_idx in epoch_bar:
                batch_losses = []

                for batch in tqdm(self.dataloader, desc="Batch", leave=False):
                    noise_pred, loss = self.model(obs=batch, scheduler=scheduler)
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

                val_loss = None
                if self.val_dataloader is not None and (
                    (epoch_idx + 1) % self.eval_every == 0
                ):
                    self.model.set_eval_mode()
                    evaluator = DiffusionPolicyEvaluator(
                        model=self.model,
                        dataloader=self.val_dataloader,
                        device=self.device,
                    )
                    val_metrics = evaluator.evaluate(self.val_dataloader)
                    val_loss = float(val_metrics.get("mse_loss", float("inf")))
                    self.model.set_train_mode()

                metric_value = val_loss if val_loss is not None else mean_loss
                metric_name = "val_loss" if val_loss is not None else "train_loss"

                postfix = {"train_loss": mean_loss}
                if val_loss is not None:
                    postfix["val_loss"] = val_loss
                epoch_bar.set_postfix(postfix)

                self.save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    epoch_idx=epoch_idx,
                    metric_value=metric_value,
                    metric_name=metric_name,
                )

        return {
            "best_metric_name": self.best_metric_name,
            "best_metric": self.best_metric,
            "best_ckpt": self.best_ckpt_path,
            "checkpoint_dir": ckpt_dir,
        }
