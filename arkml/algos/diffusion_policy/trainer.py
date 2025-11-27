"""Trainer for diffusion policies with validation-aware checkpointing."""

import copy
import json
from pathlib import Path
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
from arkml.utils.stats import (
    _init_state,
    _accumulate_moments,
    _finalize_stats,
    sample_indices,
)


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
        max_steps: int | None = None,
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
        self.max_steps = int(max_steps) if max_steps is not None else None

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

    def _ensure_dataset_stats(self) -> None:
        """Compute dataset stats (state/action) and set on the model if unset.

        This samples a subset of the training dataset to compute mean/std for
        states and actions and initializes the model's normalization buffers via
        `set_normalization_stats`.

        If buffers are already non-default (i.e., not zeros/ones), this will
        still overwrite them to ensure consistency with the current dataset.
        """
        ds = getattr(self.dataloader, "dataset", None)
        if ds is None:
            return

        try:
            # Try loading stats from JSON first, if available
            dataset_path = getattr(ds, "dataset_path", None)
            stats_path: Path | None = None
            if dataset_path is not None:
                stats_path = Path(dataset_path) / "diffusion_stats.json"
            if stats_path is not None and stats_path.exists():
                with open(stats_path, "r") as f:
                    raw = json.load(f)

                def _to_np(key_aliases: list[str]):
                    for k in key_aliases:
                        if k in raw and isinstance(raw[k], dict):
                            d = raw[k]
                            if "mean" in d and "std" in d:
                                return np.asarray(d["mean"]).astype(np.float32), np.asarray(
                                    d["std"]
                                ).astype(np.float32)
                    return None, None

                s_mean, s_std = _to_np(["state", "observation.state"])
                a_mean, a_std = _to_np(["action"])

                if s_mean is not None and s_std is not None and a_mean is not None and a_std is not None:
                    self.model.set_normalization_stats(
                        action_mean=torch.from_numpy(a_mean).to(self.device, dtype=torch.float32),
                        action_std=torch.from_numpy(a_std).to(self.device, dtype=torch.float32),
                        state_mean=torch.from_numpy(s_mean).to(self.device, dtype=torch.float32),
                        state_std=torch.from_numpy(s_std).to(self.device, dtype=torch.float32),
                    )
                    print(
                        f"[DiffusionTrainer] Loaded normalization stats from {stats_path}"
                    )
                    return

            # Sample indices across the dataset for a reasonable estimate
            indices = sample_indices(len(ds))

            # Peek shapes from the first sample
            first = ds[indices[0]]
            state_dim = int(np.asarray(first["state"]).shape[-1])
            action = np.asarray(first["action"], dtype=np.float32)
            action_dim = int(action.shape[-1])

            s_state = _init_state((state_dim,))
            s_action = _init_state((action_dim,))

            for i in indices:
                sample = ds[i]
                # State: (D,)
                st = np.asarray(sample["state"], dtype=np.float32)
                if st.ndim != 1:
                    st = st.reshape(-1)
                _accumulate_moments(st[None, :], s_state)

                # Action window: (T, D) -> collapse time
                act = np.asarray(sample["action"], dtype=np.float32)
                act = act.reshape(-1, action_dim)
                _accumulate_moments(act, s_action)

            f_state = _finalize_stats(s_state)
            f_action = _finalize_stats(s_action)

            # Set on model (broadcasted in model)
            self.model.set_normalization_stats(
                action_mean=torch.from_numpy(f_action["mean"]).to(self.device, dtype=torch.float32),
                action_std=torch.from_numpy(f_action["std"]).to(self.device, dtype=torch.float32),
                state_mean=torch.from_numpy(f_state["mean"]).to(self.device, dtype=torch.float32),
                state_std=torch.from_numpy(f_state["std"]).to(self.device, dtype=torch.float32),
            )
            print(
                f"[DiffusionTrainer] Set normalization stats: state_dim={state_dim}, action_dim={action_dim}, "
                f"samples={len(indices)}"
            )

            # Persist stats for reuse if dataset path is known
            if stats_path is not None:
                try:
                    payload = {
                        "state": {
                            "mean": f_state["mean"].tolist(),
                            "std": f_state["std"].tolist(),
                        },
                        "action": {
                            "mean": f_action["mean"].tolist(),
                            "std": f_action["std"].tolist(),
                        },
                    }
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(stats_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    print(f"[DiffusionTrainer] Saved normalization stats to {stats_path}")
                except Exception as e:
                    print(
                        f"[DiffusionTrainer] Warning: failed to save stats to {stats_path} ({e})"
                    )
        except Exception as e:
            print(f"[DiffusionTrainer] Warning: failed to compute dataset stats ({e})")

    def save_checkpoint(
        self,
        ckpt_dir: str,
        epoch_idx: int,
        metric_value: float,
        metric_name: str,
    ) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)

        def save_model(model, prefix: str) -> str:
            filename = f"{prefix}.pth"
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
        # Compute and set dataset stats before training
        self._ensure_dataset_stats()
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
                steps_done = 0

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
                    steps_done += 1
                    if self.max_steps is not None and steps_done >= self.max_steps:
                        # Save checkpoint at the boundary and exit
                        mean_loss = (
                            float(np.mean(batch_losses)) if batch_losses else float("inf")
                        )
                        self.save_checkpoint(
                            ckpt_dir=ckpt_dir,
                            epoch_idx=epoch_idx,
                            metric_value=mean_loss,
                            metric_name="train_loss",
                        )
                        return {
                            "best_metric_name": self.best_metric_name,
                            "best_metric": self.best_metric,
                            "best_ckpt": self.best_ckpt_path,
                            "checkpoint_dir": ckpt_dir,
                            "stopped_at_steps": steps_done,
                        }

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
