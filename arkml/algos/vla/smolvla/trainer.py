import os
from datetime import datetime
from typing import Any

import torch
from contextlib import nullcontext
from arkml.core.algorithm import Trainer
from arkml.core.policy import BasePolicy
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluator import smolVLAEvaluator


class smolVLATrainer(Trainer):
    """Initialize the smovla trainer.

    Args:
      model: Policy/model to train.
      dataloader: Training dataloader yielding batches compatible with the model's `forward`.
      device: Device identifier or torch.device (e.g., "cuda", "cpu").
      lr: Learning rate for the AdamW optimizer.
      weight_decay: Weight decay coefficient for AdamW.
      num_epochs: Number of epochs to train.
      grad_accum: Gradient accumulation steps.
      output_dir: Directory where training artifacts (checkpoints) are written.
      use_bf16: If True, use bfloat16 autocast; otherwise use fp16 autocast (CUDA only).
      val_dataloader: Optional validation dataloader for periodic evaluation.
      eval_every: Run evaluation every N epochs (default: 1).
    """

    def __init__(
        self,
        model: BasePolicy,
        dataloader: DataLoader,
        device: Any,
        lr: float,
        weight_decay: float,
        num_epochs: int,
        grad_accum: float,
        output_dir: str,
        use_bf16: bool,
        *,
        val_dataloader: DataLoader | None = None,
        eval_every: int = 1,
    ):
        self.model = model.to_device(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.eval_every = max(1, int(eval_every))
        self.device = device
        self.num_epochs = num_epochs
        self.grad_accum = max(1, int(grad_accum))
        self.output_dir = output_dir
        self.trainable_params = self.model.get_trainable_params()

        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=lr, weight_decay=weight_decay
        )

        # Device/AMP setup
        device_str = str(device)
        self.device_type = (
            "cuda"
            if torch.cuda.is_available()
            and (device_str.startswith("cuda") or getattr(device, "type", "") == "cuda")
            else "cpu"
        )
        self.use_bf16 = use_bf16
        # GradScaler only for CUDA fp16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device_type == "cuda" and not self.use_bf16)
        )

    def fit(self, *args, **kwargs) -> dict[str, Any]:
        """
        Run the training loop and return summary metrics.

        Returns:
            A summary dictionary with:
              - global_steps: Optimizer step count
              - best_metric_name: "val_loss" if val loader provided, else "train_loss"
              - best_loss: Best metric value observed
              - best_ckpt: Path to best checkpoint directory
        """

        self.model.set_train_mode()
        global_steps = 0
        best_metric = float("inf")
        best_ckpt_path = None

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_dir = os.path.join(self.output_dir, date_str)
        os.makedirs(ckpt_dir, exist_ok=True)

        best_metric_name = (
            "val_loss" if self.val_dataloader is not None else "train_loss"
        )

        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0.0
            num_batches = 0

            self.optimizer.zero_grad(set_to_none=True)

            progress_bar = tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                leave=False,
            )

            for i, batch in progress_bar:
                # Choose autocast context
                if self.device_type == "cuda":
                    ac_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
                    ac = torch.autocast("cuda", dtype=ac_dtype)
                else:
                    ac = (
                        torch.autocast("cpu", dtype=torch.bfloat16)
                        if self.use_bf16
                        else nullcontext()
                    )

                with ac:
                    out = self.model.forward(batch)
                    loss = out if torch.is_tensor(out) else out[0]

                # Gradient accumulation
                loss_to_backprop = loss / self.grad_accum

                if self.device_type == "cuda" and not self.use_bf16:
                    self.scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

                step_now = ((i + 1) % self.grad_accum == 0) or (
                    i + 1 == len(self.dataloader)
                )
                if step_now:
                    if self.device_type == "cuda" and not self.use_bf16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.trainable_params, max_norm=1.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.trainable_params, max_norm=1.0
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    global_steps += 1

                epoch_loss += float(loss.item())
                num_batches += 1

            avg_train_loss = epoch_loss / max(1, num_batches)
            print(f"[epoch {epoch + 1}] train_loss={avg_train_loss:.6f}")

            # Optional validation
            current_metric = avg_train_loss
            if self.val_dataloader is not None and ((epoch + 1) % self.eval_every == 0):
                self.model.set_eval_mode()
                evaluator = smolVLAEvaluator(
                    model=self.model, dataloader=self.val_dataloader, device=self.device
                )
                val_metrics = evaluator.evaluate()
                val_loss = float(val_metrics.get("val_loss", float("inf")))
                print(
                    f"[epoch {epoch + 1}] val_loss={val_loss:.6f} over {val_metrics.get('batches', 0)} batches"
                )
                current_metric = val_loss
                self.model.set_train_mode()

            # Save best checkpoint
            if current_metric < best_metric:
                print(
                    f"[epoch {epoch + 1}] New best {best_metric_name} {current_metric:.6f} (prev {best_metric:.6f})"
                )
                best_metric = current_metric
                best_ckpt_path = os.path.join(ckpt_dir, f"best_epoch{epoch + 1}")
                self.model.save_policy(best_ckpt_path)
                print(f"[checkpoint] Saved new best checkpoint to {best_ckpt_path}")

        return {
            "global_steps": global_steps,
            "best_metric_name": best_metric_name,
            "best_loss": best_metric,
            "best_ckpt": best_ckpt_path,
        }
