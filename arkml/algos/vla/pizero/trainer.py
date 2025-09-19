import os
from datetime import datetime
from typing import Any

import torch
from arkml.core.algorithm import Trainer
from arkml.core.policy import BasePolicy
from torch.utils.data import DataLoader
from tqdm import tqdm


class PiZeroTrainer(Trainer):
    """Initialize the PiZero trainer.

    Sets up the model on the target device, collects trainable parameters,
    configures an AdamW optimizer, and prepares mixed-precision utilities
    (bfloat16 autocast or fp16 with GradScaler) for training.

    Args:
      model: Policy/model to train.
      dataloader: Training dataloader yielding batches compatible with the model's `forward`.
      device: Device identifier (e.g., "cuda" or "cpu").
      lr : Learning rate for the AdamW optimizer.
      weight_decay: Weight decay coefficient for AdamW.
      num_epochs: Number of epochs to train.
      grad_accum : Gradient accumulation steps
      output_dir: Directory where training artifacts (e.g., checkpoints) may be written.
      use_bf16: If True, use bfloat16 autocast (no GradScaler). If False, use fp16 autocast with GradScaler.

    Attributes:
      model (BasePolicy): The model moved to the target device.
      dataloader (DataLoader): The training dataloader.
      device (str): Target device for training.
      num_epochs (int): Number of epochs to train.
      grad_accum (float): Configured gradient accumulation steps.
      output_dir (str): Base directory for checkpoints and outputs.
      trainable_params (Iterable[nn.Parameter]): Parameters to optimize.
      optimizer (torch.optim.Optimizer): Configured AdamW optimizer.
      scaler (torch.cuda.amp.GradScaler): GradScaler used when `use_bf16` is False.
      use_bf16 (bool): Whether bfloat16 autocast is enabled.
    """

    def __init__(
        self,
        model: BasePolicy,
        dataloader: DataLoader,
        device: str,
        lr: float,
        weight_decay: float,
        num_epochs: int,
        grad_accum: float,
        output_dir: str,
        use_bf16: bool,
    ):
        self.model = model.to_device(device)  # PiZeroNet
        self.dataloader = dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.grad_accum = grad_accum
        self.output_dir = output_dir
        self.trainable_params = self.model.get_trainable_params()

        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=lr, weight_decay=weight_decay
        )

        # only needed for fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
        self.use_bf16 = use_bf16

    def fit(self, *args, **kwargs) -> dict[str, Any]:
        """Run the training loop and return summary metrics.

        Args:
            *args: Unused positional arguments for interface compatibility.
            **kwargs: Unused keyword arguments for interface compatibility.

        Returns:
            A summary dictionary with:
                - global_steps (int): Total number of optimizer steps performed.
                - best_loss (float): Lowest average epoch loss observed.
                - best_ckpt (str | None): Path to the best checkpoint directory.

        Raises:
            ValueError: If ``self.output_dir`` is None or invalid (required for checkpointing).

        """

        self.model.set_train_mode()
        global_step = 0

        best_loss = float("inf")
        best_ckpt_path = None
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_dir = os.path.join(self.output_dir, date_str)
        os.makedirs(ckpt_dir, exist_ok=True)

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

                # autocast: bf16 (A100/H100) or fp16 (consumer GPUs)
                autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
                with torch.autocast("cuda", dtype=autocast_dtype):
                    out = self.model.forward(batch)
                    loss = out if torch.is_tensor(out) else out[0]

                if self.use_bf16:
                    # bf16 doesn’t need GradScaler
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    # fp16 needs GradScaler
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                global_step += 1

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            print(f"[epoch {epoch + 1}] avg_loss={avg_loss:.6f}")

            # --- Checkpoint logic ---
            if 1:  # avg_loss < best_loss: # TODO
                print(
                    f"[epoch {epoch + 1}] New best loss {avg_loss:.6f} (prev {best_loss:.6f})"
                )

                # remove old best checkpoint
                if best_ckpt_path and os.path.exists(best_ckpt_path):
                    # shutil.rmtree(best_ckpt_path)
                    print(f"[checkpoint] Removed old best checkpoint {best_ckpt_path}")

                # save new best
                best_loss = avg_loss
                best_ckpt_path = os.path.join(ckpt_dir, f"best_epoch{epoch + 1}")
                self.model.save_policy(best_ckpt_path)
                print(f"[checkpoint] Saved new best checkpoint to {best_ckpt_path}")

        return {
            "global_steps": global_step,
            "best_loss": best_loss,
            "best_ckpt": best_ckpt_path,
        }
