import os
from contextlib import nullcontext

import torch
from tqdm import tqdm

from arkml.core.algorithm import Trainer

def masked_l1(pred, target, mask):
    diff = (pred - target).abs()  # (B,K,A)
    m = mask.unsqueeze(-1)
    num = (diff * m).sum()
    den = (m.sum() * pred.size(-1)).clamp_min(1.0)
    return num / den


def kl_loss(mu, logvar):
    # KL(q||p), p=N(0,I)
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # (B,)


class ACTransformerTrainer(Trainer):
    def __init__(
        self,
        model,
        dataloader,
        epochs: int = 1,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        beta: float = 10.0,
        device="cuda",
    ):
        self.device = device
        self.epochs = epochs
        self.beta = beta

        self.dataloader = dataloader

        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.reconstruction_loss = masked_l1
        self.kl_loss = kl_loss
        self.grad_clip = grad_clip

        self.amp_ctx = (
            torch.autocast(device_type=self.device, dtype=torch.float16)
            if self.device in ("cuda", "mps")
            else nullcontext()
        )

        self.scaler = None
        if self.device == "cuda":
            try:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                self.scaler = None

        self.best_train_loss = float("inf")
        self.ckpt_path = "./ckpt_path_act/"

    def _save_checkpoint(self, epoch: int, train_loss: float):
        os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": (
                    self.scaler.state_dict() if self.scaler is not None else None
                ),
                "train_loss": train_loss,
            },
            self.ckpt_path,
        )

    def fit(self):
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            epoch_loss = 0.0
            step = 0

            for batch in self.dataloader:
                state = batch["state"].to(self.device)
                image = batch["image"].to(self.device)
                target = batch["action_chunk"].to(self.device)
                mask = batch["action_mask"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler is not None:
                    with self.amp_ctx:
                        pred, mu, logvar = self.model(image, state, target, mask)
                        reconstruction_loss = self.reconstruction_loss(
                            pred, target, mask
                        )
                        kl_loss_ = self.kl_loss(mu, logvar).mean()
                        loss = reconstruction_loss + (self.beta * kl_loss_)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    with self.amp_ctx:
                        pred, mu, logvar = self.model(image, state, target, mask)
                        reconstruction_loss = self.reconstruction_loss(
                            pred, target, mask
                        )
                        kl_loss_ = self.kl_loss(mu, logvar).mean()
                        loss = reconstruction_loss + (self.beta * kl_loss_)
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()

                epoch_loss += loss.item()
                step += 1

            # average train loss for the epoch
            train_loss = epoch_loss / max(1, step)

            # --- save best checkpoint if improved ---
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self._save_checkpoint(epoch=epoch + 1, train_loss=train_loss)
                print(
                    f"[BEST] Epoch {epoch + 1}: train_loss improved to {train_loss:.6f}. "
                    f"Saved checkpoint to {self.ckpt_path}."
                )

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}] " f"Loss: {train_loss:.6f}")
