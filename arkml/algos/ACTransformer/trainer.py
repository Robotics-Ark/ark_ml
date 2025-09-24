import os
import shutil
from contextlib import nullcontext
import torch.nn as nn
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

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

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
        count_parameters(model)

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

        # --- new: consistent checkpoint directory ---
        self.ckpt_dir = "./model_checkpoints_act_run4"
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _checkpoint_payload(self, epoch: int, train_loss: float):
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.scaler is not None else None
            ),
            "train_loss": train_loss,
            "beta": self.beta,
            "grad_clip": self.grad_clip,
            "epochs": self.epochs,
            "device": self.device,
        }

    def _save_checkpoint(self, epoch: int, train_loss: float, is_best: bool):
        # save every epoch
        epoch_path = os.path.join(self.ckpt_dir, f"epoch_{epoch:04d}.pt")
        torch.save(self._checkpoint_payload(epoch, train_loss), epoch_path)

        # update best.pt if improved
        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            # copy ensures compatibility across OSes without symlink permissions
            shutil.copyfile(epoch_path, best_path)

        # also keep/refresh a last.pt pointer for convenience
        last_path = os.path.join(self.ckpt_dir, "last.pt")
        shutil.copyfile(epoch_path, last_path)

        return epoch_path

    def fit(self):
        self.model.train()
        for epoch in tqdm(range(1, self.epochs + 1), desc="Epochs"):
            epoch_loss = 0.0
            step = 0

            for batch in tqdm(self.dataloader, desc="dl"):
                state = batch["state"].to(self.device)
                image = batch["image"].to(self.device)
                target = batch["action_chunk"].to(self.device)
                mask = batch["action_mask"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler is not None:
                    with self.amp_ctx:
                        pred, mu, logvar = self.model(image, state, target, mask)
                        reconstruction_loss = self.reconstruction_loss(pred, target, mask)
                        kl_loss_ = self.kl_loss(mu, logvar).mean()
                        loss = reconstruction_loss + (self.beta * kl_loss_)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    with self.amp_ctx:
                        pred, mu, logvar = self.model(image, state, target, mask)
                        reconstruction_loss = self.reconstruction_loss(pred, target, mask)
                        kl_loss_ = self.kl_loss(mu, logvar).mean()
                        loss = reconstruction_loss + (self.beta * kl_loss_)
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                epoch_loss += loss.item()
                step += 1

            # average train loss for the epoch
            train_loss = epoch_loss / max(1, step)

            # save every epoch + update best
            is_best = train_loss < self.best_train_loss
            epoch_path = self._save_checkpoint(epoch=epoch, train_loss=train_loss, is_best=is_best)

            if is_best:
                self.best_train_loss = train_loss
                print(f"[BEST] Epoch {epoch}: train_loss improved to {train_loss:.6f}. "
                      f"Saved epoch to {epoch_path} and updated best.pt.")

            # periodic logging (optional)
            if (epoch) % 10 == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch [{epoch}/{self.epochs}] Loss: {train_loss:.6f}")
