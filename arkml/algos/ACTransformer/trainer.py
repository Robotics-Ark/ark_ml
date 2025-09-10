
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import masked_l1, kl_loss
from contextlib import nullcontext
from ark_ml.arkml.core.algorithm import Trainer
from ark_ml.arkml.core.registry import DATASETS, MODELS

class ACTransformerTrainer(Trainer):
    def __init__(self, cfg, device="cuda"):
        self.device = device
        self.cfg = cfg

        # Build dataset
        dataset_cls = DATASETS.get(cfg.data.name)
        self.dataset = dataset_cls(**cfg.data)
        # Use a sensible num_workers if available; fall back to 0 instead of batch_size
        num_workers = getattr(self.cfg.data, "num_workers", 0)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # Build model
        model_cls = MODELS.get(cfg.algo.model.name)
        self.model = model_cls().to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay
        )
        self.reconstruction_loss = masked_l1
        self.kl_loss = kl_loss

        self.amp_ctx = (
            torch.autocast(device_type=self.device, dtype=torch.float16)
            if getattr(cfg, "amp", False) and self.device in ("cuda", "mps")
            else nullcontext()
        )

        self.scaler = None
        if getattr(cfg, "amp", False) and self.device == "cuda":
            try:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                self.scaler = None

        # --- Checkpointing state ---
        self.best_train_loss = float("inf")
        # let cfg.trainer.checkpoint_path override; default to "./best.ckpt"
        self.ckpt_path = getattr(self.cfg.trainer, "checkpoint_path", "./best.ckpt")

    def _save_checkpoint(self, epoch: int, train_loss: float):
        os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": (self.scaler.state_dict() if self.scaler is not None else None),
                "train_loss": train_loss,
                "cfg": self.cfg,
            },
            self.ckpt_path,
        )

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.cfg.trainer.epochs), desc="Epochs"):
            epoch_loss = 0.0
            step = 0

            for batch in self.dataloader:
                state = batch["state"].to(self.device)
                image = batch["image"].to(self.device)
                target = batch["action_chunk"].to(self.device)
                mask = batch["mask"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler is not None:
                    with self.amp_ctx:
                        pred, mu, logvar = self.model(image, state, target, mask)
                        reconstruction_loss = self.reconstruction_loss(pred, target, mask)
                        kl_loss_ = self.kl_loss(mu, logvar).mean()
                        loss = reconstruction_loss + (self.cfg.beta * kl_loss_)
                    self.scaler.scale(loss).backward()
                    if getattr(self.cfg, "grad_clipping", None):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clipping
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    with self.amp_ctx:
                        pred, mu, logvar = self.model(image, state, target, mask)
                        reconstruction_loss = self.reconstruction_loss(pred, target, mask)
                        kl_loss_ = self.kl_loss(mu, logvar).mean()
                        loss = reconstruction_loss + (self.cfg.beta * kl_loss_)
                    loss.backward()
                    if getattr(self.cfg, "grad_clipping", None):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clipping
                        )
                    self.optimizer.step()

                # --- accumulate epoch stats correctly inside the inner loop ---
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

            # periodic logging
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.cfg.trainer.epochs}] "
                    f"Loss: {train_loss:.6f}"
                )
