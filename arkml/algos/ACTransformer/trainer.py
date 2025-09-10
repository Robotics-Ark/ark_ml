from math import degrees

import torch
import torch.nn as nn
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
        self.dataloader = DataLoader(self.dataset, batch_size=self.cfg.data.batch_size, shuffle=True, num_workers=self.cfg.trainer.batch_size)

        # Building the policy model
        model_cls = MODELS.get(cfg.algo.model.name)
        self.model = model_cls() # TODO: model classes

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.trainer.lr, weight_decay=self.cfg.trainer.weight_decay)
        self.reconstruction_loss = masked_l1
        self.kl_loss = kl_loss

        self.amp_ctx = (
            torch.autocast(device_type=self.device, dtype=torch.float16)
            if cfg.amp and self.device in ('cuda', 'mps') else nullcontext()
        )

        if cfg.amp and self.device == 'cuda':
            try:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception as e:
                self.scaler = None

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.cfg.trainer.epochs)):
            epoch_loss = 0.0
            step = 0
            for batch in self.dataloader:
                state = batch["state"].to(self.device)
                image = batch["image"].to(self.device)
                target = batch["action_chunk"].to(self.device)
                mask = batch["mask"].to(self.device)

                self.optimizer.zero_grad()

                if self.scaler is not None:
                    with self.amp_ctx:
                        pred, mu, logvar = self.model(image, state, target, mask)
                        reconstruction_loss = self.reconstruction_loss(pred, target, mask)
                        kl_loss_ = self.kl_loss(mu, logvar).mean()
                        loss = reconstruction_loss + (self.cfg.beta * kl_loss_)

                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clipping:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clipping)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    with self.amp_ctx:
                        pred, mu, logvar = self.model(image, state, target, mask)
                        reconstruction_loss = self.reconstruction_loss(pred, target, mask)
                        kl_loss_ = self.kl_loss(mu, logvar).mean()
                        loss = reconstruction_loss + (self.cfg.beta * kl_loss_)

                    loss.backward()
                    if self.cfg.grad_clipping:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clipping)
                    self.optimizer.step()

            epoch_loss += loss.item()
            step += 1
            train_loss = epoch_loss / max(1,step)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.cfg.trainer.epochs}] Loss: {epoch_loss:.6f}")






