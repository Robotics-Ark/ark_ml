import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ark_ml.arkml.core.registry import DATASETS, MODELS


# from core.utils.registry import MODELS, DATASETS

class PiZeroTrainer:
    def __init__(self, cfg, device="cuda"):
        self.device = device
        self.cfg = cfg

        # Build dataset
        dataset_cls = DATASETS.get(cfg.data.name)
        self.dataset = dataset_cls(**cfg.data)
        self.dataloader = DataLoader(self.dataset, batch_size=cfg.trainer.batch_size, shuffle=True)

        # Build policy model
        model_cls = MODELS.get(cfg.algo.model.name)
        self.model = model_cls(cfg.algo.model.state_dim, cfg.algo.model.action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
        self.criterion = nn.MSELoss()

    def train(self):
        self.model.train()
        for epoch in range(self.cfg.trainer.max_epochs):
            epoch_loss = 0.0
            for states, actions, _ in self.dataloader:
                states, actions = states.to(self.device), actions.to(self.device)
                self.optimizer.zero_grad()
                pred_actions = self.model(states)
                loss = self.criterion(pred_actions, actions)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.dataloader)
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.cfg.trainer.max_epochs}] Loss: {epoch_loss:.6f}")
