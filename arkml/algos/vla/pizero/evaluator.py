import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F

class PiZeroTrainer:
    """
    Trainer for Pi-Zero policy.
    """
    def __init__(self, model, optimizer, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device


    def fit(self, dataloader: DataLoader, epochs: int = 100):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                obs = batch["state"].float().to(self.device)
                action_gt = batch["action"].float().to(self.device)

                pred_action = self.model(obs)
                loss = F.mse_loss(pred_action, action_gt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
