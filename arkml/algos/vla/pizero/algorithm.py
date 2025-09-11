import torch
from arkml.core.registry import ALGOS

@ALGOS.register("pizero")
class PiZeroAlgorithm:
    def __init__(self, model, trainer, device="cpu"):
        self.model = model.to(device)
        self.trainer = trainer
        self.device = device

    def train(self, dataloader):
        self.trainer.fit(self.model, dataloader)

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(obs).cpu().numpy()
