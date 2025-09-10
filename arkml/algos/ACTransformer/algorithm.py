import torch
from ark_ml.arkml.core.registry import ALGOS

from arkml.core.algorithm import BaseAlgorithm


@ALGOS.register("action_chunking_transformer")
class ACTalgorithm(BaseAlgorithm):
    def __init__(self, model, trainer, device="cpu"):
        self.model = model
        self.trainer = trainer
        self.device = device

    def train(self, dataloader):
        self.trainer.fit(self.model, dataloader)

    def act(self, obs):
        obs = torch.tensor(obs).float().to(self.device)
        with torch.no_grad():
            return self.model.act(obs)
