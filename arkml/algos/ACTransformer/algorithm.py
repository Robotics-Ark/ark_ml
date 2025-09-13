import torch
from typing import Any

from arkml.algos.ACTransformer.evaluator import ACTransformerEvaluator
from arkml.core.registry import ALGOS
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from arkml.core.algorithm import BaseAlgorithm
from arkml.algos.ACTransformer.trainer import ACTransformerTrainer

# TODO use the configurations correctly
@ALGOS.register("action_chunking_transformer")
class ACTalgorithm(BaseAlgorithm):
    def __init__(self, policy, device: str, cfg: DictConfig):
        super().__init__()
        self.policy = policy.to(device=device)
        self.device = device
        self.cfg = cfg


    def train(self, dataloader: DataLoader, *args, **kwargs) -> Any:
        trainer = ACTransformerTrainer(self.policy, dataloader, device=self.device)
        return trainer.fit()

    def eval(self, dataloader: DataLoader, *args, **kwargs) -> dict:
        evaluator = ACTransformerEvaluator(self.policy, dataloader, self.device)
        return evaluator.evaluate()
