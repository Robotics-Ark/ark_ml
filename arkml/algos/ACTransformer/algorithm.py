import torch

from arkml.algos.ACTransformer.evaluator import ACTransformerEvaluator
from arkml.core.registry import ALGOS

from arkml.core.algorithm import BaseAlgorithm
from arkml.algos.ACTransformer.trainer import ACTransformerTrainer



@ALGOS.register("action_chunking_transformer")
class ACTalgorithm(BaseAlgorithm):
    def __init__(self, policy, dataloader, device="cpu", cfg=None):
        self.policy = policy
        self.dataloader = dataloader
        self.device = device
        self.trainer = ACTransformerTrainer(self.policy, self.dataloader, device=device)
        self.evaluator = ACTransformerEvaluator(self.policy, self.dataloader, device)

    def train(self):
        self.trainer.fit()

    # def act(self, obs):
    #     obs = torch.tensor(obs).float().to(self.device)
    #     with torch.no_grad():
    #         return self.policy.act(obs)
