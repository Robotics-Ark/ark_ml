from ark_ml.arkml.algos.diffusion_policy.evaluator import DiffusionEvaluator
from ark_ml.arkml.algos.diffusion_policy.trainer import DiffusionTrainer
from ark_ml.arkml.core.algorithm import BaseAlgorithm
from ark_ml.arkml.core.registry import ALGOS


@ALGOS.register("diffusion_policy")
class DiffusionPolicyAlgorithm(BaseAlgorithm):
    def __init__(self, policy, dataloader, device="cuda", cfg=None):
        self.policy = policy
        self.trainer = DiffusionTrainer(
            model=policy,
            dataloader=dataloader,
            device=device,
            num_epochs=cfg.algo.trainer.max_epochs,
            lr=cfg.algo.trainer.lr,
            obs_horizon=cfg.algo.model.obs_horizon,
        )
        self.evaluator = DiffusionEvaluator(policy, dataloader, device=device)
