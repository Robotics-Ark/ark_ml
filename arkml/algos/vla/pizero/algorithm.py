import os
from typing import Any

from arkml.core.algorithm import BaseAlgorithm
from arkml.core.policy import BasePolicy
from arkml.core.registry import ALGOS
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .evaluator import PiZeroEvaluator
from .trainer import PiZeroTrainer


@ALGOS.register("pizero")
class PiZeroAlgorithm(BaseAlgorithm):
    """Algorithm wrapper for PiZero training and evaluation.


    Args:
        policy : The policy to be trained.
        device : Device identifier used to move the policy and run training.
        cfg : Configuration object containing all configuration parameters.

    """

    def __init__(self, policy: BasePolicy, device: str, cfg: DictConfig) -> None:
        super().__init__()
        self.model = policy.to_device(device)
        self.device = device
        self.cfg = cfg

    def train(self, dataloader: DataLoader, *args, **kwargs) -> Any:
        """Run training via the underlying trainer.

        Args:
         dataloader : The training dataloader


        Returns:
            The result of ``PiZeroTrainer.fit()``, typically training
            metrics or artifacts as defined by the trainer implementation.
        """
        self.trainer = PiZeroTrainer(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            lr=self.cfg.algo.trainer.lr,
            weight_decay=(
                getattr(self.cfg.algo.lora, "weight_decay", 0.0) if self.cfg else 0.0
            ),
            num_epochs=self.cfg.algo.trainer.max_epochs if self.cfg else 3,
            grad_accum=(
                getattr(self.cfg.algo.trainer, "grad_accum", 8) if self.cfg else 8
            ),
            output_dir=str(os.path.join(self.cfg.output_dir, self.alg_name)),
            use_bf16=(
                getattr(self.cfg.algo.trainer, "use_bf16", False) if self.cfg else False
            ),
        )
        return self.trainer.fit()

    def eval(self, dataloader: DataLoader, *args, **kwargs) -> dict:
        """Run validation via the underlying evaluator.

        Args:
            dataloader: The validation dataloader.

        Returns:
            dict: Metrics reported by :class:`PiZeroEvaluator.evaluate()`.
        """
        evaluator = PiZeroEvaluator(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
        )
        return evaluator.evaluate()
