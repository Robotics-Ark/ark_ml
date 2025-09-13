import os
from typing import Any

import torch
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.policy import BasePolicy
from arkml.core.registry import ALGOS
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import PiZeroDataset
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

        # Load dataset with task information
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset = PiZeroDataset(
            dataset_path=cfg.data.dataset_path,
            transform=transform,
            task_prompt=cfg.task_prompt,
        )

        # Train/val split (80/20)
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        train_dataset, val_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.algo.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.algo.trainer.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.algo.trainer.batch_size,
            shuffle=False,
            num_workers=cfg.algo.trainer.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        print(f"Data split -> train: {train_len}, val: {val_len}")

    def train(self, *args, **kwargs) -> Any:
        """Run training via the underlying trainer.

        Args:
         dataloader : The training dataloader


        Returns:
            The result of ``PiZeroTrainer.fit()``, typically training
            metrics or artifacts as defined by the trainer implementation.
        """
        trainer = PiZeroTrainer(
            model=self.model,
            dataloader=self.train_loader,
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
        return trainer.fit()

    def eval(self, *args, **kwargs) -> dict:
        """Run validation via the underlying evaluator.

        Args:
            dataloader: The validation dataloader.

        Returns:
            dict: Metrics reported by :class:`PiZeroEvaluator.evaluate()`.
        """
        evaluator = PiZeroEvaluator(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
        )
        return evaluator.evaluate()
