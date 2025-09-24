import sys
from typing import Any

import torch
from arkml.algos.diffusion_policy.evaluator import DiffusionEvaluator
from arkml.algos.diffusion_policy.trainer import DiffusionTrainer
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.registry import ALGOS
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from arkml.algos.diffusion_policy.dataset import DiffusionDataset
from arkml.core.policy import BasePolicy


@ALGOS.register("diffusion_policy")
class DiffusionPolicyAlgorithm(BaseAlgorithm):
    def __init__(self, policy: BasePolicy, device: str, cfg: DictConfig) -> None:
        super().__init__()
        self.policy = policy
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

        dataset = DiffusionDataset(
            dataset_path=cfg.data.dataset_path,
            transform=transform,
            task_prompt=cfg.task_prompt,
            pred_horizon=cfg.algo.model.pred_horizon,
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
        num_workers = cfg.algo.trainer.num_workers
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.algo.trainer.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0 and sys.platform != "win32"),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.algo.trainer.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0 and sys.platform != "win32"),
        )

        print(f"Data split : train: {train_len}, val: {val_len}")

    def train(self, *args, **kwargs) -> Any:
        """Run training via the underlying trainer.

        Args:
         dataloader : The training dataloader


        Returns:
            The result of ``PiZeroTrainer.fit()``, typically training
            metrics or artifacts as defined by the trainer implementation.
        """
        trainer_cfg = self.cfg.algo.trainer
        model_cfg = self.cfg.algo.model
        trainer = DiffusionTrainer(
            model=self.policy,
            dataloader=self.train_loader,
            device=self.device,
            num_epochs=trainer_cfg.max_epochs,
            lr=trainer_cfg.lr,
            weight_decay=trainer_cfg.get("weight_decay", 1e-6),
            num_diffusion_iters=model_cfg.diffusion_steps,
            obs_horizon=model_cfg.obs_horizon,
            pred_horizon=model_cfg.pred_horizon,
            use_ema=trainer_cfg.get("ema", True),
            ema_power=trainer_cfg.get("ema_power", 0.75),
            grad_clip=trainer_cfg.get("grad_clip", None),
        )
        return trainer.fit()

    def eval(self, *args, **kwargs) -> dict:
        """Run validation via the underlying evaluator.

        Args:
            dataloader: The validation dataloader.

        Returns:
            dict: Metrics reported by :class:`PiZeroEvaluator.evaluate()`.
        """
        evaluator = DiffusionEvaluator(
            model=self.policy,
            dataloader=self.val_loader,
            device=self.device,
        )
        return evaluator.evaluate()
