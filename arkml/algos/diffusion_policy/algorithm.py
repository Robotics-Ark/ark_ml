import os
import sys
import math
from typing import Any

import torch
from arkml.algos.diffusion_policy.dataset import DiffusionPolicyDataset
from arkml.algos.diffusion_policy.trainer import DiffusionTrainer
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.policy import BasePolicy
from arkml.core.registry import ALGOS
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .evaluator import DiffusionPolicyEvaluator


@ALGOS.register("diffusion_policy")
class DiffusionPolicyAlgorithm(BaseAlgorithm):
    def __init__(self, policy: BasePolicy, device: str, cfg: DictConfig) -> None:
        super().__init__()
        self.policy = policy
        self.device = device
        self.cfg = cfg

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # TODO read from config
            ]
        )

        dataset = DiffusionPolicyDataset(
            dataset_path=cfg.data.dataset_path,
            transform=transform,
            pred_horizon=cfg.algo.model.pred_horizon,
            obs_horizon=cfg.algo.model.obs_horizon,
            action_horizon=cfg.algo.model.action_horizon,
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
        prefetch_factor = getattr(cfg.algo.trainer, "prefetch_factor", 2)
        dl_kwargs = dict(
            batch_size=cfg.algo.trainer.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0 and sys.platform != "win32"),
        )
        val_kwargs = dict(
            batch_size=cfg.algo.trainer.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0 and sys.platform != "win32"),
        )

        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor
            val_kwargs["prefetch_factor"] = prefetch_factor

        self.train_loader = DataLoader(train_dataset, **dl_kwargs)
        self.val_loader = DataLoader(val_dataset, **val_kwargs)

        print(f"Data split : train: {train_len}, val: {val_len}")

    def train(self, *args, **kwargs) -> Any:
        """Run training via the underlying trainer.


        Returns:
            The result of ``fit()``, typically training

        """
        trainer_cfg = self.cfg.algo.trainer
        model_cfg = self.cfg.algo.model
        # Allow step-based training by mapping a step budget to epochs
        max_steps = getattr(trainer_cfg, "max_steps", None)
        num_epochs = trainer_cfg.max_epochs
        if max_steps is not None:
            steps_per_epoch = max(1, len(self.train_loader))
            num_epochs = math.ceil(int(max_steps) / steps_per_epoch)
            print(
                f"[DiffusionPolicyAlgorithm] Using step budget: max_steps={max_steps}, "
                f"steps_per_epoch={steps_per_epoch}, computed_epochs={num_epochs}"
            )

        trainer = DiffusionTrainer(
            model=self.policy,
            dataloader=self.train_loader,
            device=self.device,
            output_dir=str(os.path.join(self.cfg.output_dir, self.alg_name)),
            num_epochs=num_epochs,
            lr=trainer_cfg.lr,
            weight_decay=trainer_cfg.get("weight_decay", 1e-6),
            obs_horizon=model_cfg.obs_horizon,
            pred_horizon=model_cfg.pred_horizon,
            use_ema=trainer_cfg.get("ema", True),
            ema_power=trainer_cfg.get("ema_power", 0.75),
            grad_clip=trainer_cfg.get("grad_clip", None),
            max_steps=max_steps,
        )
        return trainer.fit()

    def eval(self, *args, **kwargs) -> dict:
        """Run validation via the underlying evaluator.

        Returns:
            dict: Metrics reported by : evaluate()`.
        """
        evaluator = DiffusionPolicyEvaluator(
            model=self.policy,
            dataloader=self.val_loader,
            device=self.device,
        )
        return evaluator.evaluate()
