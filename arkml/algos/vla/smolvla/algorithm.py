import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from ark.utils.utils import ConfigPath
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.policy import BasePolicy
from arkml.core.registry import ALGOS
from arkml.utils.schema_io import get_visual_features
from arkml.utils.utils import _normalise_shape
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import smolVLADataset
from .evaluator import smolVLAEvaluator
from .trainer import smolVLATrainer
from .compute_stats import compute_smolvla_stats


@ALGOS.register("smolvla")
class smolVLAAlgorithm(BaseAlgorithm):
    """Algorithm wrapper for smolVLA training and evaluation.


    Args:
        policy : The policy to be trained.
        device : Device identifier used to move the policy and run training.
        cfg : Configuration object containing all configuration parameters.

    """

    def __init__(self, policy: BasePolicy, device: str, cfg: DictConfig) -> None:
        super().__init__()
        self.model = policy
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

        img_dim = _normalise_shape(cfg.algo.model.image_dim)

        dataset = smolVLADataset(
            dataset_path=cfg.data.dataset_path,
            transform=transform,
            pred_horizon=cfg.algo.model.pred_horizon,
        )
        self.calculate_dataset_stats(
            dataset_path=cfg.data.dataset_path,
            obs_dim=cfg.algo.model.obs_dim,
            action_dim=cfg.algo.model.action_dim,
            image_dim=img_dim,
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

        Returns:
            The result of ``smolVLATrainer.fit()``, typically training
            metrics or artifacts as defined by the trainer implementation.
        """
        trainer = smolVLATrainer(
            model=self.model,
            dataloader=self.train_loader,
            device=self.device,
            lr=self.cfg.algo.trainer.lr,
            weight_decay=getattr(self.cfg.algo.trainer, "weight_decay", 0.0),
            num_epochs=getattr(self.cfg.algo.trainer, "max_epochs", 3),
            grad_accum=getattr(self.cfg.algo.trainer, "grad_accum", 8),
            output_dir=str(os.path.join(self.cfg.output_dir, self.alg_name)),
            use_bf16=getattr(self.cfg.algo.trainer, "use_bf16", False),
            val_dataloader=self.val_loader,
            eval_every=1,
        )
        return trainer.fit()

    def eval(self, *args, **kwargs) -> dict:
        """Run validation via the underlying evaluator.

        Returns:
            dict: Metrics reported by :class:`smolVLAEvaluator.evaluate()`.
        """
        evaluator = smolVLAEvaluator(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
        )
        return evaluator.evaluate()

    def calculate_dataset_stats(
        self,
        dataset_path,
        *,
        obs_dim: int,
        action_dim: int,
        image_dim: tuple[int, int, int],
    ) -> None:
        """
        Compute and save dataset statistics for the smolvla algorithm.
        Args:
            dataset_path: Path to the dataset directory containing trajectory files.
            obs_dim: Dimension of the observation state vector.
            action_dim: Dimension of the action vector.
            image_dim: Dimensions of image data in (channels, height, width) format.

        Returns:
            None
        """

        try:
            stats_path = Path(dataset_path) / "smolvla_stats.json"
            print(f"[smolVLAAlgorithm] Computing dataset stats : {stats_path}")
            if not stats_path.exists():
                stats = compute_smolvla_stats(
                    dataset_path,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    image_channels=image_dim[0],
                    sample_images_only=True,
                )
                stats_path.parent.mkdir(parents=True, exist_ok=True)

                with open(stats_path, "w") as f:
                    json.dump(
                        {
                            k: {kk: vv.tolist() for kk, vv in d.items()}
                            for k, d in stats.items()
                        },
                        f,
                        indent=2,
                    )

            self.model.load_dataset_stats(str(stats_path))
        except Exception as e:
            print(f"[smolVLAAlgorithm] Warning: failed to ensure dataset stats ({e})")
            raise RuntimeError(f"[smolVLAAlgorithm] Warning: {e}")
