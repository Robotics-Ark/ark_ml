import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Type, Callable

import torch
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.policy import BasePolicy
from arkml.utils.utils import _normalise_shape
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class CommonAlgorithm(BaseAlgorithm):
    """
    Common base algorithm class that provides shared functionality for all algorithms.

    Args:
        policy: The policy to be trained.
        device: Device identifier used to move the policy and run training.
        cfg: Configuration object containing all configuration parameters.
        dataset_class: Dataset class to instantiate.
        trainer_class: Trainer class to instantiate.
        evaluator_class: Evaluator class to instantiate.
        compute_stats_fn: Optional function to compute dataset statistics.
        stats_filename: Name of the stats file (e.g., "pizero_stats.json").
    """

    def __init__(
            self,
            policy: BasePolicy,
            device: str,
            cfg: DictConfig,
            dataset_class: Type[Dataset],
            trainer_class: Type,
            evaluator_class: Type,
            compute_stats_fn: Optional[Callable] = None,
            stats_filename: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.model = policy  # Some algorithms use self.model, others use self.policy
        self.device = device
        self.cfg = cfg

        # Build transforms
        transform = self._build_transforms()

        # Get dataset parameters
        dataset_params = self._get_dataset_params()

        # Initialize dataset
        dataset = dataset_class(
            dataset_path=cfg.data.dataset_path,
            transform=transform,
            **dataset_params,
        )

        # Compute and load dataset stats if needed
        if compute_stats_fn is not None and stats_filename is not None:
            self._compute_and_load_stats(
                compute_stats_fn=compute_stats_fn,
                stats_filename=stats_filename,
            )

        # Create train/val split
        self.train_loader, self.val_loader = self._create_dataloaders(dataset)

    def _build_transforms(self) -> transforms.Compose:
        """
        Build image transformation pipeline.
        Can be overridden by subclasses for custom transforms.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _get_dataset_params(self) -> dict:
        """
        Extract dataset parameters from config.
        Can be overridden by subclasses for custom parameters.
        """
        params = {}

        # Common parameters across algorithms
        if hasattr(self.cfg.algo.model, 'pred_horizon'):
            params['pred_horizon'] = self.cfg.algo.model.pred_horizon

        if hasattr(self.cfg.algo.model, 'obs_horizon'):
            params['obs_horizon'] = self.cfg.algo.model.obs_horizon

        if hasattr(self.cfg.algo.model, 'action_horizon'):
            params['action_horizon'] = self.cfg.algo.model.action_horizon

        if hasattr(self.cfg.algo.model, 'chunk_size'):
            params['chunk_size'] = self.cfg.algo.model.chunk_size

        return params

    def _create_dataloaders(self, dataset: Dataset) -> tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders with 80/20 split.
        """
        # Train/val split (80/20)
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        train_dataset, val_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

        # Get dataloader parameters
        num_workers = self.cfg.algo.trainer.num_workers
        batch_size = self.cfg.algo.trainer.batch_size
        prefetch_factor = getattr(self.cfg.algo.trainer, "prefetch_factor", 2)

        # Common dataloader kwargs
        dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0 and sys.platform != "win32"),
        )

        val_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0 and sys.platform != "win32"),
        )

        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor
            val_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(train_dataset, **dl_kwargs)
        val_loader = DataLoader(val_dataset, **val_kwargs)

        print(f"Data split: train: {train_len}, val: {val_len}")

        return train_loader, val_loader

    def _compute_and_load_stats(
            self,
            compute_stats_fn: Callable,
            stats_filename: str,
    ) -> None:
        """
        Compute and save dataset statistics if they don't exist, then load them.
        """
        try:
            stats_path = Path(self.cfg.data.dataset_path) / stats_filename
            print(f"[{self.__class__.__name__}] Computing dataset stats: {stats_path}")

            if not stats_path.exists():
                # Get image dimensions
                img_dim = _normalise_shape(self.cfg.algo.model.image_dim)

                # Compute stats
                stats = compute_stats_fn(
                    self.cfg.data.dataset_path,
                    obs_dim=self.cfg.algo.model.obs_dim,
                    action_dim=self.cfg.algo.model.action_dim,
                    image_channels=img_dim[0],
                    sample_images_only=True,
                )

                # Save stats
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

            # Load stats into policy/model
            if hasattr(self.policy, 'load_dataset_stats'):
                self.policy.load_dataset_stats(str(stats_path))
            elif hasattr(self.model, 'load_dataset_stats'):
                self.model.load_dataset_stats(str(stats_path))

        except Exception as e:
            print(f"[{self.__class__.__name__}] Warning: failed to ensure dataset stats ({e})")
            raise RuntimeError(f"[{self.__class__.__name__}] Warning: {e}")

    def _get_trainer_params(self) -> dict:
        """
        Extract trainer parameters from config.
        Can be overridden by subclasses for custom parameters.
        """
        trainer_cfg = self.cfg.algo.trainer

        params = {
            'model': self.policy,
            'dataloader': self.train_loader,
            'device': self.device,
            'lr': trainer_cfg.lr,
            'output_dir': str(os.path.join(self.cfg.output_dir, self.alg_name)),
        }

        # Optional common parameters
        if hasattr(trainer_cfg, 'weight_decay'):
            params['weight_decay'] = getattr(trainer_cfg, 'weight_decay', 0.0)

        if hasattr(trainer_cfg, 'max_epochs'):
            params['num_epochs'] = getattr(trainer_cfg, 'max_epochs', 3)
        elif hasattr(trainer_cfg, 'epochs'):
            params['num_epochs'] = trainer_cfg.epochs

        if hasattr(trainer_cfg, 'grad_accum'):
            params['grad_accum'] = getattr(trainer_cfg, 'grad_accum', 8)

        if hasattr(trainer_cfg, 'use_bf16'):
            params['use_bf16'] = getattr(trainer_cfg, 'use_bf16', False)

        if hasattr(trainer_cfg, 'grad_clip'):
            params['grad_clip'] = getattr(trainer_cfg, 'grad_clip', None)

        if hasattr(self, 'val_loader'):
            params['val_dataloader'] = self.val_loader
            params['eval_every'] = 1

        return params

    def train(self, trainer_class: Type, *args, **kwargs) -> Any:
        """
        Run training via the underlying trainer.

        Args:
            trainer_class: The trainer class to instantiate.

        Returns:
            The result of trainer.fit().
        """
        trainer_params = self._get_trainer_params()
        trainer_params.update(kwargs)  # Allow override via kwargs

        trainer = trainer_class(**trainer_params)
        return trainer.fit()

    def eval(self, evaluator_class: Type, *args, **kwargs) -> dict:
        """
        Run evaluation via the underlying evaluator.

        Args:
            evaluator_class: The evaluator class to instantiate.

        Returns:
            dict: Metrics reported by evaluator.evaluate().
        """
        evaluator = evaluator_class(
            model=self.policy,
            dataloader=self.val_loader,
            device=self.device,
        )
        return evaluator.evaluate()