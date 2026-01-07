from typing import Any
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.policy import BasePolicy
from arkml.core.registry import ALGOS
from arkml.algos.vla.pi05.trainer import Pi05Trainer
from arkml.algos.vla.pi05.evaluator import Pi05Evaluator
from omegaconf import DictConfig
from arkml.utils.utils import _normalise_shape
from torchvision import transforms
from arkml.algos.vla.pi05.dataset import Pi05Dataset
from torch.utils.data import random_split
from arkml.algos.vla.pizero.compute_stats import compute_pizero_stats
# from .compute_stats import compute_pizero_stats


@ALGOS.register("pi05")
class Pi05Algorithm(BaseAlgorithm):
    """
    Algorithm wrapper for Pi0.5 training and evaluation.
    Implements the complete training pipeline for Pi0.5 with multi-stage training.
    """

    def __init__(self, policy: BasePolicy, device: str, cfg: DictConfig) -> None:
        self.policy = policy
        self.device = device
        self.cfg = cfg

        # Extract trainer configuration with safe defaults
        # Follow the intended architecture: cfg.algo.trainer, cfg.algo.training, etc.
        # But be robust to missing algo section for rollout scenarios
        algo_cfg = getattr(cfg, 'algo', {})

        # If algo section is missing, try to use top-level config as fallback for rollout
        if not algo_cfg:
            # For rollout scenarios where full training config isn't provided
            trainer_cfg = getattr(cfg, 'trainer', {})
        else:
            # For training scenarios following maintainer's intended structure
            trainer_cfg = getattr(algo_cfg, 'trainer', {})

        self.lr = getattr(trainer_cfg, 'lr', 2e-4)
        self.batch_size = getattr(trainer_cfg, 'batch_size', 8)
        self.max_epochs = getattr(trainer_cfg, 'max_epochs', 10)
        self.weight_decay = getattr(trainer_cfg, 'weight_decay', 0.0)
        self.num_workers = getattr(trainer_cfg, 'num_workers', 4)
        self.use_bf16 = getattr(trainer_cfg, 'use_bf16', True)

        # Training-specific config following the intended architecture
        if not algo_cfg:
            # Rollout scenario fallback
            training_cfg = getattr(cfg, 'training', {})
            dataset_cfg = getattr(cfg, 'dataset', {})
        else:
            # Training scenario - maintainer's intended structure
            training_cfg = getattr(algo_cfg, 'training', {})
            dataset_cfg = getattr(algo_cfg, 'dataset', {})

        self._training_config = training_cfg
        self._dataset_config = dataset_cfg

        # Set defaults that can be overridden during training if needed
        self.training_stage = getattr(self._training_config, 'stage', 'pretrain')
        self.flow_alpha = getattr(self._training_config, 'flow_alpha', 10.0)
        self.pretrain_steps = getattr(self._training_config, 'pretrain_steps', 280000)
        self.posttrain_steps = getattr(self._training_config, 'posttrain_steps', 80000)
        self.integration_steps = getattr(self._training_config, 'integration_steps', 10)
        
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

        dataset = Pi05Dataset(
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

    def train(self) -> Any:
        """
        Train the Pi0.5 model with multi-stage approach.
        """

        # Load dataset - check if dataset config exists
        dataset_path = getattr(self._dataset_config, 'dataset_path', None)
        if self.cfg.data.dataset_path is None:
            raise ValueError("Dataset path is required for training but not provided in config")

        # Get pred_horizon from either cfg.algo.model or cfg.model
        algo_cfg = getattr(self.cfg, 'algo', {})
        model_cfg = getattr(algo_cfg, 'model', {})
        if not model_cfg:  # If algo.model is empty, check top-level model
            model_cfg = getattr(self.cfg, 'model', {})
        pred_horizon = getattr(model_cfg, 'pred_horizon', 1)



        # Initialize trainer with config
        trainer = Pi05Trainer(
            model=self.policy,
            dataloader=self.train_loader,
            device=self.device,
            lr=getattr(self._training_config, 'lr', self.lr),
            weight_decay=getattr(self._training_config, "weight_decay", self.weight_decay),
            num_epochs=getattr(self._training_config, "max_epochs", self.max_epochs),
            grad_accum=getattr(self._training_config, "grad_accum", 8),
            output_dir=getattr(self.cfg, 'output_dir', './output'),
            use_bf16=getattr(self._training_config, "use_bf16", self.use_bf16),
            flow_alpha=self.flow_alpha,
            val_dataloader=self.val_loader,
            eval_every=1
        )

        # Set the training stage on the model
        self.policy.training_stage = self.training_stage

        # Perform training based on stage
        return trainer.fit()

    def eval(self, eval_dataset) -> dict:
        """
        Evaluate the Pi0.5 model performance.
        """
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # Initialize evaluator
        evaluator = Pi05Evaluator(
            model=self.policy,
            dataloader=eval_dataloader,
            device=self.device
        )

        # Perform evaluation
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
        Compute and save dataset statistics for the PiZero algorithm.
        Args:
            dataset_path: Path to the dataset directory containing trajectory files.
            obs_dim: Dimension of the observation state vector.
            action_dim: Dimension of the action vector.
            image_dim: Dimensions of image data in (channels, height, width) format.

        Returns:
            None
        """

        try:
            stats_path = Path(dataset_path) / "pizero_stats.json"
            print(f"[PiZeroAlgorithm] Computing dataset stats : {stats_path}")
            if not stats_path.exists():
                stats = compute_pizero_stats(
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

            self.policy.load_dataset_stats(str(stats_path))
        except Exception as e:
            print(f"[PiZeroAlgorithm] Warning: failed to ensure dataset stats ({e})")
            raise RuntimeError(f"[PiZeroAlgorithm] Warning: {e}")
