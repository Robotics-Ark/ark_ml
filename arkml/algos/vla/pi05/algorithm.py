from typing import Any
import torch
from torch.utils.data import DataLoader
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.policy import BasePolicy
from arkml.core.registry import ALGOS
from arkml.algos.vla.pi05.trainer import Pi05Trainer
from arkml.algos.vla.pi05.evaluator import Pi05Evaluator
from omegaconf import DictConfig

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

        # Extract training configuration
        self.lr = cfg.algo.trainer.get('lr', 2e-4)
        self.batch_size = cfg.algo.trainer.get('batch_size', 8)
        self.max_epochs = cfg.algo.trainer.get('max_epochs', 10)
        self.weight_decay = cfg.algo.trainer.get('weight_decay', 0.0)
        self.num_workers = cfg.algo.trainer.get('num_workers', 4)
        self.use_bf16 = cfg.algo.trainer.get('use_bf16', True)

        # Training-specific config
        self.training_stage = cfg.algo.training.get('stage', 'pretrain')
        self.flow_alpha = cfg.algo.training.get('flow_alpha', 10.0)
        self.pretrain_steps = cfg.algo.training.get('pretrain_steps', 280000)
        self.posttrain_steps = cfg.algo.training.get('posttrain_steps', 80000)
        self.integration_steps = cfg.algo.training.get('integration_steps', 10)

    def train(self) -> Any:
        """
        Train the Pi0.5 model with multi-stage approach.
        """
        # Load datasets using self.cfg following the pattern from PiZero
        from arkml.algos.vla.pi05.dataset import Pi05Dataset
        from torch.utils.data import random_split
        import sys
        from torchvision import transforms

        # Define transform
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

        # Load dataset
        dataset = Pi05Dataset(
            dataset_path=self.cfg.algo.dataset.dataset_path,
            transform=transform,
            pred_horizon=self.cfg.algo.model.pred_horizon,
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

        num_workers = self.cfg.algo.training.num_workers
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.algo.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0 and sys.platform != "win32"),
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.algo.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0 and sys.platform != "win32"),
        )

        # Initialize trainer with config
        trainer = Pi05Trainer(
            model=self.policy,
            dataloader=train_dataloader,
            device=self.device,
            lr=self.cfg.algo.training.lr,
            weight_decay=getattr(self.cfg.algo.training, "weight_decay", 0.0),
            num_epochs=getattr(self.cfg.algo.training, "max_epochs", 3),
            grad_accum=getattr(self.cfg.algo.training, "grad_accum", 8),
            output_dir=self.cfg.output_dir,
            use_bf16=getattr(self.cfg.algo.training, "use_bf16", False),
            flow_alpha=self.flow_alpha,
            val_dataloader=val_dataloader,
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
