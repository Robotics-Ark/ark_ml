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
        self.lr = cfg.trainer.get('lr', 2e-4)
        self.batch_size = cfg.trainer.get('batch_size', 8)
        self.max_epochs = cfg.trainer.get('max_epochs', 10)
        self.weight_decay = cfg.trainer.get('weight_decay', 0.0)
        self.num_workers = cfg.trainer.get('num_workers', 4)
        self.use_bf16 = cfg.trainer.get('use_bf16', True)

        # Training-specific config
        self.training_stage = cfg.training.get('stage', 'pretrain')
        self.flow_alpha = cfg.training.get('flow_alpha', 10.0)
        self.pretrain_steps = cfg.training.get('pretrain_steps', 280000)
        self.posttrain_steps = cfg.training.get('posttrain_steps', 80000)
        self.integration_steps = cfg.training.get('integration_steps', 10)

    def train(self, train_dataset, val_dataset=None) -> Any:
        """
        Train the Pi0.5 model with multi-stage approach.
        """
        # Create data loaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_dataloader = None
        if val_dataset:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

        # Initialize trainer with config
        trainer = Pi05Trainer(
            model=self.policy,
            dataloader=train_dataloader,
            device=self.device,
            lr=self.lr,
            weight_decay=self.weight_decay,
            num_epochs=self.max_epochs,
            grad_accum=1.0,  # Gradient accumulation
            output_dir='./output',  # TODO: Get from config
            use_bf16=self.use_bf16,
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