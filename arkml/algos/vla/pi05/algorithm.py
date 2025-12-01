from typing import Any
import torch
from torch.utils.data import DataLoader
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.policy import BasePolicy
from arkml.core.registry import ALGOS
from omegaconf import DictConfig

@ALGOS.register("pi05")
class Pi05Algorithm(BaseAlgorithm):
    """
    Algorithm wrapper for Pi0.5 training and evaluation.
    """

    def __init__(self, policy: BasePolicy, device: str, cfg: DictConfig) -> None:
        self.policy = policy
        self.device = device
        self.cfg = cfg
        # Set up optimizer based on config
        self.optimizer = torch.optim.Adam(
            policy.get_trainable_params(),
            lr=cfg.get("learning_rate", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.01)
        )

    def train(self, *args, **kwargs) -> Any:
        """Train the model for one epoch using the dataloader."""
        dataloader = args[0] if args else kwargs.get('dataloader')
        if dataloader is None:
            raise ValueError("dataloader is required for training")

        self.policy.set_train_mode()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass
            loss = self.policy.forward(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"avg_loss": avg_loss, "num_batches": num_batches}

    def eval(self, *args, **kwargs) -> dict:
        """Evaluate the model using the dataloader."""
        dataloader = args[0] if args else kwargs.get('dataloader')
        if dataloader is None:
            raise ValueError("dataloader is required for evaluation")

        self.policy.set_eval_mode()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Forward pass
                loss = self.policy.forward(batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"avg_eval_loss": avg_loss, "num_batches": num_batches}

