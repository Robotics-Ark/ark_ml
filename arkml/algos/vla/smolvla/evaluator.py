"""Evaluation loop for the PiZero algorithm.

Computes average validation loss across the provided dataloader by leveraging
the model's forward method, which is expected to return either a loss tensor
directly or a tuple whose first element is the loss.
"""

from typing import Any

import torch
from arkml.core.algorithm import Evaluator
from arkml.core.policy import BasePolicy
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class smolVLAEvaluator(Evaluator):
    """Evaluator for PiZero models.

    Args:
        model: Model compatible with the training setup to evaluate.
        dataloader: Validation dataloader yielding batches.
        device: Target device for evaluation.
    """

    def __init__(
        self,
        model: BasePolicy,
        dataloader: DataLoader,
        device: str,
    ) -> None:
        self.model = model.to_device(device)
        self.dataloader = dataloader
        self.model.set_eval_mode()

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Run evaluation over the validation dataloader.

        Returns:
            A metrics dictionary with keys:
                - ``val_loss``: Average loss over the validation set.
                - ``batches``: Number of evaluated batches.
        """

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc="Validation",
            leave=False,
        )

        for _, batch in progress_bar:
            out = self.model.forward(batch)
            loss = out if torch.is_tensor(out) else out[0]
            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return {"val_loss": avg_loss, "batches": num_batches}
