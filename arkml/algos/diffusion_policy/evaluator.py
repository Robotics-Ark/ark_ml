from typing import Any

import torch
from arkml.core.algorithm import Evaluator
from datasets import tqdm
from torch.utils.data import DataLoader

from arkml.core.policy import BasePolicy


class DiffusionPolicyEvaluator(Evaluator):
    def __init__(
        self,
        model: BasePolicy,
        dataloader: DataLoader,
        device: str,
    ) -> None:
        self.model = model.to_device(device)
        self.dataloader = dataloader
        self.model.set_eval_mode()
        self.scheduler = self.model.build_scheduler()

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
            out, loss = self.model.forward(obs=batch, scheduler=self.scheduler)
            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return {"val_loss": avg_loss, "batches": num_batches}
