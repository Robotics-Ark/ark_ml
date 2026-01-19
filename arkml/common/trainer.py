from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import DataLoader

from arkml.core.policy import BasePolicy


class BaseTrainer(ABC):
    """
    Common trainer interface for all algorithms.
    """

    def __init__(self, model: BasePolicy, dataloader: DataLoader, device: str):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    @abstractmethod
    def fit(self, *args, **kwargs) -> dict[str, Any]:
        """
        Run the training loop and return metrics/artifacts.
        """
        ...
