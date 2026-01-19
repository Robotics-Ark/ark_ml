from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Base dataset interface shared by all algorithms.
    Implementations should return a normalized sample dict.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return a single sample as a structured dict.
        """
        ...
