from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import DataLoader

from arkml.core.policy import BasePolicy


class Trainer(ABC):
    """
    Abstract base class for training drivers.
    Defines the interface for components that optimize a model.
    Subclasses should implement `fit` to run the training loop and
    return a summary of metrics and/or artifact paths.
    """

    @abstractmethod
    def fit(self, *args, **kwargs) -> dict[str, Any]:
        """Run the training procedure and return summary metrics.

        Args:
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            A dictionary of training results, such as losses,
            accuracies, global step counts, and/or artifact locations.

        """
        ...


class Evaluator(ABC):
    """
    Abstract base class for evaluation routines.
    Defines the interface for components that compute validation/test metrics without updating model parameters.
    Subclasses should implement `evaluate`to run the evaluation loop and return a metrics dictionary.
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, Any]:
        """Compute evaluation metrics and return a summary.

        Args:
            *args: Optional positional arguments (e.g., dataloader, metrics).
            **kwargs: Optional keyword arguments (e.g., device, logger).

        Returns:
            A dictionary of evaluation results, such as average
            loss, accuracy, F1, or other task‑specific metrics.
        """
        ...


class BaseAlgorithm(ABC):
    policy: BasePolicy
    trainer: Trainer
    evaluator: Evaluator

    def __init__(self):
        # Use registry name if present, otherwise fallback to class name
        self.alg_name = getattr(
            self.__class__, "_registry_name", self.__class__.__name__
        )

    @abstractmethod
    def train(self, dataloader: DataLoader, *args, **kwargs) -> Any:
        """Train the algorithm using the provided dataloader.

        Args:
            dataloader: Training dataloader yielding batches.
            *args: Optional positional arguments for subclass-specific needs.
            **kwargs: Optional keyword arguments for subclass-specific needs.

        Returns:
            Any: Training output as defined by the implementation (e.g.,
            metrics dict, training history, or artifacts path).
        """

        ...

    @abstractmethod
    def eval(self, dataloader: DataLoader, *args, **kwargs) -> Any:
        """Evaluate the algorithm using the provided dataloader.

        Args:
            dataloader : Evaluation/validation dataloader.
            *args: Optional positional arguments for subclass-specific needs.
            **kwargs: Optional keyword arguments for subclass-specific needs.

        Returns:
            Any: Evaluation output as defined by the implementation (e.g.,
            metrics dict or evaluation report).
        """
        ...
