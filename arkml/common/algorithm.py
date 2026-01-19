from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from arkml.common.evaluator import BaseEvaluator
from arkml.common.trainer import BaseTrainer
from arkml.core.policy import BasePolicy


class CommonAlgorithm(ABC):
    """
    Common algorithm shell that wires model/trainer/evaluator together.
    """

    def __init__(
        self,
        model: BasePolicy,
        trainer: BaseTrainer,
        evaluator: BaseEvaluator,
    ):
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator

        self.alg_name = getattr(
            self.__class__, "_registry_name", self.__class__.__name__
        )

    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        """
        Train the algorithm using the configured trainer.
        """
        ...

    @abstractmethod
    def eval(self, *args, **kwargs) -> Any:
        """
        Evaluate the algorithm using the configured evaluator.
        """
        ...
