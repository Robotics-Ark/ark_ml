from abc import ABC, abstractmethod
from typing import Dict, Any

from ark_ml.arkml.core.policy import BasePolicy


class Trainer(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs) -> Dict[str, Any]:
        ...


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        ...


class BaseAlgorithm(ABC):
    policy: BasePolicy
    trainer: Trainer
    evaluator: Evaluator
