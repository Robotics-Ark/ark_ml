from typing import Any
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from arkml.algos.ACTransformer.evaluator import ACTransformerEvaluator
from arkml.algos.ACTransformer.trainer import ACTransformerTrainer
from arkml.core.algorithm import BaseAlgorithm
from arkml.core.registry import ALGOS
from .dataset import ActionChunkingArkDataset


# TODO use the configurations correctly
@ALGOS.register("action_chunking_transformer")
class ACTalgorithm(BaseAlgorithm):
    def __init__(self, policy, device: str, cfg: DictConfig):
        super().__init__()
        self.policy = policy.to(device=device)
        self.device = device
        self.cfg = cfg

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])

        if cfg.chunk_size:
            chunk_size = cfg.chunk_size
        else:
            chunk_size = 50

        dataset = ActionChunkingArkDataset(
            dataset_path=cfg.dataset.path,
            transform=transform,
            chunk_size=chunk_size,
        )

        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        train_dataset, val_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.algo.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.algo.trainer.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.algo.trainer.batch_size,
            shuffle=False,
            num_workers=cfg.algo.trainer.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def train(self, *args, **kwargs) -> Any:
        trainer = ACTransformerTrainer(self.policy, self.train_loader, device=self.device)
        return trainer.fit()

    def eval(self, *args, **kwargs) -> dict:
        evaluator = ACTransformerEvaluator(self.policy, self.val_loader, self.device)
        return evaluator.evaluate()
