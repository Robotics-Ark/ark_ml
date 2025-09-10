import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ark_ml.arkml.core.factory import build_model
from ark_ml.arkml.core.registry import DATASETS, ALGOS


@hydra.main(config_path="../configs", config_name="defaults.yaml")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # Load dataset with task information
    dataset_cls = DATASETS.get(cfg.data.name)
    dataset = dataset_cls(
        dataset_path=cfg.data.dataset_path,
        pred_horizon=cfg.algo.model.pred_horizon,
        obs_horizon=cfg.algo.model.obs_horizon,
        action_horizon=cfg.algo.model.action_horizon,
        subsample=cfg.data.subsample,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.algo.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.algo.trainer.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dataset.collate_fn,
    )

    # Build model (policy)
    policy = build_model(cfg.algo.model)

    # 2. Load algorithm
    algo_cls = ALGOS.get(cfg.algo.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo_cls(policy=policy, dataloader=dataloader, device=device, cfg=cfg)

    # 3. Run training
    history = algo.trainer.fit()
    print("✅ Training finished. History:", history)


if __name__ == "__main__":
    """
    # Train diffusion
    python tools/train.py algo=diffusion data=diffusion_dataset
    """
    main()
