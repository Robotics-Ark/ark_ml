import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from ark_ml.arkml.core.factory import build_model
from ark_ml.arkml.core.registry import DATASETS, ALGOS


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.3")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # Load dataset with task information
    dataset_cls = DATASETS.get(cfg.data.name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = dataset_cls(
        dataset_path=cfg.data.dataset_path,
        transform=transform,
        task_prompt=cfg.task_prompt,
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
    policy = build_model(cfg.algo)

    # 2. Load algorithm
    algo_cls = ALGOS.get(cfg.algo.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo_cls(policy=policy, dataloader=dataloader, device=device, cfg=cfg)

    # 3. Run training
    history = algo.train()
    print("✅ Training finished. History:", history)


if __name__ == "__main__":
    """
    # Train diffusion
    python tools/train.py algo=diffusion data=diffusion_dataset
    """
    main()
