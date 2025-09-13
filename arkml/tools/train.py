import hydra
import torch
from arkml.core.factory import build_model
from arkml.core.registry import DATASETS, ALGOS
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


# TODO Move dataloader to algorithm
@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.3")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # Load dataset with task information
    dataset_cls = DATASETS.get(cfg.data.name)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = dataset_cls(
        dataset_path=cfg.data.dataset_path,
        transform=transform,
        # task_prompt=cfg.task_prompt, # TODO
    )

    # Train/val split (80/20)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.algo.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.algo.trainer.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.algo.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.algo.trainer.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Data split -> train: {train_len}, val: {val_len}")

    # Build model (policy)
    policy = build_model(cfg.algo)

    # Load algorithm
    algo_cls = ALGOS.get(cfg.algo.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo_cls(policy=policy, device=device, cfg=cfg)

    #  Run training
    history = algo.train(dataloader=train_loader)
    print("Training finished :", history)

    # Run Evaluation
    history = algo.eval(dataloader=val_loader)
    print("Validation finished :", history)


if __name__ == "__main__":
    """
    # Train a policy
    HYDRA_FULL_ERROR=1 python -m arkml.tools.train algo=pizero \
    data=pizero_dataset task_prompt="Pick the yellow cube and place it in the white background area of the table" \
    data.dataset_path=/path/tp/data/set
    """
    main()
