#!/usr/bin/env python3
"""
Tiny pretrain test for Pi0.5 to verify gradients and loss behavior.
Runs exactly 5 iterations to check for finite values and basic functionality.
"""
import torch
import requests
from torch.utils.data import DataLoader
from arkml.core.factory import load_config
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.algos.vla.pi05.dataset import Pi05Dataset
from arkml.algos.vla.pi05.trainer import Pi05Trainer


def main():
    # Load config
    config = load_config(
        algo="pi05",
        preset="pi05_base",
        overrides={"data.num_samples": 8}
    )

    # Load dataset and dataloader
    dataset = Pi05Dataset(config.data)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

    # Instantiate model
    model = Pi05Policy(config.model)

    # Load backbone checkpoint with network error handling
    try:
        model.load_backbone_checkpoint(config.model.checkpoint)
    except (requests.exceptions.RequestException, OSError, EnvironmentError) as e:
        print(f"[ERROR] Failed to download or load backbone checkpoint: {e}")
        print("[INFO] Falling back to skeleton model")
        # Continue with the model as is without the checkpoint

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Instantiate trainer with minimal parameters - no dataloader passed
    trainer = Pi05Trainer(
        model=model,
        device=device,
        lr=1e-4,
        weight_decay=0.0,
        grad_accum=1,
        output_dir="./test_output"
    )

    # Run exactly 5 iterations
    for i in range(5):
        batch = next(iter(dataloader))
        loss_dict = trainer.train_step_pretrain(batch)
        print(f"step {i}: {loss_dict}")

        # Assert finite values
        for v in loss_dict.values():
            assert torch.isfinite(v), "Non-finite value in loss"

    print("Pretrain smoke test complete.")


if __name__ == "__main__":
    main()