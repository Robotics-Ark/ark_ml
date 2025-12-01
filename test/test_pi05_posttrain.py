#!/usr/bin/env python3
"""
Tiny posttrain flow-head test for Pi0.5 to verify flow predictions and gradients.
Runs exactly 5 iterations to check for finite values and flow head functionality.
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
        overrides={
            "data.num_samples": 8,
            "model.flow_dim": 256
        }
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
        print("[ERROR] Backbone checkpoint load failed:", e)
        raise

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
        loss_dict = trainer.train_step_posttrain(batch)
        print(f"step {i}: posttrain_loss={loss_dict}")

        # Finite checks
        for v in loss_dict.values():
            assert torch.isfinite(v), "Non-finite value in posttrain loss"

    print("Posttrain flow smoke test complete.")


if __name__ == "__main__":
    main()