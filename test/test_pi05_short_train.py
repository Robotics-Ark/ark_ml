#!/usr/bin/env python3
"""
Short combined pretrain + posttrain test for Pi0.5 to validate end-to-end training.
Runs 10 pretrain steps + 10 posttrain steps to check stability and finite values.
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
            "data.num_samples": 32,
            "model.flow_dim": 256
        }
    )

    # Load dataset and dataloader
    dataset = Pi05Dataset(config.data)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)

    # Instantiate model
    model = Pi05Policy(config.model)

    # Load backbone checkpoint with network error handling
    try:
        model.load_backbone_checkpoint(config.model.checkpoint)
    except (requests.exceptions.RequestException, OSError, EnvironmentError) as e:
        print("[ERROR] Failed loading backbone:", e)
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

    # Pretrain loop (10 steps)
    print("=== PRETRAIN (10 steps) ===")
    for i in range(10):
        batch = next(iter(dataloader))
        loss_dict = trainer.train_step_pretrain(batch)
        print(f"pretrain step {i}: {loss_dict}")

        # Finite checks
        for v in loss_dict.values():
            assert torch.isfinite(v), "NaN/INF detected in pretrain loss"

    # Posttrain loop (10 steps)
    print("=== POSTTRAIN (10 steps) ===")
    for i in range(10):
        batch = next(iter(dataloader))
        loss_dict = trainer.train_step_posttrain(batch)
        print(f"posttrain step {i}: {loss_dict}")

        # Finite checks
        for v in loss_dict.values():
            assert torch.isfinite(v), "NaN/INF detected in posttrain loss"

    print("Short end-to-end Pi05 training test complete.")


if __name__ == "__main__":
    main()