#!/usr/bin/env python3
"""
Pi0.5 Posttrain Script
"""
import argparse
import torch
from torch.utils.data import DataLoader
from arkml.core.factory import load_config
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.algos.vla.pi05.dataset import Pi05Dataset
from arkml.algos.vla.pi05.trainer import Pi05Trainer


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 Posttrain Script")
    parser.add_argument("--config", type=str, default="pi05_base", help="Configuration preset")
    args = parser.parse_args()

    # Load configuration
    config = load_config(algo="pi05", preset=args.config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate dataset
    dataset = Pi05Dataset(
        dataset_path=getattr(config.data, 'dataset_path', '/tmp'),
        config_path=getattr(config.data, 'config_path', 'arkml/configs/data/pi05_dataset.yaml')
    )
    print(f"Dataset instantiated with {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Keep batch size = 1
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    # Instantiate model
    model = Pi05Policy(
        model_path=config.model.get("model_path", config.model.get("checkpoint", "")),
        obs_dim=config.model.get("obs_dim", 256),
        action_dim=config.model.get("action_dim", 256),
        image_dim=config.model.get("image_dim", 224),
        flow_dim=config.model.get("flow_dim", 0),
        preset=config.model.get("preset", "lerobot/pi0.5_base")
    )
    model.to(device)
    model.train()  # Set model to training mode
    print("Model instantiated and set to training mode")

    # Initialize trainer
    trainer = Pi05Trainer(
        model=model,
        dataloader=dataloader,  # Pass dataloader, not iterator
        device=device,
        lr=config.training.get("lr", 1e-4),
        weight_decay=config.training.get("weight_decay", 0.01),
        num_epochs=1,
        grad_accum=config.training.get("grad_accum", 1),
        output_dir=config.training.get("output_dir", "/tmp"),
        use_bf16=config.training.get("use_bf16", False),
        val_dataloader=None,
        eval_every=config.training.get("eval_every", float('inf'))
    )
    print("Trainer initialized")

    # Run posttrain iterations
    print("Starting posttrain iterations...")

    dataloader_iter = iter(dataloader)

    for step in range(100):
        try:
            # Get next batch from dataloader
            batch = next(dataloader_iter)
        except StopIteration:
            # If we run out of data, reset the iterator
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        # Run posttrain step
        loss = trainer.posttrain_step(batch)

        # Print iteration number and loss
        if isinstance(loss, dict):
            loss_val = loss.get('loss', torch.tensor(float('nan')))
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
            print(f"Posttrain Iteration {step+1}/100, Loss: {loss_val}")
        else:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            print(f"Posttrain Iteration {step+1}/100, Loss: {loss}")

    print("Posttrain completed successfully!")


if __name__ == "__main__":
    main()