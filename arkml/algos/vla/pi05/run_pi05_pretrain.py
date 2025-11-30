#!/usr/bin/env python3
"""
Pi0.5 Pretrain Script
"""
import argparse
import torch
from torch.utils.data import DataLoader
from arkml.core.factory import load_config
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.algos.vla.pi05.dataset import Pi05Dataset
from arkml.algos.vla.pi05.trainer import Pi05Trainer


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 Pretrain Script")
    parser.add_argument("--config", type=str, default="pi05_base", help="Configuration preset")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of pretraining steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
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
        batch_size=args.batch_size,
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
        num_epochs=1,  # We'll control iterations manually
        grad_accum=config.training.get("grad_accum", 1),
        output_dir=config.training.get("output_dir", "/tmp"),
        use_bf16=config.training.get("use_bf16", False),
        val_dataloader=None,
        eval_every=config.training.get("eval_every", float('inf'))
    )
    print("Trainer initialized")

    # Run pretraining steps
    print(f"Starting pretraining for {args.num_steps} steps...")

    dataloader_iter = iter(dataloader)

    for step in range(args.num_steps):
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

        # Perform pretrain step
        # Compute pretrain loss using the trainer
        loss_dict = trainer.train_step_pretrain(batch)
        pretrain_loss = loss_dict.get('loss', torch.tensor(float('nan')))

        # Print iteration number and loss
        print(f"Step {step+1}/{args.num_steps}, Loss: {pretrain_loss.item():.6f}")

    print("Pretraining completed successfully!")


if __name__ == "__main__":
    main()