#!/usr/bin/env python3
"""
Pi0.5 Full Training Script (Pretrain + Posttrain + Evaluation)
"""
import argparse
import torch
from torch.utils.data import DataLoader
from arkml.core.factory import load_config
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.algos.vla.pi05.dataset import Pi05Dataset
from arkml.algos.vla.pi05.trainer import Pi05Trainer
from arkml.algos.vla.pi05.evaluator import Pi05Evaluator


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 Full Training Script")
    parser.add_argument("--config", type=str, default="pi05_base", help="Configuration preset")
    parser.add_argument("--pretrain_steps", type=int, default=100, help="Number of pretrain steps")
    parser.add_argument("--posttrain_steps", type=int, default=100, help="Number of posttrain steps")
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

    # Create dataloader with batch size = 1
    dataloader = DataLoader(
        dataset,
        batch_size=1,
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

    # PRETRAIN phase
    print(f"Starting PRETRAIN phase for {args.pretrain_steps} steps...")
    dataloader_iter = iter(dataloader)

    for i in range(args.pretrain_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        # Run pretrain step
        loss = trainer.train_step_pretrain(batch)
        if isinstance(loss, dict) and 'loss' in loss:
            loss_val = loss['loss']
        else:
            loss_val = loss

        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()

        print(f"pretrain step {i+1}, loss {loss_val}")

    print("PRETRAIN phase completed!")

    # POSTTRAIN phase
    print(f"Starting POSTTRAIN phase for {args.posttrain_steps} steps...")
    dataloader_iter = iter(dataloader)

    for j in range(args.posttrain_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        # Run posttrain step
        loss = trainer.posttrain_step(batch)
        if isinstance(loss, dict) and 'loss' in loss:
            loss_val = loss['loss']
        else:
            loss_val = loss

        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()

        print(f"posttrain step {j+1}, loss {loss_val}")

    print("POSTTRAIN phase completed!")

    # Run evaluation
    print("Starting evaluation...")
    model.eval()  # Set model to evaluation mode
    evaluator = Pi05Evaluator(
        model=model,
        dataloader=dataloader,
        device=device
    )
    print("Evaluator instantiated")

    # Evaluate on a few batches
    dataloader_eval = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    evaluator.reset()
    print("Evaluator reset")

    # Process a few batches for evaluation
    eval_batches = min(10, len(dataloader_eval))  # Evaluate on 10 batches max
    for k, batch in enumerate(dataloader_eval):
        if k >= eval_batches:
            break

        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        evaluator.evaluate_pretrain(batch)

    # Summarize metrics
    metrics = evaluator.summarize()
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Optionally save checkpoint
    print("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()