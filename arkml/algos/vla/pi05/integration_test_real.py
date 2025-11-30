"""
DEPRECATED TEST FILE — superseded by test_pi05_integration.py
Kept only for reference. Should not be run.
"""

#!/usr/bin/env python3
"""
Integration test script for Pi0.5 to verify end-to-end functionality with real data and weights.
This script loads a checkpoint from Hugging Face and tests the complete pipeline.
"""
import os
import traceback
import requests
import torch
from torch.utils.data import DataLoader
from arkml.core.factory import load_config
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.algos.vla.pi05.dataset import Pi05Dataset
from arkml.algos.vla.pi05.trainer import Pi05Trainer
from arkml.algos.vla.pi05.evaluator import Pi05Evaluator
from omegaconf import OmegaConf

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
algo_config_path = os.path.join(repo_root, "arkml/configs/algo/pi05.yaml")
data_config_path = os.path.join(repo_root, "arkml/configs/data/pi05_dataset.yaml")

print(f"[DEBUG] Resolved repo root: {repo_root}")
print(f"[DEBUG] Resolved algo config path: {algo_config_path}")
print(f"[DEBUG] Resolved data config path: {data_config_path}")


def main():
    """
    Run the Pi0.5 integration test with real data and weights.
    """
    success = True

    # Load main config using the factory
    try:
        config = load_config(
            algo="pi05",
            preset="pi05_base",
            overrides={
                "model.checkpoint": "lerobot/pi0.5_base",
                "data.num_samples": 4
            }
        )
        print("[INFO] Successfully loaded configuration")
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        traceback.print_exc()
        return False

    # Instantiate dataset
    try:
        dataset = Pi05Dataset(
            dataset_path=getattr(config.data, 'dataset_path', '/tmp'),  # Use appropriate dataset path
            config_path=data_config_path  # Use resolved data config path instead of default relative path
        )
        print("[INFO] Successfully instantiated dataset")
    except Exception as e:
        print(f"[ERROR] Failed to instantiate dataset: {e}")
        traceback.print_exc()
        return False

    # Instantiate model
    try:
        print("[DEBUG] model_path:", config.model.get("model_path", config.model.get("checkpoint", "")))
        print("[DEBUG] obs_dim:", config.model.get("obs_dim", 256))
        print("[DEBUG] action_dim:", config.model.get("action_dim", 256))
        print("[DEBUG] image_dim:", config.model.get("image_dim", 224))
        print("[DEBUG] flow_dim:", config.model.get("flow_dim", 0))
        print("[DEBUG] preset:", config.model.get("preset", "lerobot/pi0.5_base"))

        model = Pi05Policy(
            model_path=config.model.get("model_path", config.model.get("checkpoint", "")),
            obs_dim=config.model.get("obs_dim", 256),
            action_dim=config.model.get("action_dim", 256),
            image_dim=config.model.get("image_dim", 224),
            flow_dim=config.model.get("flow_dim", 0),
            preset=config.model.get("preset", "lerobot/pi0.5_base")
        )
        print("[INFO] Successfully instantiated model")
    except Exception as e:
        print(f"[ERROR] Failed to instantiate model: {e}")
        traceback.print_exc()
        return False

    # Load backbone checkpoint with network error handling
    try:
        model.load_backbone_checkpoint(config.model.checkpoint)
        print("[INFO] Successfully loaded backbone checkpoint")
    except (requests.exceptions.RequestException, OSError, EnvironmentError) as e:
        print(f"[ERROR] Failed to download or load backbone checkpoint: {e}")
        print("[INFO] Falling back to skeleton model")
        # Continue with the model as is without the checkpoint
    except Exception as e:
        print(f"[ERROR] Unexpected error during checkpoint loading: {e}")
        traceback.print_exc()
        return False

    # Pull one batch from the dataset
    try:
        # Create a dataloader to get a batch with proper collation
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        batch = next(iter(dataloader))
        print("[INFO] Successfully retrieved batch from dataset")
    except Exception as e:
        print(f"[ERROR] Failed to retrieve batch from dataset: {e}")
        traceback.print_exc()
        return False

    # Convert images to device and dtype expected by model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Move batch to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    # Run forward pass
    try:
        with torch.no_grad():
            fast = batch.get("fast_tokens", None)
            outputs = model(batch["images"], fast_tokens=fast)

        # Check for finite values in outputs
        for k, v in outputs.items():
            if hasattr(v, "isfinite"):
                assert torch.isfinite(v).all(), f"Non-finite values in output {k}"

        print("[INFO] Forward pass successful")
        print("Output shapes and dtypes:")
        for key, value in outputs.items():
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)} = {value}")
    except AssertionError as e:
        print(f"[ERROR] Non-finite values detected in forward pass: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        traceback.print_exc()
        return False

    # Run trainer smoke test
    try:
        trainer = Pi05Trainer(
            model=model,
            dataloader=iter([batch]),
            device=device,
            lr=config.training.get("lr", 1e-4),
            weight_decay=config.training.get("weight_decay", 0.01),
            num_epochs=1,
            grad_accum=1,
            output_dir="/tmp",
            use_bf16=False,
            val_dataloader=None,
            eval_every=1
        )

        loss_dict = trainer.train_step_pretrain(batch)
        print("[INFO] Trainer pretrain step successful")
        print("Loss dictionary:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"[ERROR] Trainer pretrain step failed: {e}")
        traceback.print_exc()
        return False

    # Run evaluator
    try:
        evaluator = Pi05Evaluator(
            model=model,
            dataloader=dataloader,
            device=device
        )
        print("[INFO] Successfully instantiated evaluator")

        # Reset evaluator and run evaluation
        evaluator.reset()
        print("[INFO] Evaluator reset")

        # Evaluate using the batch
        evaluator.evaluate_pretrain(batch)
        print("[INFO] Evaluator pretrain evaluation successful")

        metrics = evaluator.summarize()
        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"[ERROR] Evaluator failed: {e}")
        traceback.print_exc()
        return False

    print("[INFO] All integration tests passed successfully!")
    return success


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
    else:
        exit(0)