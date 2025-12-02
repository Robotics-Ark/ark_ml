#!/usr/bin/env python3
"""
Pi0.5 Model Benchmark Script
Measures forward-pass latency and performance metrics.
"""
import time
import torch
import statistics
from arkml.core.factory import load_config
from arkml.algos.vla.pi05.models import Pi05Policy


def main():
    # Load configuration
    config = load_config(algo="pi05", preset="pi05_base")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model
    model = Pi05Policy(
        policy_type=config.model.get("policy_type", "pi05"),
        model_path=config.model.get("model_path", config.model.get("checkpoint", "")),
        obs_dim=config.model.get("obs_dim", 256),
        action_dim=config.model.get("action_dim", 8),  # Default to 8-dim actions
        image_dim=config.model.get("image_dim", (3, 224, 224)),  # Updated to tuple format
        pred_horizon=config.model.get("pred_horizon", 1),
        hidden_dim=config.model.get("hidden_dim", 512),
        vocab_size=config.model.get("vocab_size", 32000),
        fast_vocab_size=config.model.get("fast_vocab_size", 1000)
    )
    model.to(device)
    model.eval()  # Set to evaluation mode for benchmarking
    print("Model loaded and set to evaluation mode")

    # Generate synthetic input tensors
    batch_size = 1
    channels = 3
    height = config.model.get("image_dim", (3, 224, 224))[1] if isinstance(config.model.get("image_dim"), tuple) else 224
    width = config.model.get("image_dim", (3, 224, 224))[2] if isinstance(config.model.get("image_dim"), tuple) else 224

    # Create synthetic image tensor
    fake_images = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)

    # Create synthetic text input
    fake_text = ["pick up the block"] * batch_size  # Repeat for batch

    print(f"Synthetic inputs generated - Images: {fake_images.shape}, Text: {fake_text}")

    # Warmup iterations - now using observation dict with both image and instruction
    print("Starting warmup iterations...")
    for i in range(200):
        with torch.no_grad():
            # Pass observation dict with both image and instruction
            observation = {
                "image": fake_images,
                "instruction": fake_text[0] if batch_size == 1 else fake_text
            }
            _ = model(observation)
        if i % 50 == 0:
            print(f"Warmup progress: {i}/200")

    print("Warmup completed. Starting timed benchmark...")

    # Timed iterations - now using observation dict with both image and instruction
    latencies = []
    for i in range(500):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            # Pass observation dict with both image and instruction
            observation = {
                "image": fake_images,
                "instruction": fake_text[0] if batch_size == 1 else fake_text
            }
            # Use the predict method for consistent forward signature
            _ = model.predict(observation)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        if i % 100 == 0:
            print(f"Benchmark progress: {i}/500")

    # Calculate statistics
    mean_latency = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p90 = float(torch.quantile(torch.tensor(latencies), 0.90).item())
    p95 = float(torch.quantile(torch.tensor(latencies), 0.95).item())
    p99 = float(torch.quantile(torch.tensor(latencies), 0.99).item())
    fps = 1000.0 / mean_latency

    # Print benchmark results table
    print("\n" + "="*60)
    print("Pi0.5 Forward-Pass Benchmark Results")
    print("="*60)
    print(f"{'Metric':<15} {'Value':<15}")
    print("-"*30)
    print(f"{'Mean Latency':<15} {mean_latency:.3f} ms")
    print(f"{'P50 Latency':<15} {p50:.3f} ms")
    print(f"{'P90 Latency':<15} {p90:.3f} ms")
    print(f"{'P95 Latency':<15} {p95:.3f} ms")
    print(f"{'P99 Latency':<15} {p99:.3f} ms")
    print(f"{'FPS':<15} {fps:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()