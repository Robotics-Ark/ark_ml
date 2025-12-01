#!/usr/bin/env python3
"""
Pi05 Benchmark Script
"""
import torch
import time
import subprocess
import platform
import psutil
from arkml.core.factory import load_config  # Using factory as per earlier examples
from arkml.algos.vla.pi05.models import Pi05Policy
import statistics
import io
import sys
from contextlib import redirect_stdout


def get_gpu_info():
    """Get GPU information if available."""
    gpu_name = "N/A (CPU only)"
    gpu_memory_total = "N/A"
    gpu_memory_allocated = "N/A"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        gpu_memory_allocated = f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB"
    
    return gpu_name, gpu_memory_total, gpu_memory_allocated


def get_cpu_info():
    """Get CPU information."""
    try:
        # Try to get CPU info using lscpu on Linux
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Model name:' in line:
                    return line.split(':', 1)[1].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback to platform.processor()
    return platform.processor()


def main():
    # Capture all output
    output_buffer = io.StringIO()
    
    with redirect_stdout(output_buffer):
        # 1. Print hardware info
        print("Hardware Information:")
        gpu_name, gpu_memory_total, gpu_memory_allocated = get_gpu_info()
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory (Total): {gpu_memory_total}")
        print(f"GPU Memory (Allocated): {gpu_memory_allocated}")
        print(f"CPU: {get_cpu_info()}")
        print(f"System RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Version: {torch.version.cuda}")
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            print(f"CUDA Capability: {capability[0]}.{capability[1]}")
        else:
            print(f"CUDA Capability: N/A")
        print()
        
        # 2. Load Pi05 model
        cfg = load_config(algo="pi05", preset="pi05_base", overrides={})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Pi05Policy(
            policy_type=cfg.model.policy_type,
            model_path=cfg.model.model_path,
            obs_dim=cfg.model.obs_dim,
            action_dim=cfg.model.action_dim,
            image_dim=cfg.model.image_dim
        ).to(device)
        model.eval()
        print("Model loaded successfully.")
        print()
        
        # 3. Generate synthetic data
        fake_images = torch.randn(1, 3, 480, 640, dtype=torch.float32, device=device)  # Use the image_dim from config
        print(f"Synthetic data generated: images shape {fake_images.shape}")
        print()

        # 4. Collect forward-pass latency metrics
        print("Running forward pass benchmarks...")
        # Prepare the observation dict as expected by the model
        observation = {"image": fake_images}

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(observation)

        # Synchronize and measure 50 iterations
        latencies = []
        for _ in range(50):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()

            with torch.no_grad():
                _ = model(observation)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate metrics
        mean_latency = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p90 = statistics.quantiles(latencies, n=100)[89]  # 90th percentile
        p95 = statistics.quantiles(latencies, n=100)[94]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        fps = 1000.0 / mean_latency
        
        # 5. Print summary
        print("======== PI05 Benchmark Results ========")
        print(f"GPU: {gpu_name}")
        print(f"Mean Latency: {mean_latency:.2f} ms")
        print(f"P50: {p50:.2f} ms")
        print(f"P90: {p90:.2f} ms")
        print(f"P95: {p95:.2f} ms")
        print(f"P99: {p99:.2f} ms")
        print(f"FPS: {fps:.2f}")
        print("========================================")
    
    # 6. Write all output to file
    with open('benchmark_pi05_results.txt', 'w') as f:
        f.write(output_buffer.getvalue())
    
    print("Benchmark results written to benchmark_pi05_results.txt")


if __name__ == "__main__":
    main()