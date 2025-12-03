"""
Benchmarking script for Pi0.5 implementation.
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from arkml.algos.vla.pi05.models import Pi05Policy, flow_matching_loss, DummyBackbone, ActionFlowExpert
from arkml.algos.vla.pi05.config_utils import get_pi05_config
from arkml.algos.vla.pi05.dataset import Pi05Dataset
from arkml.utils.utils import print_trainable_summary


def benchmark_flow_matching_loss():
    """Benchmark flow matching loss computation."""
    print("Benchmarking flow matching loss...")
    
    # Test different tensor sizes
    sizes = [(100, 8), (1000, 8), (100, 64), (1000, 64)]
    
    results = []
    for batch_size, action_dim in sizes:
        pred = torch.randn(batch_size, action_dim, requires_grad=True)
        target = torch.randn(batch_size, action_dim)
        
        # Warmup
        for _ in range(3):
            loss = flow_matching_loss(pred, target)
            loss.backward()
            pred.grad.zero_()
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            loss = flow_matching_loss(pred, target)
            loss.backward()
            pred.grad.zero_()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to milliseconds
        results.append((batch_size, action_dim, avg_time))
        print(f"  Size ({batch_size}, {action_dim}): {avg_time:.4f} ms/iter")
    
    return results


def benchmark_dummy_backbone():
    """Benchmark DummyBackbone forward pass."""
    print("Benchmarking DummyBackbone...")
    
    # Test different configurations
    configs = [
        (1, 512, "Small batch"),
        (8, 512, "Medium batch"),
        (32, 512, "Large batch"),
        (8, 1024, "Wide hidden"),
    ]
    
    backbone = DummyBackbone(hidden_dim=512)
    
    results = []
    for batch_size, hidden_dim, label in configs:
        if hidden_dim != 512:
            backbone = DummyBackbone(hidden_dim=hidden_dim)
        
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Warmup
        for _ in range(5):
            _ = backbone(x)
        
        # Benchmark
        start_time = time.time()
        for _ in range(50):
            _ = backbone(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 50 * 1000  # Convert to milliseconds
        results.append((batch_size, hidden_dim, avg_time, label))
        print(f"  {label} ({batch_size}, {hidden_dim}): {avg_time:.4f} ms/iter")
    
    return results


def benchmark_action_flow_expert():
    """Benchmark ActionFlowExpert operations."""
    print("Benchmarking ActionFlowExpert...")
    
    configs = [
        (1, 256, 8, "Small"),
        (8, 256, 8, "Medium"),
        (32, 256, 8, "Large"),
        (8, 512, 16, "High-dim"),
    ]
    
    results = []
    for batch_size, hidden_dim, action_dim, label in configs:
        flow_expert = ActionFlowExpert(hidden_dim=hidden_dim, action_dim=action_dim)
        hidden_states = torch.randn(batch_size, hidden_dim)
        target_actions = torch.randn(batch_size, action_dim)
        
        # Test forward with target (training)
        # Warmup
        for _ in range(5):
            _ = flow_expert(hidden_states, target_action=target_actions)
        
        start_time = time.time()
        for _ in range(50):
            _ = flow_expert(hidden_states, target_action=target_actions)
        forward_time = (time.time() - start_time) / 50 * 1000
        
        # Test prediction
        # Warmup
        for _ in range(5):
            _ = flow_expert.predict(hidden_states, steps=5, step_size=0.1)
        
        start_time = time.time()
        for _ in range(50):
            _ = flow_expert.predict(hidden_states, steps=5, step_size=0.1)
        predict_time = (time.time() - start_time) / 50 * 1000
        
        results.append((batch_size, hidden_dim, action_dim, forward_time, predict_time, label))
        print(f"  {label}: Forward={forward_time:.4f}ms, Predict={predict_time:.4f}ms")
    
    return results


def benchmark_dataset_operations():
    """Benchmark dataset operations."""
    print("Benchmarking dataset operations...")
    
    # Create a mock dataset
    dataset = Pi05Dataset("/mock/path", max_samples=1000)
    
    # Benchmark getitem
    start_time = time.time()
    for i in range(0, min(100, len(dataset)), len(dataset)//20):  # Sample 20 points
        _ = dataset[i]
    end_time = time.time()
    
    avg_getitem_time = (end_time - start_time) / min(20, len(dataset)) * 1000
    print(f"  Dataset getitem: {avg_getitem_time:.4f} ms/sample")
    
    return avg_getitem_time


def benchmark_memory_usage():
    """Benchmark memory usage of components."""
    print("Benchmarking memory usage...")
    
    # Check memory for different components
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Flow matching loss memory
    pred = torch.randn(1000, 8, requires_grad=True)
    target = torch.randn(1000, 8)
    loss = flow_matching_loss(pred, target)
    
    print(f"  Flow matching loss memory (approx): {(pred.element_size() * pred.nelement() + target.element_size() * target.nelement())/1024/1024:.2f} MB")
    
    # Dummy backbone memory
    backbone = DummyBackbone(hidden_dim=512)
    x = torch.randn(8, 3, 224, 224)
    output = backbone(x)
    
    backbone_memory = sum(p.numel() * p.element_size() for p in backbone.parameters())
    print(f"  DummyBackbone parameters memory: {backbone_memory/1024/1024:.2f} MB")
    
    return {
        'flow_matching_memory_mb': (pred.element_size() * pred.nelement() + target.element_size() * target.nelement())/1024/1024,
        'backbone_memory_mb': backbone_memory/1024/1024
    }


def run_comprehensive_benchmark():
    """Run all benchmarks."""
    print("=" * 60)
    print("Pi0.5 Comprehensive Benchmarking")
    print("=" * 60)
    
    # Run all benchmarks
    print("\n1. Flow Matching Loss Benchmark:")
    flow_results = benchmark_flow_matching_loss()
    
    print("\n2. Dummy Backbone Benchmark:")
    backbone_results = benchmark_dummy_backbone()
    
    print("\n3. ActionFlowExpert Benchmark:")
    action_results = benchmark_action_flow_expert()
    
    print("\n4. Dataset Operations Benchmark:")
    dataset_time = benchmark_dataset_operations()
    
    print("\n5. Memory Usage Benchmark:")
    memory_usage = benchmark_memory_usage()
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Fastest flow matching: {min([r[2] for r in flow_results]):.4f} ms")
    print(f"Fastest backbone: {min([r[2] for r in backbone_results]):.4f} ms")
    print(f"Fastest ActionFlowExpert forward: {min([r[3] for r in action_results]):.4f} ms")
    print(f"Dataset getitem time: {dataset_time:.4f} ms")
    print(f"Memory usage - Flow matching: {memory_usage['flow_matching_memory_mb']:.2f} MB")
    print(f"Memory usage - Backbone: {memory_usage['backbone_memory_mb']:.2f} MB")
    
    return {
        'flow_results': flow_results,
        'backbone_results': backbone_results,
        'action_results': action_results,
        'dataset_time': dataset_time,
        'memory_usage': memory_usage
    }


def run_performance_regression_test():
    """Run performance regression test."""
    print("\nRunning Performance Regression Test...")
    
    # Test with PyTorch's built-in performance testing
    torch.backends.cudnn.benchmark = True  # Enable cuDNN optimization if available
    
    # Test tensor operations speed
    sizes = [100, 500, 1000, 2000]
    times = []
    
    for size in sizes:
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # Warmup
        for _ in range(3):
            _ = torch.mm(a, b)
        
        # Benchmark matrix multiplication
        start_time = time.time()
        for _ in range(10):
            _ = torch.mm(a, b)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        times.append((size, avg_time))
        print(f"  Matrix mult ({size}x{size}): {avg_time*1000:.4f} ms")
    
    return times


if __name__ == "__main__":
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    # Run performance regression test
    regression_results = run_performance_regression_test()
    
    print(f"\nAll benchmarks completed successfully!")
    print(f"Performance regression test completed for {len(regression_results)} matrix sizes.")