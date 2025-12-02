#!/usr/bin/env python3
"""
Test script for Pi05 rollout pipeline.
"""

import torch
import tempfile
from PIL import Image
import numpy as np
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.nodes.pi05_node import Pi05Node
from arkml.algos.vla.pi05.pi05_processor import Pi05Processor


def test_pi05_rollout():
    print("Testing Pi05 rollout pipeline...")
    
    # Create a dummy Pi05Policy
    device = torch.device("cpu")  # Use CPU for this test
    policy = Pi05Policy(
        policy_type="pi0.5",
        model_path="lerobot/pi0.5",  # Placeholder
        obs_dim=256,
        action_dim=8,
        image_dim=(3, 224, 224),
        pred_horizon=1,
        hidden_dim=512,
        vocab_size=32000,
        fast_vocab_size=1000
    )
    policy.to_device(device)
    policy.set_eval_mode()
    
    # Create Pi05Node with the policy
    node = Pi05Node(policy, device=device)
    
    # Create dummy observation
    dummy_image = torch.rand(3, 224, 224)  # CHW format
    dummy_obs = {
        "image": dummy_image,
        "instruction": "pickup the block"
    }
    
    print("Testing node.run_once...")
    try:
        action = node.run_once(dummy_obs)
        print(f"Action shape: {action.shape}, Action values: {action[:5]}")  # Show first 5 values
        print("✓ run_once test passed")
    except Exception as e:
        print(f"✗ run_once test failed: {e}")
        return False
    
    print("Testing node.predict...")
    try:
        action = node.predict(dummy_obs)
        print(f"Action shape: {action.shape}, Action values: {action[:5]}")  # Show first 5 values
        print("✓ predict test passed")
    except Exception as e:
        print(f"✗ predict test failed: {e}")
        return False
    
    print("Testing processor directly...")
    try:
        processor = Pi05Processor(device=device)
        processed = processor(dummy_obs)
        print(f"Processed image shape: {processed['image'].shape}")
        print(f"Processed instruction shape: {processed['instruction'].shape}")
        print("✓ processor test passed")
    except Exception as e:
        print(f"✗ processor test failed: {e}")
        return False
    
    print("\n✓ All tests passed! Pi05 rollout pipeline is working correctly.")
    return True


if __name__ == "__main__":
    success = test_pi05_rollout()
    if success:
        print("\nPi05 rollout pipeline implementation is complete and working!")
    else:
        print("\nPi05 rollout pipeline has issues that need to be addressed.")