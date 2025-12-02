#!/usr/bin/env python3
"""
Debug script for Pi05 rollout pipeline.
"""

import torch
import tempfile
from PIL import Image
import numpy as np
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.nodes.pi05_node import Pi05Node
from arkml.algos.vla.pi05.pi05_processor import Pi05Processor


def debug_pi05_rollout():
    print("Debugging Pi05 rollout pipeline...")
    
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
    
    print("Policy initialized successfully")
    
    # Create dummy observation - let's try with proper format
    dummy_image = torch.rand(1, 3, 224, 224)  # B, C, H, W format
    dummy_obs = {
        "image": dummy_image,
        "instruction": "pickup the block"
    }
    
    print("Testing policy.predict directly...")
    try:
        action = policy.predict(dummy_obs)
        print(f"Direct policy action shape: {action.shape}, values: {action}")
        print("✓ Direct policy test passed")
    except Exception as e:
        print(f"✗ Direct policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Testing processor separately...")
    try:
        processor = Pi05Processor(device=device)
        processed = processor(dummy_obs)
        print(f"Processed image shape: {processed['image'].shape}")
        print(f"Processed instruction shape: {processed['instruction'].shape}")
        print("✓ Processor test passed")
    except Exception as e:
        print(f"✗ Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create Pi05Node with the policy
    node = Pi05Node(policy, device=device)
    
    print("Testing node.predict...")  
    try:
        action = node.predict(dummy_obs)
        print(f"Node action shape: {action.shape}, values: {action}")
        print("✓ Node predict test passed")
    except Exception as e:
        print(f"✗ Node predict test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Testing node.run_once...")
    try:
        action = node.run_once(dummy_obs)
        print(f"Node run_once action shape: {action.shape}, values: {action}")
        print("✓ Node run_once test passed")
    except Exception as e:
        print(f"✗ Node run_once test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All tests passed! Pi05 rollout pipeline is working correctly.")
    return True


if __name__ == "__main__":
    success = debug_pi05_rollout()
    if success:
        print("\nPi05 rollout pipeline implementation is complete and working!")
    else:
        print("\nPi05 rollout pipeline has issues that need to be addressed.")