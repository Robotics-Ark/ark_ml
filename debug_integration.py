#!/usr/bin/env python3
"""
Debug the specific issue with processor and policy integration
"""

import torch
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.nodes.pi05_node import Pi05Node
from arkml.algos.vla.pi05.pi05_processor import Pi05Processor

def test_integration():
    print("Creating policy...")
    device = torch.device("cpu")
    policy = Pi05Policy(
        policy_type="pi0.5",
        model_path="lerobot/pi0.5",
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
    
    print(f"Text embedding shape: {policy.text_embedding.weight.shape}")
    print(f"Text projection shape: {policy.text_projection.weight.shape}")
    
    # Create observation
    dummy_image = torch.rand(1, 3, 224, 224)
    dummy_obs = {
        "image": dummy_image,
        "instruction": "pickup the block"
    }
    
    print("\nTesting processor...")
    processor = Pi05Processor(device=device)
    processed_obs = processor(dummy_obs)
    
    print(f"Processed image shape: {processed_obs['image'].shape}")
    print(f"Processed instruction shape: {processed_obs['instruction'].shape}")
    print(f"Instruction tokens: {processed_obs['instruction'][:10]}")  # First 10
    print(f"Max token: {processed_obs['instruction'].max().item()}")
    print(f"Min token: {processed_obs['instruction'].min().item()}")
    
    # Test the policy's encode_text with these tokens directly
    print("\nTesting encode_text directly...")
    tokens = processed_obs['instruction'].unsqueeze(0)  # Add batch dimension if needed
    print(f"Tokens shape: {tokens.shape}")
    try:
        text_emb = policy.encode_text(tokens)
        print(f"Text embedding result shape: {text_emb.shape}")
        print("encode_text succeeded!")
    except Exception as e:
        print(f"encode_text failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test the full policy predict
    print("\nTesting policy.predict...")
    try:
        action = policy.predict(processed_obs)
        print(f"Action shape: {action.shape}")
        print("Policy predict succeeded!")
    except Exception as e:
        print(f"Policy predict failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    test_integration()