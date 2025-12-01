"""
Unit tests for Pi0.5 components that avoid circular import issues.
These tests are designed to work without importing the full ARK-ML system.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def test_fast_encode_decode_roundtrip():
    """Test that FAST encode/decode roundtrip preserves values within quantization error."""
    # Import within test to avoid global import issues
    from arkml.algos.vla.tokenizers.fast import FASTTokenizer
    
    tokenizer = FASTTokenizer(vocab_path="", num_bins=100, min_val=-1.0, max_val=1.0)
    
    # Test with simple continuous values
    original_actions = np.array([0.0, 0.5, -0.5, 0.9, -0.9])
    tokens = tokenizer.encode(original_actions)
    decoded_actions = tokenizer.decode(tokens)
    
    # Check that values are preserved within quantization error
    # Since we're quantizing to 100 bins over [-1, 1], max error should be ~0.02
    assert len(tokens) == len(original_actions)
    assert decoded_actions.shape == original_actions.shape
    
    # Quantization error should be reasonable
    max_error = 2.0 / 100  # Range is 2, divided by 100 bins
    assert np.allclose(original_actions, decoded_actions, atol=max_error * 2)  # Allow some tolerance


def test_flow_matching_loss_backward_pass():
    """Test that flow matching loss supports backward pass."""
    from arkml.algos.vla.pi05.models import flow_matching_loss
    
    pred = torch.rand(4, 8, requires_grad=True)
    target = torch.rand(4, 8)
    
    loss = flow_matching_loss(pred, target)
    
    # Should be a scalar tensor
    assert loss.shape == torch.Size([])
    assert loss.requires_grad
    
    # Should be able to perform backward pass
    loss.backward()
    
    # Gradients should be computed for pred
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_action_flow_expert():
    """Test the ActionFlowExpert functionality."""
    from arkml.algos.vla.pi05.models import ActionFlowExpert
    
    hidden_dim = 512
    action_dim = 8
    batch_size = 3
    
    flow_expert = ActionFlowExpert(hidden_dim, action_dim)
    
    # Test forward pass with target (for training)
    hidden_states = torch.rand(batch_size, hidden_dim)
    target_actions = torch.rand(batch_size, action_dim)
    
    flow_vectors = flow_expert(hidden_states, target_action=target_actions)
    assert flow_vectors.shape == (batch_size, action_dim)
    
    # Test forward pass without target (for inference)
    flow_vectors_inf = flow_expert(hidden_states)
    assert flow_vectors_inf.shape == (batch_size, action_dim)
    
    # Test predict method
    predicted_actions = flow_expert.predict(hidden_states, steps=5, step_size=0.1)
    assert predicted_actions.shape == (batch_size, action_dim)


def test_dummy_backbone():
    """Test the DummyBackbone functionality."""
    from arkml.algos.vla.pi05.models import DummyBackbone
    
    hidden_dim = 256
    backbone = DummyBackbone(hidden_dim=hidden_dim)
    
    batch_size = 2
    images = torch.rand(batch_size, 3, 224, 224)
    
    output = backbone(images)
    assert output.shape == (batch_size, hidden_dim)


def test_pi05_policy_creation():
    """Test Pi05Policy model creation and basic functionality."""
    from arkml.algos.vla.pi05.models import Pi05Policy
    
    # Create a simple Pi05Policy model
    model = Pi05Policy(
        policy_type="pi0.5",
        model_path="test_path",
        obs_dim=10,
        action_dim=8,
        image_dim=(3, 224, 224),
        pred_horizon=1,
        hidden_dim=512,
        vocab_size=32000,
        fast_vocab_size=1000
    )
    
    # Test that all required components exist
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'subtask_head')
    assert hasattr(model, 'fast_head')
    assert hasattr(model, 'flow_head')
    
    # Test basic forward pass with minimal data
    batch = {
        "image": torch.rand(1, 3, 224, 224),
        "action": torch.rand(1, 8),  # Continuous actions
    }
    
    output = model.forward(batch)
    
    # Output should be a scalar loss tensor
    assert output.shape == torch.Size([])
    assert output.requires_grad  # Should be differentiable


if __name__ == "__main__":
    # Run tests individually to avoid import issues
    import sys
    # Temporarily block problematic modules to avoid import issues
    sys.modules['arkml.algos.vla.pizero.algorithm'] = type(sys)('arkml.algos.vla.pizero.algorithm')
    sys.modules['arkml.algos.vla.pizero.models'] = type(sys)('arkml.algos.vla.pizero.models')
    sys.modules['arkml.algos.act.algorithm'] = type(sys)('arkml.algos.act.algorithm')
    sys.modules['arkml.algos.act.models'] = type(sys)('arkml.algos.act.models')
    sys.modules['arkml.algos.diffusion_policy.algorithm'] = type(sys)('arkml.algos.diffusion_policy.algorithm')
    sys.modules['arkml.algos.diffusion_policy.models'] = type(sys)('arkml.algos.diffusion_policy.models')
    sys.modules['arkml.core.policy'] = type(sys)('arkml.core.policy')
    sys.modules['arkml.core.registry'] = type(sys)('arkml.core.registry')
    sys.modules['arkml.core.algorithm'] = type(sys)('arkml.core.algorithm')
    
    print("Running individual tests...")
    
    test_fast_encode_decode_roundtrip()
    print("✓ FAST encode/decode roundtrip test passed")
    
    test_flow_matching_loss_backward_pass()
    print("✓ Flow matching loss backward pass test passed")
    
    test_action_flow_expert()
    print("✓ ActionFlowExpert test passed")
    
    test_dummy_backbone()
    print("✓ DummyBackbone test passed")
    
    test_pi05_policy_creation()
    print("✓ Pi05Policy creation test passed")
    
    print("\nAll tests passed!")