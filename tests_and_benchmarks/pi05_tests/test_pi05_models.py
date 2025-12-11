"""
Comprehensive tests for Pi0.5 models.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from arkml.algos.vla.pi05.models import Pi05Policy, flow_matching_loss, DummyBackbone, ActionFlowExpert


class TestPi05Models:
    """Test suite for Pi0.5 models."""

    def test_flow_matching_loss_basic(self):
        """Test basic functionality of flow matching loss."""
        pred = torch.rand(4, 8, requires_grad=True)
        target = torch.rand(4, 8)
        
        loss = flow_matching_loss(pred, target)
        
        assert loss.shape == torch.Size([])
        assert loss.requires_grad
        assert loss >= 0.0
        
        # Test backward pass
        loss.backward()
        assert pred.grad is not None

    def test_flow_matching_loss_edge_cases(self):
        """Test edge cases for flow matching loss."""
        # Test with identical tensors (should be ~0)
        identical = torch.ones(2, 3)
        loss = flow_matching_loss(identical, identical)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
        
        # Test with zero tensors
        zero1, zero2 = torch.zeros(2, 3), torch.zeros(2, 3)
        loss = flow_matching_loss(zero1, zero2)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_dummy_backbone(self):
        """Test DummyBackbone functionality."""
        backbone = DummyBackbone(hidden_dim=512)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)
        
        assert output.shape == (2, 512)
        assert torch.is_tensor(output)
        
        # Test different batch sizes
        x2 = torch.randn(5, 3, 224, 224)
        output2 = backbone(x2)
        assert output2.shape == (5, 512)

    def test_action_flow_expert_training_mode(self):
        """Test ActionFlowExpert in training mode (with target)."""
        flow_expert = ActionFlowExpert(hidden_dim=256, action_dim=8)
        
        hidden_states = torch.randn(3, 256)
        target_actions = torch.randn(3, 8)
        
        # Forward with target (training mode)
        flow_vectors = flow_expert(hidden_states, target_action=target_actions)
        
        assert flow_vectors.shape == (3, 8)
        assert torch.is_tensor(flow_vectors)

    def test_action_flow_expert_inference_mode(self):
        """Test ActionFlowExpert in inference mode (without target)."""
        flow_expert = ActionFlowExpert(hidden_dim=256, action_dim=8)
        
        hidden_states = torch.randn(3, 256)
        
        # Forward without target (inference mode)
        pred_vectors = flow_expert(hidden_states)
        
        assert pred_vectors.shape == (3, 8)
        assert torch.is_tensor(pred_vectors)

    def test_action_flow_expert_predict(self):
        """Test ActionFlowExpert prediction method."""
        flow_expert = ActionFlowExpert(hidden_dim=256, action_dim=8)
        
        hidden_states = torch.randn(3, 256)
        
        # Use predict method
        actions = flow_expert.predict(hidden_states, steps=5, step_size=0.1)
        
        assert actions.shape == (3, 8)
        assert torch.is_tensor(actions)

    @patch('lerobot.policies.pi05.modeling_pi05.PI05Policy')
    def test_pi05_policy_mock_integration(self, mock_pi05_class):
        """Test Pi05Policy with mocked LeRobot integration."""
        # Setup mock
        mock_policy_instance = Mock()
        mock_policy_instance.config = Mock()
        mock_policy_instance.config.n_action_steps = 1
        mock_policy_instance.config.use_fast_tokens = True
        mock_policy_instance.config.use_flow_matching = True
        mock_policy_instance.config.backbone_type = 'siglip_gemma'
        mock_policy_instance.forward.return_value = (torch.tensor(0.5, requires_grad=True), {})
        mock_policy_instance.select_action.return_value = torch.randn(1, 8)
        mock_policy_instance.reset.return_value = None
        mock_policy_instance.eval.return_value = None
        mock_policy_instance.train.return_value = None
        mock_policy_instance.to.return_value = mock_policy_instance
        mock_policy_instance.config.input_features = {}
        mock_policy_instance.config.output_features = {}
        
        mock_pi05_class.from_pretrained.return_value = mock_policy_instance
        
        # Test policy creation
        with patch('arkml.core.app_context.ArkMLContext') as mock_context:
            mock_context.visual_input_features = ['image']
            
            policy = Pi05Policy(
                policy_type='pi0.5',
                model_path='test_path',
                backbone_type='siglip_gemma',
                use_fast_tokens=True,
                use_flow_matching=True,
                obs_dim=9,
                action_dim=8,
                image_dim=(3, 224, 224),
                pred_horizon=1
            )
            
            assert policy.obs_dim == 9
            assert policy.action_dim == 8
            assert policy._policy is mock_policy_instance

    @patch('lerobot.policies.pi05.modeling_pi05.PI05Policy')
    def test_pi05_policy_forward_pass(self, mock_pi05_class):
        """Test Pi05Policy forward pass with mocked LeRobot."""
        # Setup mock
        mock_policy_instance = Mock()
        mock_policy_instance.forward.return_value = (torch.tensor(0.5, requires_grad=True), {})
        mock_policy_instance.config = Mock()
        mock_policy_instance.config.input_features = {}
        mock_policy_instance.config.output_features = {}
        
        mock_pi05_class.from_pretrained.return_value = mock_policy_instance
        
        with patch('arkml.core.app_context.ArkMLContext') as mock_context:
            mock_context.visual_input_features = ['image']
            
            policy = Pi05Policy(
                policy_type='pi0.5',
                model_path='test_path',
                obs_dim=9,
                action_dim=8,
                image_dim=(3, 224, 224)
            )
            
            # Test forward pass
            batch = {
                'observation.images.image': torch.randn(2, 3, 224, 224),
                'action': torch.randn(2, 8)
            }
            
            loss = policy.forward(batch)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() == 0.5  # Mocked value

    def test_pi05_policy_device_management(self):
        """Test Pi05Policy device management methods."""
        # Test with minimal instantiation to avoid LeRobot dependency
        policy = Pi05Policy.__new__(Pi05Policy)  # Create without __init__
        policy.device = None
        policy._policy = Mock()
        policy._policy.to.return_value = policy._policy  # Mock the to method to return self
        
        policy = policy.to_device('cpu')
        assert policy.device == 'cpu'

    def test_pi05_policy_mode_switching(self):
        """Test Pi05Policy mode switching methods."""
        # Test with minimal instantiation
        policy = Pi05Policy.__new__(Pi05Policy)
        policy._policy = Mock()
        
        # Test eval mode
        policy.set_eval_mode()
        policy._policy.eval.assert_called_once()
        
        # Reset mock and test train mode
        policy._policy.reset_mock()
        policy.set_train_mode()
        policy._policy.train.assert_called_once()

    def test_pi05_policy_reset(self):
        """Test Pi05Policy reset method."""
        policy = Pi05Policy.__new__(Pi05Policy)
        policy._policy = Mock()
        
        policy.reset()
        policy._policy.reset.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])