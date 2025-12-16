"""
Comprehensive tests for Pi0.5 models.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from arkml.algos.vla.pi05.models import Pi05Policy, flow_matching_loss, ActionFlowExpert


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

    def test_pi05_policy_mock_integration(self):
        """Test Pi05Policy with mocked LeRobot integration."""
        from unittest.mock import Mock, patch
        import torch

        # Setup mock for the LeRobot policy
        mock_le_robot_policy = Mock()
        mock_le_robot_policy.config = Mock()
        mock_le_robot_policy.config.n_action_steps = 1
        mock_le_robot_policy.config.use_fast_tokens = True
        mock_le_robot_policy.config.use_flow_matching = True
        mock_le_robot_policy.config.backbone_type = 'siglip_gemma'
        mock_le_robot_policy.forward.return_value = (torch.tensor(0.5, requires_grad=True), {})
        mock_le_robot_policy.select_action.return_value = torch.randn(1, 8)
        mock_le_robot_policy.reset.return_value = None
        mock_le_robot_policy.eval.return_value = None
        mock_le_robot_policy.train.return_value = None
        mock_le_robot_policy.to.return_value = mock_le_robot_policy
        mock_le_robot_policy.config.input_features = {}
        mock_le_robot_policy.config.output_features = {}

        with patch('arkml.algos.vla.pi05.models.LeRobotPI05Policy') as mock_class:
            mock_class.from_pretrained.return_value = mock_le_robot_policy

            # Test policy creation with mocked context
            with patch('arkml.core.app_context.ArkMLContext') as mock_context:
                mock_context.visual_input_features = ['image']

                # Mock the class attribute too
                mock_context_class = Mock()
                mock_context_class.visual_input_features = ['image']

                with patch('arkml.algos.vla.pi05.models.ArkMLContext', mock_context_class):
                    policy = Pi05Policy(
                        policy_type='pi0.5',
                        model_path='test_model_path',
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
                    assert policy.image_dim == (3, 224, 224)
                    assert policy._policy is mock_le_robot_policy

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

    def test_pi05_policy_mock_integration(self):
        """Test Pi05Policy with mocked LeRobot integration."""
        from unittest.mock import Mock, patch
        import torch

        # Setup mock for the LeRobot policy
        mock_le_robot_policy = Mock()
        mock_le_robot_policy.config = Mock()
        mock_le_robot_policy.config.n_action_steps = 1
        mock_le_robot_policy.config.use_fast_tokens = True
        mock_le_robot_policy.config.use_flow_matching = True
        mock_le_robot_policy.config.backbone_type = 'siglip_gemma'
        mock_le_robot_policy.forward.return_value = (torch.tensor(0.5, requires_grad=True), {})
        mock_le_robot_policy.select_action.return_value = torch.randn(1, 8)
        mock_le_robot_policy.reset.return_value = None
        mock_le_robot_policy.eval.return_value = None
        mock_le_robot_policy.train.return_value = None
        mock_le_robot_policy.to.return_value = mock_le_robot_policy
        mock_le_robot_policy.config.input_features = {}
        mock_le_robot_policy.config.output_features = {}

        with patch('arkml.algos.vla.pi05.models.LeRobotPI05Policy') as mock_class:
            mock_class.from_pretrained.return_value = mock_le_robot_policy

            # Test policy creation with mocked context
            with patch('arkml.core.app_context.ArkMLContext') as mock_context:
                mock_context.visual_input_features = ['image']

                # Mock the class attribute too
                mock_context_class = Mock()
                mock_context_class.visual_input_features = ['image']

                with patch('arkml.algos.vla.pi05.models.ArkMLContext', mock_context_class):
                    policy = Pi05Policy(
                        policy_type='pi0.5',
                        model_path='test_model_path',
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
                    assert policy.image_dim == (3, 224, 224)
                    assert policy._policy is mock_le_robot_policy

    def test_pi05_policy_forward_pass(self):
        """Test Pi05Policy forward pass with mocked LeRobot."""
        from unittest.mock import Mock, patch
        import torch

        # Setup mock for the LeRobot policy
        mock_le_robot_policy = Mock()
        mock_le_robot_policy.forward.return_value = (torch.tensor(0.5, requires_grad=True), {})
        mock_le_robot_policy.config = Mock()
        mock_le_robot_policy.config.input_features = {}
        mock_le_robot_policy.config.output_features = {}

        with patch('arkml.algos.vla.pi05.models.LeRobotPI05Policy') as mock_class:
            mock_class.from_pretrained.return_value = mock_le_robot_policy

            with patch('arkml.core.app_context.ArkMLContext') as mock_context:
                mock_context.visual_input_features = ['image']

                # Mock the class attribute too
                mock_context_class = Mock()
                mock_context_class.visual_input_features = ['image']

                with patch('arkml.algos.vla.pi05.models.ArkMLContext', mock_context_class):
                    policy = Pi05Policy(
                        policy_type='pi0.5',
                        model_path='test_model_path',
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
                    # Should be the tensor value, not .item() since it's the loss tensor
                    assert loss.requires_grad

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