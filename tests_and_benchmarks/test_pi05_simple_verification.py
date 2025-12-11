"""
Simplified verification tests for Pi0.5 implementation
"""

import pytest
import torch
from unittest.mock import Mock, patch


def test_pi05_core_functionality():
    """Test the core functionality of the Pi05 wrapper"""
    with patch('arkml.algos.vla.pi05.models.LeRobotPI05Policy') as mock_policy_class:
        # Setup mock policy
        mock_policy = Mock()
        mock_policy.config = Mock()
        mock_policy.config.n_action_steps = 1
        mock_policy.config.use_fast_tokens = True
        mock_policy.config.use_flow_matching = True
        mock_policy.config.backbone_type = 'siglip_gemma'
        mock_policy.forward.return_value = (torch.tensor(0.5, requires_grad=True), {})
        mock_policy.select_action.return_value = torch.randn(1, 8)
        mock_policy.reset.return_value = None
        mock_policy.eval.return_value = None
        mock_policy.train.return_value = None
        mock_policy.to.return_value = mock_policy
        mock_policy.config.input_features = {}
        mock_policy.config.output_features = {}
        
        mock_policy_class.from_pretrained.return_value = mock_policy
        
        # Mock context
        with patch('arkml.core.app_context.ArkMLContext') as mock_context:
            mock_context.visual_input_features = ['image']
            
            # Import and create policy
            from arkml.algos.vla.pi05.models import Pi05Policy
            
            # Mock ArkMLContext in the models module
            import arkml.algos.vla.pi05.models
            mock_context_obj = Mock()
            mock_context_obj.visual_input_features = ['image']
            arkml.algos.vla.pi05.models.ArkMLContext = mock_context_obj
            
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
            
            assert hasattr(policy, 'predict')
            assert hasattr(policy, 'forward')
            assert hasattr(policy, 'to_device')
            assert policy.obs_dim == 9
            assert policy.action_dim == 8
            assert policy.image_dim == (3, 224, 224)


def test_pi05_backward_compatibility():
    """Test that Pi05 and PiZero can coexist"""
    # Mock both models
    with patch('arkml.algos.vla.pizero.models.PI0Policy') as mock_pizero_class, \
         patch('arkml.algos.vla.pi05.models.LeRobotPI05Policy') as mock_pi05_class:
        
        # Setup mock PiZero
        mock_pizero_policy = Mock()
        mock_pizero_policy.config = Mock()
        mock_pizero_policy.config.n_action_steps = 1
        mock_pizero_policy.forward.return_value = (torch.tensor(0.3), {})
        mock_pizero_policy.select_action.return_value = torch.randn(1, 8)
        mock_pizero_policy.reset.return_value = None
        mock_pizero_policy.eval.return_value = None
        mock_pizero_policy.train.return_value = None
        mock_pizero_policy.to.return_value = mock_pizero_policy
        mock_pizero_policy.config.input_features = {}
        mock_pizero_policy.config.output_features = {}
        
        mock_pizero_class.from_pretrained.return_value = mock_pizero_policy
        
        # Setup mock Pi05
        mock_pi05_policy = Mock()
        mock_pi05_policy.config = Mock()
        mock_pi05_policy.config.n_action_steps = 1
        mock_pi05_policy.config.use_fast_tokens = True
        mock_pi05_policy.config.use_flow_matching = True
        mock_pi05_policy.config.backbone_type = 'siglip_gemma'
        mock_pi05_policy.forward.return_value = (torch.tensor(0.5), {})
        mock_pi05_policy.select_action.return_value = torch.randn(1, 8)
        mock_pi05_policy.reset.return_value = None
        mock_pi05_policy.eval.return_value = None
        mock_pi05_policy.train.return_value = None
        mock_pi05_policy.to.return_value = mock_pi05_policy
        mock_pi05_policy.config.input_features = {}
        mock_pi05_policy.config.output_features = {}
        
        mock_pi05_class.from_pretrained.return_value = mock_pi05_policy
        
        # Test both can be instantiated with proper context mocking
        with patch('arkml.core.app_context.ArkMLContext') as mock_context:
            mock_context.visual_input_features = ['image']
            
            # Import both models
            from arkml.algos.vla.pizero.models import PiZeroNet
            from arkml.algos.vla.pi05.models import Pi05Policy
            
            # Mock contexts for both
            import arkml.algos.vla.pizero.models
            import arkml.algos.vla.pi05.models
            mock_context_obj = Mock()
            mock_context_obj.visual_input_features = ['image']
            arkml.algos.vla.pizero.models.ArkMLContext = mock_context_obj
            arkml.algos.vla.pi05.models.ArkMLContext = mock_context_obj
            
            # Create both
            pizero = PiZeroNet(
                policy_type='pi0',
                model_path='test_path',
                obs_dim=9,
                action_dim=8,
                image_dim=(3, 224, 224),
                pred_horizon=1
            )
            
            pi05 = Pi05Policy(
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
            
            assert pizero is not None
            assert pi05 is not None
            assert hasattr(pizero, 'predict')
            assert hasattr(pi05, 'predict')


def test_pi05_prediction():
    """Test prediction functionality"""
    with patch('arkml.algos.vla.pi05.models.LeRobotPI05Policy') as mock_policy_class:
        # Setup mock policy
        mock_policy = Mock()
        mock_policy.config = Mock()
        mock_policy.config.n_action_steps = 1
        mock_policy.config.use_fast_tokens = True
        mock_policy.config.use_flow_matching = True
        mock_policy.config.backbone_type = 'siglip_gemma'  
        mock_policy.forward.return_value = (torch.tensor(0.5, requires_grad=True), {})
        mock_policy.select_action.return_value = torch.randn(1, 8)  # Return 1x8 tensor
        mock_policy.reset.return_value = None
        mock_policy.eval.return_value = None
        mock_policy.train.return_value = None
        mock_policy.to.return_value = mock_policy
        mock_policy.config.input_features = {}
        mock_policy.config.output_features = {}
        
        mock_policy_class.from_pretrained.return_value = mock_policy
        
        with patch('arkml.core.app_context.ArkMLContext') as mock_context:
            mock_context.visual_input_features = ['image']
            
            from arkml.algos.vla.pi05.models import Pi05Policy
            
            import arkml.algos.vla.pi05.models
            mock_context_obj = Mock()
            mock_context_obj.visual_input_features = ['image']
            arkml.algos.vla.pi05.models.ArkMLContext = mock_context_obj
            
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
            
            # Test prediction
            obs = {
                'image': torch.randn(1, 3, 224, 224),
                'state': torch.randn(9),
                'task': 'test task'
            }
            
            action = policy.predict(obs)
            assert isinstance(action, torch.Tensor)
            # Should be compatible with the action_dim
            assert action.shape[-1] == 8  # Last dimension should match action_dim


def test_pi05_forward_pass():
    """Test forward pass functionality"""
    with patch('arkml.algos.vla.pi05.models.LeRobotPI05Policy') as mock_policy_class:
        # Setup mock policy
        mock_policy = Mock()
        mock_policy.config = Mock()
        mock_policy.config.n_action_steps = 1
        mock_policy.config.use_fast_tokens = True
        mock_policy.config.use_flow_matching = True
        mock_policy.config.backbone_type = 'siglip_gemma'
        mock_policy.forward.return_value = (torch.tensor(0.5, requires_grad=True), {})
        mock_policy.select_action.return_value = torch.randn(1, 8)
        mock_policy.reset.return_value = None
        mock_policy.eval.return_value = None
        mock_policy.train.return_value = None
        mock_policy.to.return_value = mock_policy
        mock_policy.config.input_features = {}
        mock_policy.config.output_features = {}
        
        mock_policy_class.from_pretrained.return_value = mock_policy
        
        with patch('arkml.core.app_context.ArkMLContext') as mock_context:
            mock_context.visual_input_features = ['image']
            
            from arkml.algos.vla.pi05.models import Pi05Policy
            
            import arkml.algos.vla.pi05.models
            mock_context_obj = Mock()
            mock_context_obj.visual_input_features = ['image']
            arkml.algos.vla.pi05.models.ArkMLContext = mock_context_obj
            
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
            
            # Test forward pass
            batch = {
                'observation.images.image': torch.randn(2, 3, 224, 224),
                'action': torch.randn(2, 8)
            }
            
            loss = policy.forward(batch)
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == torch.Size([])  # scalar
            assert loss.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])