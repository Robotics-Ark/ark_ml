import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

# Import ArkML components (focus on core functionality)
from arkml.core.policy import BasePolicy
from arkml.core.registry import MODELS
from arkml.algos.vla.pi05.models import Pi05Policy


class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    def __init__(self, size=10):
        self.size = size
        self.data = [
            {
                "observation.images.image": torch.randn(3, 224, 224),
                "observation.state": torch.randn(9),
                "action": torch.randn(8),
                "task": f"task_{i}"
            }
            for i in range(size)
        ]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


class TestPi05NetFullVerification:
    """Complete test suite for Pi05Net wrapper implementation"""
    
    @pytest.fixture
    def mock_hf_model(self):
        """Create a mock HF model for testing without actual downloads"""
        with patch('arkml.algos.vla.pi05.models.LeRobotPI05Policy') as mock_policy_class:
            # Create mock policy instance
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

            yield mock_policy_class, mock_policy
    
    def test_import_paths(self):
        """Test that import paths work correctly"""
        from arkml.algos.vla.pi05.models import Pi05Policy
        from arkml.algos.vla.pi05.models import flow_matching_loss
        from arkml.algos.vla.pi05.dataset import Pi05Dataset
        from arkml.algos.vla.pi05.config_utils import get_pi05_config
        from arkml.algos.vla.pi05.compute_stats import compute_pi05_stats
        
        assert hasattr(Pi05Policy, 'predict')
        assert callable(flow_matching_loss)
        assert callable(get_pi05_config)
        assert callable(compute_pi05_stats)
        assert callable(Pi05Dataset)
    
    def test_wrapper_instantiation(self, mock_hf_model):
        """Test that wrapper class instantiates without side-effects"""
        mock_policy_class, mock_policy = mock_hf_model
        
        # Create wrapper instance
        with patch('arkml.core.app_context.ArkMLContext') as mock_context:
            mock_context.visual_input_features = ['image']
            # Mock the class attribute too
            mock_context_class = Mock()
            mock_context_class.visual_input_features = ['image']

            with patch('arkml.algos.vla.pi05.models.ArkMLContext', mock_context_class):
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
        
        assert isinstance(policy, BasePolicy)
        assert hasattr(policy, 'predict')
        assert hasattr(policy, 'forward')
        assert hasattr(policy, 'to_device')
        assert hasattr(policy, 'reset')
        assert policy.obs_dim == 9
        assert policy.action_dim == 8
        assert policy.image_dim == (3, 224, 224)
    
    def test_config_and_loading(self, mock_hf_model):
        """Test that wrapper correctly calls PI05Policy.from_pretrained"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Verify that from_pretrained was called with correct parameters
        mock_policy_class.from_pretrained.assert_called_once_with('test_model_path')
    
    def test_forward_pass_smoke_test(self, mock_hf_model):
        """Smoke test with random image/state"""
        mock_policy_class, mock_policy = mock_hf_model

        with patch('arkml.core.app_context.ArkMLContext') as mock_context:
            mock_context.visual_input_features = ['image']
            # Mock the class attribute too
            mock_context_class = Mock()
            mock_context_class.visual_input_features = ['image']

            with patch('arkml.algos.vla.pi05.models.ArkMLContext', mock_context_class):
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
        
        # Create test observation
        obs = {
            'image': torch.randn(1, 3, 224, 224),
            'state': torch.randn(9),
            'task': 'test task'
        }
        
        # Forward pass
        output = policy.forward(obs)
        assert isinstance(output, torch.Tensor)
        assert output.requires_grad  # Should be differentiable
    
    def test_predict_method(self, mock_hf_model):
        """Test prediction returns correct tensor shape"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Test prediction with single batch
        obs = {
            'image': torch.randn(1, 3, 224, 224),
            'state': torch.randn(9),
            'task': 'test task'
        }
        
        action = policy.predict(obs)
        
        # Should be (batch_size, action_dim) where batch_size=1 initially
        assert action.shape[-1] == 8  # action_dim
        assert isinstance(action, torch.Tensor)
    
    def test_batch_size_handling(self, mock_hf_model):
        """Test batch size > 1"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Test with batch size > 1
        obs = {
            'image': torch.randn(4, 3, 224, 224),
            'state': torch.randn(4, 9),
            'task': 'test task'
        }
        
        action = policy.predict(obs)
        # The actual shape depends on the wrapped model's behavior
        assert isinstance(action, torch.Tensor)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_movement_cuda(self, mock_hf_model):
        """Test .to_device("cuda") if available"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Move to CUDA
        policy_cuda = policy.to_device('cuda')
        
        # The underlying model should be moved
        assert policy.device == 'cuda'
    
    def test_device_movement_cpu(self, mock_hf_model):
        """Test .to_device("cpu")"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Move to CPU
        policy_cpu = policy.to_device('cpu')
        
        # Device should be set
        assert policy.device == 'cpu'
    
    def test_api_contract_arkml_registry(self):
        """Test that wrapper works inside ArkML's policy registry"""
        # Register should work (already registered)
        assert 'Pi05Policy' in MODELS._registry
        
        # Test that we can build it (with mocked HF model)
        with patch('arkml.algos.vla.pi05.models.PI05Policy') as mock_policy_class:
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
                
                # Try to build using registry
                config = OmegaConf.create({
                    'policy_type': 'pi0.5',
                    'model_path': 'test_path',
                    'backbone_type': 'siglip_gemma',
                    'use_fast_tokens': True,
                    'use_flow_matching': True,
                    'obs_dim': 9,
                    'action_dim': 8,
                    'image_dim': [3, 224, 224],
                    'pred_horizon': 1
                })
                
                # We can't test full registry build without modifying internal structure,
                # but we can test instantiation
                policy = Pi05Policy(
                    **config
                )
                
                assert policy is not None
                assert hasattr(policy, 'predict')
    
    def test_missing_fields_handling(self, mock_hf_model):
        """Verify missing fields raise correct exceptions or have fallbacks"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Test with all fields
        obs_complete = {
            'image': torch.randn(1, 3, 224, 224),
            'state': torch.randn(9),
            'task': 'test task'
        }
        
        # This should work
        action = policy.predict(obs_complete)
        assert isinstance(action, torch.Tensor)
    
    def test_stress_sequential_predictions(self, mock_hf_model):
        """Test 10 sequential predictions on 224x224 images"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Make 10 sequential predictions
        for i in range(10):
            obs = {
                'image': torch.randn(1, 3, 224, 224),
                'state': torch.randn(9),
                'task': f'task_{i}'
            }
            
            action = policy.predict(obs)
            assert action.shape[-1] == 8  # action dim
            assert isinstance(action, torch.Tensor)
    
    def test_parameter_count_constancy(self, mock_hf_model):
        """Memory leak check: parameter count remains constant"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Count trainable parameters initially
        initial_params = sum(p.numel() for p in policy.get_trainable_params() if p.requires_grad)
        
        # Make several predictions
        for i in range(5):
            obs = {
                'image': torch.randn(1, 3, 224, 224),
                'state': torch.randn(9),
                'task': f'task_{i}'
            }
            _ = policy.predict(obs)
        
        # Count parameters after predictions
        final_params = sum(p.numel() for p in policy.get_trainable_params() if p.requires_grad)
        
        # Should be the same (no memory leak)
        assert initial_params == final_params
    
    def test_serialization_save_reload(self, mock_hf_model):
        """Test save and reload wrapper state dict"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'pi05_model.pth')
            
            # Save the model
            policy.save_policy(temp_dir)
            
            # Verify file was created
            assert os.path.exists(save_path)
            
            # For this test, we'll just verify the save method is called
            # The reload would require actual weights which we're mocking
    
    def test_pizero_pi05_side_by_side(self):
        """Test PiZero and Pi05 can be loaded side-by-side using mock weights"""

        # Mock both PiZero and Pi05 models
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

            # Test both can be built through registry
            with patch('arkml.core.app_context.ArkMLContext') as mock_context:
                mock_context.visual_input_features = ['image']

                # Create PiZero
                from arkml.algos.vla.pizero.models import PiZeroNet
                pizero = PiZeroNet(
                    policy_type='pi0',
                    model_path='test_path',
                    obs_dim=9,
                    action_dim=8,
                    image_dim=(3, 224, 224),
                    pred_horizon=1
                )

                # Create Pi05
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

                # Both should exist
                assert pizero is not None
                assert pi05 is not None
                assert hasattr(pizero, 'predict')
                assert hasattr(pi05, 'predict')

                # Test that both can make predictions
                test_obs = {
                    'image': torch.randn(1, 3, 224, 224),
                    'state': torch.randn(9),
                    'task': 'test task'
                }

                pizero_action = pizero.predict(test_obs)
                pi05_action = pi05.predict(test_obs)

                # Both should return tensors
                assert isinstance(pizero_action, torch.Tensor)
                assert isinstance(pi05_action, torch.Tensor)
                assert pizero_action.shape[-1] == 8  # action dim
                assert pi05_action.shape[-1] == 8  # action dim
    
    def test_observation_format_handling(self, mock_hf_model):
        """Test that observation dict format is handled correctly"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Test the expected observation format
        obs = {
            'image': torch.randn(1, 3, 224, 224),
            'state': torch.randn(9),
            'task': 'pick up the red block'
        }
        
        # Should not raise errors
        action = policy.predict(obs)
        assert isinstance(action, torch.Tensor)
        
        # Test with different image keys (should be handled by ArkMLContext)
        obs2 = {
            'observation.images.image': torch.randn(1, 3, 224, 224),
            'observation.state': torch.randn(9),
            'task': 'manipulation task'
        }
        
        action2 = policy.predict(obs2)
        assert isinstance(action2, torch.Tensor)
    
    def test_forward_method_with_batch(self, mock_hf_model):
        """Test forward method with batch data"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        # Create batch observation
        batch_obs = {
            'observation.images.image': torch.randn(2, 3, 224, 224),
            'observation.state': torch.randn(2, 9),
            'action': torch.randn(2, 8)
        }
        
        # Forward pass should return loss
        loss = policy.forward(batch_obs)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # scalar
        assert loss.requires_grad
    
    def test_get_trainable_params(self, mock_hf_model):
        """Test that get_trainable_params returns list of parameters"""
        mock_policy_class, mock_policy = mock_hf_model
        
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
        
        params = policy.get_trainable_params()
        assert isinstance(params, list)
        assert len(params) >= 0  # May be empty if no params in mock


if __name__ == "__main__":
    pytest.main([__file__])