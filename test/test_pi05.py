import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from arkml.algos.vla.tokenizers.fast import FASTTokenizer
from arkml.algos.vla.pi05.models import Pi05Policy, flow_matching_loss, DummyBackbone, ActionFlowExpert
from arkml.algos.vla.pi05.trainer import Pi05Trainer
from arkml.algos.vla.pi05.evaluator import Pi05Evaluator


class TestFASTTokenizer:
    """Test the FAST tokenizer encode/decode functionality."""
    
    def test_encode_decode_roundtrip(self):
        """Test that encode/decode roundtrip preserves values within quantization error."""
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

    def test_encode_decode_edge_cases(self):
        """Test edge cases like boundary values and out-of-range inputs."""
        tokenizer = FASTTokenizer(vocab_path="", num_bins=100, min_val=-1.0, max_val=1.0)
        
        # Test boundary values
        boundary_actions = np.array([-1.0, 1.0])
        tokens = tokenizer.encode(boundary_actions)
        decoded_actions = tokenizer.decode(tokens)
        
        assert len(tokens) == 2
        assert np.allclose(boundary_actions, decoded_actions, atol=0.05)
        
        # Test out-of-range values (should be clipped)
        out_of_range_actions = np.array([-2.0, 2.0])
        tokens_clipped = tokenizer.encode(out_of_range_actions)
        decoded_clipped = tokenizer.decode(tokens_clipped)
        
        # Clipped values should be in range [-1, 1]
        assert np.all(decoded_clipped >= -1.0)
        assert np.all(decoded_clipped <= 1.0)


class TestPi05Policy:
    """Test the Pi05Policy model functionality."""
    
    def test_forward_output_shape(self):
        """Test that forward pass returns expected output shape."""
        # Create a simple Pi05Policy model
        model = Pi05Policy(
            policy_type="pi0.5",
            model_path="test_path",
            obs_dim=10,
            action_dim=8,
            image_dim=(3, 224, 224),
            pred_horizon=1
        )
        
        # Create dummy batch data
        batch_size = 2
        batch = {
            "image": torch.rand(batch_size, 3, 224, 224),
            "action": torch.rand(batch_size, 8),  # Continuous actions
        }
        
        # Test forward pass
        output = model.forward(batch)
        
        # Output should be a scalar loss tensor
        assert output.shape == torch.Size([])
        assert output.requires_grad  # Should be differentiable
        
        # Test with different batch sizes
        batch_large = {
            "image": torch.rand(4, 3, 224, 224),
            "action": torch.rand(4, 8),
        }
        output_large = model.forward(batch_large)
        assert output_large.shape == torch.Size([])
        assert output_large.requires_grad


class TestFlowMatchingLoss:
    """Test the flow matching loss function."""
    
    def test_backward_pass(self):
        """Test that flow matching loss supports backward pass."""
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


class TestPi05Trainer:
    """Test the Pi05Trainer functionality."""
    
    def test_pretrain_step(self):
        """Test pretrain step with dummy batch."""
        # Create model and dummy data
        model = Pi05Policy(
            policy_type="pi0.5",
            model_path="test_path",
            obs_dim=10,
            action_dim=8,
            image_dim=(3, 224, 224),
            pred_horizon=1
        )
        
        # Create a dummy dataset
        images = torch.rand(10, 3, 224, 224)
        target_tokens = torch.randint(0, 1000, (10, 50))  # 10 samples, 50 tokens each
        modality = ["fast_robot_actions"] * 10
        actions_cont = torch.rand(10, 8)
        
        dataset = TensorDataset(images, target_tokens, actions_cont)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Create a custom dataloader that yields the right format for training
        def custom_dataloader():
            for i in range(5):  # 5 batches
                yield {
                    "prefix_tokens": torch.rand(2, 150),  # Combined tokens
                    "target_tokens": torch.randint(0, 1000, (2, 10)),  # Target tokens
                    "modality": ["fast_robot_actions"] * 2,
                    "actions_cont": torch.rand(2, 8),
                }
        
        # Create trainer
        trainer = Pi05Trainer(
            model=model,
            dataloader=custom_dataloader(),
            device="cpu",
            lr=1e-4,
            weight_decay=0.01,
            num_epochs=1,
            grad_accum=1,
            output_dir="/tmp",
            use_bf16=False,
            val_dataloader=None,
            eval_every=1,
        )
        
        # Test pretrain step
        dummy_batch = {
            "prefix_tokens": torch.rand(2, 150),
            "target_tokens": torch.randint(0, 1000, (2, 10)),
            "modality": ["fast_robot_actions"],
            "actions_cont": torch.rand(2, 8),
        }
        
        loss = trainer.train_step_pretrain(dummy_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.requires_grad

    def test_posttrain_step(self):
        """Test posttrain step with dummy batch."""
        # Create model and dummy data
        model = Pi05Policy(
            policy_type="pi0.5",
            model_path="test_path",
            obs_dim=10,
            action_dim=8,
            image_dim=(3, 224, 224),
            pred_horizon=1
        )
        
        # Create trainer (reuse creation from pretrain test)
        def custom_dataloader():
            for i in range(5):  # 5 batches
                yield {
                    "prefix_tokens": torch.rand(2, 150),  # Combined tokens
                    "target_tokens": torch.randint(0, 1000, (2, 10)),  # Target tokens
                    "modality": ["fast_robot_actions"] * 2,
                    "actions_cont": torch.rand(2, 8),
                    "action": torch.rand(2, 8),  # For flow matching
                }
        
        trainer = Pi05Trainer(
            model=model,
            dataloader=custom_dataloader(),
            device="cpu",
            lr=1e-4,
            weight_decay=0.01,
            num_epochs=1,
            grad_accum=1,
            output_dir="/tmp",
            use_bf16=False,
            val_dataloader=None,
            eval_every=1,
            flow_alpha=10.0,
        )
        
        # Test posttrain step
        dummy_batch = {
            "prefix_tokens": torch.rand(2, 150),
            "target_tokens": torch.randint(0, 1000, (2, 10)),
            "modality": ["fast_robot_actions"],
            "actions_cont": torch.rand(2, 8),
            "action": torch.rand(2, 8),
        }
        
        loss = trainer.train_step_posttrain(dummy_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.requires_grad


class TestPi05Evaluator:
    """Test the Pi05Evaluator functionality."""
    
    def test_eval_subtask(self):
        """Test subtask evaluation."""
        # Create model
        model = Pi05Policy(
            policy_type="pi0.5",
            model_path="test_path",
            obs_dim=10,
            action_dim=8,
            image_dim=(3, 224, 224),
            pred_horizon=1
        )
        
        # Create evaluator (note: evaluator needs dataloader but we'll test methods separately)
        evaluator = Pi05Evaluator(model, None, "cpu")
        
        # Test subtask evaluation
        predicted_subtasks = torch.rand(5, 32000)  # 5 samples, 32k vocab
        ground_truth_subtasks = torch.randint(0, 32000, (5,))  # 5 ground truth tokens
        
        metrics = evaluator.eval_subtask(predicted_subtasks, ground_truth_subtasks)
        
        assert "subtask_accuracy" in metrics
        assert "total_evaluated" in metrics
        assert 0.0 <= metrics["subtask_accuracy"] <= 1.0
        assert metrics["total_evaluated"] == 5

    def test_eval_actions(self):
        """Test action evaluation."""
        # Create model
        model = Pi05Policy(
            policy_type="pi0.5",
            model_path="test_path",
            obs_dim=10,
            action_dim=8,
            image_dim=(3, 224, 224),
            pred_horizon=1
        )
        
        evaluator = Pi05Evaluator(model, None, "cpu")
        
        # Test action evaluation
        hidden_states = torch.rand(3, 512)  # 3 samples, 512-dim hidden state
        ground_truth_actions = torch.rand(3, 8)  # 3 samples, 8-dim actions
        
        metrics = evaluator.eval_actions(hidden_states, ground_truth_actions)
        
        assert "action_mse" in metrics
        assert "action_mae" in metrics
        assert "action_accuracy_within_threshold" in metrics
        assert "threshold" in metrics
        assert "total_evaluated" in metrics
        
        assert isinstance(metrics["action_mse"], float)
        assert isinstance(metrics["action_mae"], float)
        assert 0.0 <= metrics["action_accuracy_within_threshold"] <= 1.0
        assert metrics["total_evaluated"] == 3


if __name__ == "__main__":
    pytest.main([__file__])