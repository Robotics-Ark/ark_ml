"""
Pi0.5 Quick Start Example

This is a minimal example showing how to use Pi0.5 for inference.
"""

import torch
from arkml.algos.vla.pi05.models import Pi05Policy


def example_inference():
    """Example of loading and using Pi0.5 model."""
    
    print("=" * 50)
    print("Pi0.5 Quick Start Example")
    print("=" * 50)
    
    # 1. Initialize the model
    # NOTE: Replace 'path/to/your/model' with actual model path
    print("1. Loading Pi0.5 model...")
    
    try:
        policy = Pi05Policy(
            policy_type='pi0.5',
            model_path='path/to/your/pi05/model',  # ← Replace with your model path
            backbone_type='siglip_gemma',  # Vision-language backbone
            use_fast_tokens=True,          # Use FAST tokenization
            use_flow_matching=True,        # Use flow matching
            obs_dim=9,                     # Observation dimension
            action_dim=8,                  # Action dimension
            image_dim=(3, 224, 224),      # Image dimensions
            pred_horizon=1                 # Prediction horizon
        )
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"⚠ Model loading failed (expected for missing weights): {e}")
        print("  This is normal - provide actual model path to load weights")
        print()
        return
    
    # 2. Move to device
    print("2. Moving model to device...")
    policy = policy.to_device('cuda' if torch.cuda.is_available() else 'cpu')
    print("✓ Model moved to device")
    
    # 3. Set to evaluation mode
    print("3. Setting evaluation mode...")
    policy.set_eval_mode()
    print("✓ Evaluation mode set")
    
    # 4. Prepare observation
    print("4. Preparing observation...")
    observation = {
        'image': torch.randn(1, 3, 224, 224),  # Batch size 1, 3 channels, 224x224
        'state': torch.randn(9),               # 9-dimensional state vector
        'task': 'Pick up the object and place it'  # Task instruction
    }
    print("✓ Observation prepared")
    
    # 5. Make prediction
    print("5. Making prediction...")
    action = policy.predict(observation)
    print(f"✓ Action predicted: shape {action.shape}")
    print(f"  Action values: {action.detach().cpu().numpy()}")
    
    # 6. Multiple predictions example
    print("6. Multiple action prediction...")
    actions = policy.predict_n_actions(observation, n_actions=3)
    print(f"✓ Multiple actions: shape {actions.shape}")
    
    print()
    print("=" * 50)
    print("✅ Pi0.5 Example Completed Successfully!")
    print("🔧 Ready for your actual model and data")
    print("=" * 50)


def example_training_config():
    """Example of training configuration."""
    
    print("\\n" + "=" * 50)
    print("Pi0.5 Training Configuration Example")
    print("=" * 50)
    
    from omegaconf import DictConfig
    
    # Training configuration example
    config = DictConfig({
        'trainer': {
            'lr': 2e-4,           # Learning rate
            'batch_size': 8,      # Batch size
            'max_epochs': 10,     # Maximum epochs
            'weight_decay': 0.01, # Weight decay
            'num_workers': 4,     # Data loader workers
            'use_bf16': True      # Use bfloat16 precision
        },
        'training': {
            'stage': 'pretrain',      # 'pretrain' or 'posttrain'
            'flow_alpha': 10.0,       # Flow matching loss weight
            'pretrain_steps': 280000, # Steps for pretraining
            'posttrain_steps': 80000, # Steps for post-training
            'integration_steps': 10   # Euler integration steps
        },
        'model': {
            'backbone_type': 'siglip_gemma',
            'use_fast_tokens': True,
            'use_flow_matching': True,
            'obs_dim': 9,
            'action_dim': 8,
            'image_dim': [3, 480, 640]
        }
    })
    
    print("Training Configuration:")
    print(f"  Stage: {config.training.stage}")
    print(f"  Learning Rate: {config.trainer.lr}")
    print(f"  Flow Alpha: {config.training.flow_alpha}")
    print(f"  Backbone: {config.model.backbone_type}")
    print("✓ Configuration example ready")
    
    print("=" * 50)


if __name__ == "__main__":
    # Run the examples
    example_inference()
    example_training_config()
    
    print("\\n💡 Next steps:")
    print("1. Replace 'path/to/your/pi05/model' with actual model path")
    print("2. Use Hugging Face model ID or local path to model weights")
    print("3. Adjust obs_dim, action_dim based on your robot/env")
    print("4. Run: python run_pi05.py --model-path <your-model-path>")