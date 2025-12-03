"""
Pi0.5 Inference Script

This script demonstrates how to load a Pi0.5 model and run inference.
"""

import torch
import argparse
from arkml.algos.vla.pi05.models import Pi05Policy


def main():
    parser = argparse.ArgumentParser(description='Run Pi0.5 Inference')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to Pi0.5 model (HuggingFace Hub ID or local path)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on')
    parser.add_argument('--image-height', type=int, default=224,
                        help='Input image height')
    parser.add_argument('--image-width', type=int, default=224,
                        help='Input image width')
    parser.add_argument('--action-dim', type=int, default=8,
                        help='Action dimension')
    parser.add_argument('--obs-dim', type=int, default=9,
                        help='Observation dimension')
    parser.add_argument('--backbone-type', type=str, default='siglip_gemma',
                        help='Vision-language backbone type')
    
    args = parser.parse_args()
    
    print(f"Loading Pi0.5 model from: {args.model_path}")
    print(f"Using device: {args.device}")
    
    try:
        # Initialize the Pi0.5 policy
        policy = Pi05Policy(
            policy_type='pi0.5',
            model_path=args.model_path,
            backbone_type=args.backbone_type,
            use_fast_tokens=True,
            use_flow_matching=True,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            image_dim=(3, args.image_height, args.image_width),
            pred_horizon=1
        )
        
        print("✓ Model loaded successfully!")
        
        # Move to device
        policy = policy.to_device(args.device)
        policy.set_eval_mode()
        
        print(f"✓ Model moved to {args.device}")
        print("✓ Evaluation mode set")
        
        # Example inference with random data
        print("\\nRunning example inference...")
        
        # Create example observation
        example_obs = {
            'image': torch.randn(1, 3, args.image_height, args.image_width).to(args.device),
            'state': torch.randn(args.obs_dim).to(args.device),
            'task': 'Perform manipulation task'
        }
        
        # Make prediction
        action = policy.predict(example_obs)
        print(f"✓ Action predicted successfully: {action.shape}")
        print(f"Action values: {action.detach().cpu().numpy()}")
        
        # Example with multiple predictions
        print("\\nTesting multiple predictions...")
        actions = policy.predict_n_actions(example_obs, n_actions=5)
        print(f"✓ Multiple actions predicted: {actions.shape}")
        
        print("\\n🎉 Pi0.5 inference script completed successfully!")
        print("Model is ready for use with your actual data!")
        
    except Exception as e:
        print(f"✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()


def run_with_config(config_path=None, model_path=None):
    """
    Alternative function to run Pi0.5 with configuration file.
    
    Args:
        config_path: Path to configuration file
        model_path: Model path (overrides config if provided)
    """
    import yaml
    from omegaconf import OmegaConf
    
    if config_path:
        # Load configuration
        cfg = OmegaConf.load(config_path)
    else:
        # Use default configuration
        cfg = OmegaConf.create({
            'model': {
                'model_path': model_path or 'path/to/your/model',
                'backbone_type': 'siglip_gemma',
                'use_fast_tokens': True,
                'use_flow_matching': True,
                'obs_dim': 9,
                'action_dim': 8,
                'image_dim': [3, 224, 224],
                'pred_horizon': 1
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        })
    
    if model_path:
        cfg.model.model_path = model_path
    
    try:
        # Initialize policy with config
        policy = Pi05Policy(
            policy_type='pi0.5',
            model_path=cfg.model.model_path,
            backbone_type=cfg.model.backbone_type,
            use_fast_tokens=cfg.model.use_fast_tokens,
            use_flow_matching=cfg.model.use_flow_matching,
            obs_dim=cfg.model.obs_dim,
            action_dim=cfg.model.action_dim,
            image_dim=tuple(cfg.model.image_dim),
            pred_horizon=cfg.model.pred_horizon
        )
        
        # Move to device and set eval mode
        policy = policy.to_device(cfg.device)
        policy.set_eval_mode()
        
        print(f"✓ Model loaded from config: {cfg.model.model_path}")
        print(f"✓ Using device: {cfg.device}")
        
        return policy
        
    except Exception as e:
        print(f"✗ Error loading model with config: {e}")
        raise


if __name__ == "__main__":
    main()