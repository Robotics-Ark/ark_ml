# Pi0.5 Implementation

This directory contains the complete Pi0.5 implementation following the HuggingFace wrapper pattern for the Ark ML framework.

## Architecture Overview

Pi0.5 is an advanced Vision-Language-Action model that implements:
- **Multi-stage training**: Pretraining (CE(text) + CE(FAST tokens)) and Post-training (CE(subtask) + α × flow_matching_loss)
- **Flow matching**: For precise action prediction using vector field networks
- **Multiple prediction heads**: Subtask, FAST, and flow heads
- **Enhanced backbone**: Support for SigLIP-Gemma vision-language architecture

## Directory Structure

```
pi05/
├── models.py           # Core Pi0.5 policy (HuggingFace wrapper)
├── algorithm.py        # Training algorithm
├── trainer.py          # Multi-stage trainer
├── evaluator.py        # Evaluation metrics
├── dataset.py          # Multi-modality dataset
├── config_utils.py     # Configuration utilities
├── compute_stats.py    # Statistics computation
├── utils.py           # Utility functions
└── README.md          # This file
```

## Usage Instructions

### 1. Loading a Pre-trained Model

```python
from arkml.algos.vla.pi05.models import Pi05Policy

# Load from Hugging Face Hub or local path
policy = Pi05Policy(
    policy_type='pi0.5',
    model_path='your-huggingface-username/pi05-model',  # or local path
    backbone_type='siglip_gemma',  # Vision-language backbone
    use_fast_tokens=True,          # Enable FAST tokenization
    use_flow_matching=True,        # Enable flow matching
    obs_dim=9,                     # Observation dimension
    action_dim=8,                  # Action dimension  
    image_dim=(3, 480, 640),      # Image dimensions (C, H, W)
    pred_horizon=1                 # Prediction horizon
)

# Move to device
policy = policy.to_device('cuda')
```

### 2. Making Predictions

```python
import torch

# Prepare observation dictionary
observation = {
    'image': torch.randn(1, 3, 224, 224),  # Image tensor
    'state': torch.randn(9),               # State vector
    'task': 'pick up the red block'        # Task instruction (optional)
}

# Get action prediction
action = policy.predict(observation)
print(f"Predicted action: {action}")
```

### 3. Training a New Model

```python
from arkml.algos.vla.pi05.algorithm import Pi05Algorithm
from arkml.algos.vla.pi05.dataset import create_pi05_dataloader
from omegaconf import DictConfig

# Create your dataset and dataloader
train_dataloader = create_pi05_dataloader(
    dataset_path='path/to/your/dataset',
    batch_size=8,
    shuffle=True
)

# Load your policy
policy = Pi05Policy(
    policy_type='pi0.5',
    model_path='path/to/pretrained/model',  # Or use a base model
    # ... other parameters
)

# Configure training
config = DictConfig({
    'trainer': {
        'lr': 2e-4,
        'batch_size': 8,
        'max_epochs': 10,
        'weight_decay': 0.01,
        'num_workers': 4,
        'use_bf16': True
    },
    'training': {
        'stage': 'pretrain',      # 'pretrain' or 'posttrain'
        'flow_alpha': 10.0,       # Weight for flow matching loss
        'pretrain_steps': 280000, # Steps for pretraining
        'posttrain_steps': 80000  # Steps for post-training
    }
})

# Create algorithm and train
algorithm = Pi05Algorithm(policy=policy, device='cuda', cfg=config)
results = algorithm.train(train_dataset=your_train_dataset)
```

### 4. Configuration Options

Key configuration parameters:

- `backbone_type`: Vision-language backbone ('siglip_gemma', etc.)
- `use_fast_tokens`: Whether to use FAST tokenization for action discretization
- `use_flow_matching`: Whether to use flow matching for action prediction
- `training_stage`: 'pretrain' or 'posttrain' for multi-stage training
- `flow_alpha`: Weight for flow matching loss (default: 10.0)

## Training Stages

Pi0.5 supports multi-stage training:

### Pretraining Stage
```
CE(text) + CE(FAST tokens)
```
- Focuses on learning foundational representations
- Uses multiple modalities and FAST tokenization

### Post-training Stage  
```
CE(subtask) + α × flow_matching_loss
```
- Refines the model with flow matching and subtask prediction
- Enables precise action prediction using flow matching

## Evaluation Metrics

The evaluator provides comprehensive metrics:
- Action MSE and MAE
- Accuracy within threshold
- Subtask prediction accuracy
- Multi-modality evaluation

## Integration with LeRobot

This implementation uses the LeRobot Pi0.5 policy under the hood:
- Follows LeRobot's model architecture
- Compatible with LeRobot datasets and tools
- Supports LeRobot's training and evaluation pipelines

## Example Usage Script

For a complete example, see the example script that demonstrates:
- Model loading
- Training setup
- Prediction workflow
- Evaluation process

## Requirements

- LeRobot >= 0.4.3
- Transformers
- PyTorch >= 1.12
- Compatible with ark_ml framework

## Testing

Run tests to verify functionality:
```bash
python -m pytest tests_and_benchmarks/pi05_tests/
```

## Benchmarks

Run performance benchmarks:
```bash
python tests_and_benchmarks/pi05_benchmarks/benchmark_pi05.py
```

## Notes

- This implementation follows the same pattern as PiZero for consistency
- Multi-stage training requires different dataset configurations for each stage
- Flow matching is particularly effective for precise manipulation tasks
- FAST tokenization enables efficient action discretization during pretraining