# ark_ml

Machine learning backbone for the Ark Robotics Framework, providing core models, algorithms, and tools to enable
intelligent perception, decision-making, and control in robotic applications.

## Installation

### 1. Set up environment

1. Create and activate conda/virtual environment.

```bash
conda create -n ark_env python=3.11 -y
conda activate ark_env
```

2. Clone repository.

```bash
  - (ssh) `git clone git@github.com:Robotics-Ark/ark_ml.git`
  - (https) `git clone https://github.com/Robotics-Ark/ark_ml.git`
```

3. Install ark_ml framework.

```bash
pip install -e ark_ml
```
4. Install other Ark modules.
   The ML backbone depends on additional Ark modules. Make sure the following are also installed:

- [ark_framework](https://github.com/Robotics-Ark/ark_framework)
- [ark_types](https://github.com/Robotics-Ark/ark_types)

```bash
pip install -e ark_framework
pip install -e ark_types
```
## Rollout
Use the trained model for rollouts:

```bash
HYDRA_FULL_ERROR=1 arkml-policy algo=<ml_algorithm>  \
  algo.model.model_path=path/to/the/model \
```
### Policy services available
### Start Service
Start the policy service to get machine learning based action prediction

```bash
PolicyName/policy/stop
```
`PolicyName` can be set through command line by using below command 

```bash
HYDRA_FULL_ERROR=1 arkml-policy algo=<ml_algorithm>  \
  algo.model.model_path=path/to/the/model \
  policy_node_name=policy_name
```

### Stop Service
Stop the policy service to pause machine learning based action prediction

```bash
PolicyName/policy/stop
```

### Reset Service
Reset the policy state

```bash
PolicyName/policy/stop
```

### Predict Service
predict next action

```bash
PolicyName/policy/predict
```

## Client  Service
A client service can be started using below command
```bash
python  ark_ml/arkml/examples/franka_pick_place/franka_pick_place.py --max_step 1000 --n_episodes 5 --step_sleep 0.1
```

## Training
Train a model with a dataset:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
arkml.tools.train algo=<ml_algorithm> \
 data.dataset_path=/path/to/dataset \
 output_dir=/output/path

```

## Pi0.5

Pi0.5 is an upgraded version of the Pi0 Vision-Language-Action model with enhanced capabilities for robotic manipulation tasks. It features a multi-stage training approach with flow matching for precise action prediction.

### Pi0.5 Architecture Overview

Pi0.5 is a Vision-Language-Action (VLA) model that uses a unique flow matching approach to predict continuous robot actions. The architecture includes:

- **Vision Backbone**: Processes visual input using vision transformers (CLIP-based)
- **Q-Former**: Cross-attention module for vision-language fusion
- **Multi-Modal Fusion**: Combines vision and language representations
- **Multi-Head Architecture**:
  - Subtask Head: Predicts high-level subtask tokens
  - FAST Head: Predicts discretized action tokens
  - Flow Head: Uses flow matching for continuous action prediction

### Installation Requirements

To use Pi0.5, you need to have the following dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install omegaconf hydra-core
pip install huggingface_hub
```

### Dataset Preparation

Pi0.5 requires a specific dataset format with multiple modalities:

1. **Image data**: RGB images from robot cameras
2. **Action data**: Continuous robot actions (e.g., joint positions, end-effector poses)
3. **Subtask data**: High-level subtask instructions
4. **Language data**: Task instructions and descriptions

The dataset should be structured to support mixture sampling with configurable weights for different data sources.

### Training Stages

Pi0.5 uses a two-stage training process:

#### Stage 1: Pretraining
The pretraining stage focuses on learning foundational representations using multiple modalities and FAST tokenization:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
arkml-train algo=pi05 \
 data.dataset_path=/path/to/pi05/dataset \
 output_dir=/output/path \
 algo.model.policy_type=pi0.5 \
 algo.training.stage=pretrain \
 algo.training.pretrain_steps=280000 \
 algo.training.lr=2e-4 \
 algo.training.batch_size=8
```

During pretraining, the model optimizes:
- Cross-entropy loss for text tokens (CE(text))
- Cross-entropy loss for FAST tokens (CE(FAST tokens))

#### Stage 2: Post-training
The post-training stage refines the model with flow matching and subtask prediction:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
arkml-train algo=pi05 \
 data.dataset_path=/path/to/pi05/dataset \
 output_dir=/output/path \
 algo.model.policy_type=pi0.5 \
 algo.training.stage=posttrain \
 algo.training.posttrain_steps=80000 \
 algo.training.flow_alpha=10.0 \
 algo.training.lr=2e-4 \
 algo.training.batch_size=8
```

During post-training, the model optimizes:
- Cross-entropy loss for subtasks (CE(subtask))
- Flow matching loss weighted by alpha (alpha * flow_matching_loss)

### Configuration Parameters

**Model Configuration:**
- `model.policy_type`: Set to "pi0.5" for Pi0.5 models
- `model.model_path`: Path to the pre-trained model or checkpoint
- `model.action_dim`: Dimension of continuous robot actions (default: 8)
- `model.obs_dim`: Dimension of observation space (default: 9)
- `model.image_dim`: Image dimensions [channels, height, width] (default: [3, 480, 640])
- `model.use_fast_tokens`: Whether to use FAST tokenizer for action discretization
- `model.use_flow_matching`: Whether to use flow matching for action prediction

**Training Configuration:**
- `training.stage`: Current training stage ('pretrain' or 'posttrain')
- `training.pretrain_steps`: Number of steps for pretraining (280000 default)
- `training.posttrain_steps`: Number of steps for post-training (80000 default)
- `training.flow_alpha`: Weight for flow matching loss (10.0 default)
- `training.integration_steps`: Number of steps for Euler integration in flow matching
- `training.lr`: Learning rate (2e-4 default)
- `training.batch_size`: Training batch size (8 default)
- `training.max_epochs`: Maximum training epochs (10 default)
- `training.use_bf16`: Whether to use bfloat16 mixed precision (true default)

**Dataset Configuration:**
- `dataset.mixture.primary_dataset`: Primary dataset for training
- `dataset.mixture.secondary_datasets`: List of secondary datasets
- `dataset.mixture.weights.primary`: Weight for primary dataset (0.7 default)
- `dataset.mixture.weights.secondary`: Weight for secondary datasets (0.3 default)
- `data.dataset_path`: Path to the dataset directory

### Running Inference

#### Starting the Policy Service

To run inference with a trained Pi0.5 model:

```bash
HYDRA_FULL_ERROR=1 arkml-policy algo=pi05 \
  algo.model.model_path=path/to/pi05/model \
  policy_node_name=pi05_node \
  algo.model.policy_type=pi0.5
```

#### Inference Endpoints

Once the service is running, you can call the following endpoints:
- `pi05_node/policy/predict` - Get next action prediction
- `pi05_node/policy/reset` - Reset policy state
- `pi05_node/policy/start` - Start policy service
- `pi05_node/policy/stop` - Stop policy service

#### Example Inference Script

For direct Python inference without the service:

```python
import torch
from arkml.nodes.pi05_node import Pi05Node
from arkml.algos.vla.pi05.models import Pi05Policy

# Load the trained model
model = Pi05Policy(
    policy_type="pi0.5",
    model_path="/path/to/trained/pi05/model",
    obs_dim=9,
    action_dim=8,
    image_dim=(3, 480, 640)
)

# Load checkpoint (if available)
model.load_state_dict(torch.load("/path/to/model/checkpoint.pth"))

# Create the policy node
policy_node = Pi05Node(model, device="cuda" if torch.cuda.is_available() else "cpu")

# Prepare observation (image and other sensor data)
observation = {
    "image": torch.randn(1, 3, 480, 640),  # Example image tensor
    # Add other observation components as needed
}

# Get action prediction
predicted_action = policy_node.predict(observation)
print(f"Predicted action: {predicted_action}")
```

### Flow Matching Algorithm

Pi0.5 uses flow matching for continuous action prediction, which involves:

1. **Vector Field Learning**: The model learns to predict a vector field that describes how to transform the current state to the target action
2. **Euler Integration**: During inference, the model uses Euler integration to follow the learned flow and generate a continuous action trajectory
3. **Loss Function**: The flow matching loss computes the difference between predicted and target flow vectors

This approach enables Pi0.5 to generate smooth, precise continuous actions that are essential for robotic manipulation tasks.

### Tokenization System

Pi0.5 uses the FAST (Fast Action Sequence Tokenizer) system which:

- Discretizes continuous action values into discrete tokens during pretraining
- Uses configurable binning parameters (num_bins, min_val, max_val)
- Supports encode/decode roundtrips with quantization error handling
- Enables efficient pretraining with cross-entropy loss

### Evaluation Metrics

The Pi0.5 evaluation system measures:

- **Subtask Accuracy**: Accuracy of subtask token predictions
- **Action MSE/MAE**: Mean squared error and mean absolute error for continuous actions
- **Threshold Accuracy**: Percentage of actions within a reasonable threshold of ground truth
- **Flow Matching Performance**: Quality of flow-based action generation

### Troubleshooting

#### Common Issues:

1. **Memory Issues**: Use gradient accumulation and mixed precision (BF16) to reduce memory usage
2. **NaN Values**: Check for gradient clipping and proper normalization of action values
3. **Poor Performance**: Ensure proper stage-based training (pretrain → post-train)

#### Configuration Validation:

Before training, validate your configuration:
```bash
python -c "from arkml.core.factory import load_config; print(load_config(algo='pi05'))"
```

### Example Training Run

Complete example for full Pi0.5 training:

```bash
# 1. Pretrain stage
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
arkml-train algo=pi05 \
 data.dataset_path=/path/to/dataset \
 output_dir=/output/pretrain \
 algo.training.stage=pretrain \
 algo.training.pretrain_steps=1000 \
 algo.trainer.max_epochs=5

# 2. Post-train stage (using pre-trained model)
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
arkml-train algo=pi05 \
 data.dataset_path=/path/to/dataset \
 output_dir=/output/posttrain \
 algo.training.stage=posttrain \
 algo.training.posttrain_steps=500 \
 algo.training.flow_alpha=10.0 \
 algo.trainer.max_epochs=3 \
 algo.model.model_path=/output/pretrain/final_model.pth
```