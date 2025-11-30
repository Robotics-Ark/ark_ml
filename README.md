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

### Training Stages

#### Pretraining Stage
The pretraining stage focuses on learning foundational representations using multiple modalities and FAST tokenization:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
arkml-train algo=pi05 \
 data.dataset_path=/path/to/pi05/dataset \
 output_dir=/output/path \
 algo.model.policy_type=pi0.5 \
 algo.training.stage=pretrain \
 algo.training.pretrain_steps=280000
```

The pretraining stage optimizes:
- Cross-entropy loss for text tokens (CE(text))
- Cross-entropy loss for FAST tokens (CE(FAST tokens))

#### Post-training Stage
The post-training stage refines the model with flow matching and subtask prediction:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
arkml-train algo=pi05 \
 data.dataset_path=/path/to/pi05/dataset \
 output_dir=/output/path \
 algo.model.policy_type=pi0.5 \
 algo.training.stage=posttrain \
 algo.training.posttrain_steps=80000 \
 algo.training.flow_alpha=10.0
```

The post-training stage optimizes:
- Cross-entropy loss for subtasks (CE(subtask))
- Flow matching loss weighted by alpha (alpha * flow_matching_loss)

### Running Inference

To run inference with a trained Pi0.5 model:

```bash
HYDRA_FULL_ERROR=1 arkml-policy algo=pi05 \
  algo.model.model_path=path/to/pi05/model \
  policy_node_name=pi05_node
```

You can then call the inference endpoints:
- `pi05_node/policy/predict` - Get next action prediction
- `pi05_node/policy/reset` - Reset policy state
- `pi05_node/policy/start` - Start policy service
- `pi05_node/policy/stop` - Stop policy service

### Configuration Explanation

The Pi0.5 configuration includes several key parameters:

**Model Configuration:**
- `model.backbone_type`: Vision-language backbone architecture (e.g., 'siglip_gemma')
- `model.use_fast_tokens`: Whether to use FAST tokenizer for action discretization
- `model.use_flow_matching`: Whether to use flow matching for action prediction

**Training Configuration:**
- `training.stage`: Current training stage ('pretrain' or 'posttrain')
- `training.pretrain_steps`: Number of steps for pretraining (280000 default)
- `training.posttrain_steps`: Number of steps for post-training (80000 default)
- `training.integration_steps`: Number of steps for Euler integration in flow matching
- `training.flow_alpha`: Weight for flow matching loss (10.0 default)

**Dataset Configuration:**
The dataset configuration uses mixture sampling with:
- Primary dataset for main training data
- Secondary datasets for auxiliary data
- Configurable weights for balancing different data sources

The model uses a multi-head architecture with:
- Subtask head for high-level task planning
- FAST head for discretized action prediction
- Flow head for continuous action prediction using flow matching