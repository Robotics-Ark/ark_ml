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

