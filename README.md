# ark_ml

Machine learning backbone for the Ark Robotics Framework, providing core models, algorithms, and tools to enable
intelligent perception, decision-making, and control in robotic applications.

## 📦 Installation

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
HYDRA_FULL_ERROR=1 arkml-rollout algo=<ml_algorithm>  \
  algo.model.model_path=path/to/the/model \
  algo.model.task_prompt=task_prompt
```

## Training
Train a model with a dataset:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
arkml.tools.train algo=<ml_algorithm> \
  task_prompt=task_prompt /path/to/dataset

```