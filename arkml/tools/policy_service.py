"""
Hydra-based service launcher for policy nodes.

Uses the policy registry to instantiate the configured policy node
and spins it as a long-lived service that communicates over LCM.
"""

import hydra
import torch
from ark.client.comm_infrastructure.base_node import main
from arkml.nodes.policy_registry import get_policy_node
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.3")
def rollout(cfg: DictConfig) -> None:
    print("Config:\n", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = get_policy_node(cfg)
    main(node, cfg=cfg, device=device)


if __name__ == "__main__":
    rollout()
