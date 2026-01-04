"""
Hydra-based service launcher for policy nodes.

Uses the policy registry to instantiate the configured policy node
and spins it as a long-lived service that communicates over LCM.
"""

import hydra
import torch
from ark.client.comm_infrastructure.base_node import main
from arkml.utils.utils import ConfigPath
from arkml.core.app_context import ArkMLContext
from arkml.nodes.policy_registry import get_policy_node
from arkml.utils.schema_io import get_visual_features
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.3")
def rollout(cfg: DictConfig) -> None:
    print("Config:\n", OmegaConf.to_yaml(cfg))

    ArkMLContext.cfg = cfg
    ArkMLContext.global_config = ConfigPath(cfg.global_config).read_yaml()
    io_schema = ConfigPath(cfg["channel_schema"]).read_yaml()
    ArkMLContext.visual_input_features = get_visual_features(
        schema=io_schema["observation_space"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = get_policy_node(cfg)
    main(node, device=device)


if __name__ == "__main__":
    rollout()
