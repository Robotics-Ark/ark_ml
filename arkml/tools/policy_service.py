"""
Hydra-based service launcher for policy nodes.

Uses the policy registry to instantiate the configured policy node
and spins it as a long-lived service that communicates over LCM.
"""

import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from arkml.nodes.policy_registry import get_policy_node


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("Config:\n", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = get_policy_node(cfg, device)

    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.kill_node()
        except SystemExit:
            pass


if __name__ == "__main__":
    main()
