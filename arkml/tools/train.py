import hydra
import torch
from arkml.core.factory import build_model
from arkml.core.registry import ALGOS
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.3")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # Build model (policy)
    policy = build_model(cfg.algo)

    # Load algorithm
    algo_cls = ALGOS.get(cfg.algo.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo_cls(policy=policy, device=device, cfg=cfg)

    #  Run training
    history = algo.train()
    print("Training finished :", history)

    # Run Evaluation
    history = algo.eval()
    print("Validation finished :", history)


if __name__ == "__main__":
    main()
