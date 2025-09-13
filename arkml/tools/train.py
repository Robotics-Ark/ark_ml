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
    """
    # Train a policy
    HYDRA_FULL_ERROR=1 python -m ark_ml.arkml.tools.train algo=pizero \
    data=pizero_dataset task_prompt="Pick the yellow cube and place it in the white background area of the table" \
    data.dataset_path=/path/tp/data/set
    """
    main()
