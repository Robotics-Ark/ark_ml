from omegaconf import DictConfig


class ArkMLContext:
    """Shared runtime configuration accessible across ArkML components.

    Attributes:
        cfg: Structured configuration resolved from Hydra/OmegaConf.
        global_config: Top-level Ark Robot settings dictionary.
        visual_input_features: Metadata describing camera names/descriptors available.
    """

    cfg: DictConfig = None
    global_config: dict = None
    visual_input_features: dict = None
