from omegaconf import DictConfig


class ArkMLContext:
    """Shared runtime configuration accessible across ArkML components.

    Attributes:
        cfg: Structured configuration resolved from Hydra/OmegaConf.
        global_config: Top-level Ark Robot settings dictionary.
        io_schema: Input/output schema metadata describing input output channel interfaces.
        visual_input_features: Metadata describing camera names/descriptors available.
    """

    cfg: DictConfig = None
    global_config: dict = None
    io_schema: dict = None
    visual_input_features: dict = None
