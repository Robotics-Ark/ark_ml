from omegaconf import DictConfig

from ark_ml.arkml.core.registry import MODELS


# def build_model(model_cfg: DictConfig):
#     """Build a model from config using the MODELS registry."""
#     model_cls = MODELS.get(model_cfg.name)
#     kwargs = {k: v for k, v in model_cfg[model_cfg.name].items() if k != "name"}
#     return model_cls(**kwargs)


# from omegaconf import DictConfig
# from core.utils.registry import MODELS

def build_model(model_cfg: DictConfig):
    """Build a model from config using the MODELS registry.

    Args:
        model_cfg (DictConfig): Must contain 'name' key specifying the model name,
                                and a nested dict with parameters under that name.

    Returns:
        nn.Module: Instantiated model.
    """
    if "name" not in model_cfg:
        raise KeyError("model_cfg must contain a 'name' key specifying the model to build.")

    model_name = model_cfg.name
    model_cls = MODELS.get(model_name)

    if model_name not in model_cfg:
        return model_cls()

    kwargs = {k: v for k, v in model_cfg[model_name].items() if k != "name"}

    return model_cls(**kwargs)

