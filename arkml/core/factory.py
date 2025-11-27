"""Factory helpers for constructing registered ArkML models from config blocks."""

import inspect
from typing import Any

from arkml.core.registry import MODELS
from arkml.utils.utils import _normalise_shape
from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """
    Convert a DictConfig or mapping into a resolved plain dictionary.
    Args:
        cfg: Configuration block to convert.

    Returns:
        Dictionary of configuration blocks.
    """
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _normalize_model_cfg(
    model_cfg: DictConfig | dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Collect the model name and flattened parameters from a configuration block.
    Args:
        model_cfg: Model configuration block.

    Returns:
        Returns model name and flattened parameters from a configuration block.
    """
    d = _to_plain_dict(model_cfg["model"])
    model_name = d.get("name") or d.get("type")
    if not model_name:
        raise KeyError("Model config must include 'name' or 'type'.")

    params: dict[str, Any] = {k: v for k, v in d.items() if k not in ("name", "type")}

    # If nested block exists under the model name, merge it on top
    nested = d.get(model_name)
    if isinstance(nested, dict):
        params.update(nested)

    if "image_dim" in params and isinstance(params["image_dim"], str):
        params["image_dim"] = _normalise_shape(params["image_dim"])

    return str(model_name), params


def _filter_kwargs_for_constructor(
    cls: type, params: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """
    Partition provided parameters based on what the class constructor accepts.
    Args:
        cls: Model class
        params: Model parameters

    Returns:
        Return accepted , missing and ignored parameters.

    """
    sig = inspect.signature(cls.__init__)
    accepted_names = {
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    # Exclude "self"
    accepted_names.discard("self")

    # Required = no default and not self
    required = [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and p.default is inspect._empty
        and p.name != "self"
    ]

    accepted = {k: v for k, v in params.items() if k in accepted_names}
    ignored = {k: v for k, v in params.items() if k not in accepted_names}

    missing = [r for r in required if r not in accepted]

    return accepted, ignored, missing


def build_model(model_cfg: dict[str, Any]):
    """
    Instantiate a registered model with constructor arguments from configuration.
    Args:
        model_cfg: model configuration block

    Returns:
        Corresponding model initialized with model configurations.
    """
    model_name, params = _normalize_model_cfg(model_cfg)
    model_cls = MODELS.get(model_name)

    filtered, ignored, missing = _filter_kwargs_for_constructor(model_cls, params)

    if missing:
        missing_str = ", ".join(missing)
        raise TypeError(f"{model_name} missing required init args: {missing_str}")

    if ignored:
        print(
            f"[build_model] Ignoring extra model args for {model_name}: {list(ignored.keys())}"
        )

    return model_cls(**filtered)
