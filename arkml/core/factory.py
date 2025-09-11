import ast
import inspect
from typing import Any, Dict, Mapping, Tuple

from omegaconf import DictConfig, OmegaConf

from ark_ml.arkml.core.registry import MODELS


def _to_plain_dict(cfg: DictConfig | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _normalise_shape(shape_dim: str):
    try:
        parsed = ast.literal_eval(shape_dim)
        if isinstance(parsed, (list, tuple)):
            return tuple(parsed)
        else:
            return shape_dim
    except (ValueError, TypeError):
        return shape_dim


def _normalize_model_cfg(model_cfg: DictConfig | Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    d = _to_plain_dict(model_cfg["model"])
    model_name = d.get("name") or d.get("type")
    if not model_name:
        raise KeyError("Model config must include 'name' or 'type'.")

    # Start with flat params (minus control keys)
    params: Dict[str, Any] = {k: v for k, v in d.items() if k not in ("name", "type")}
    # If nested block exists under the model name, merge it on top
    nested = d.get(model_name)
    if isinstance(nested, dict):
        params.update(nested)

    if "image_dim" in params and isinstance(params["image_dim"], str):
        params["image_dim"] = _normalise_shape(params["image_dim"])

    if "enable_lora" in params:
        params.update(model_cfg["lora"])

    return str(model_name), params


def _filter_kwargs_for_constructor(cls: type, params: Dict[str, Any]) -> Tuple[
    Dict[str, Any], Dict[str, Any], list[str]]:
    sig = inspect.signature(cls.__init__)
    accepted_names = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    # Exclude "self"
    accepted_names.discard("self")

    # Required = no default and not self
    required = [
        p.name
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
           and p.default is inspect._empty
           and p.name != "self"
    ]

    accepted = {k: v for k, v in params.items() if k in accepted_names}
    ignored = {k: v for k, v in params.items() if k not in accepted_names}

    missing = [r for r in required if r not in accepted]

    return accepted, ignored, missing


def build_model(model_cfg: DictConfig | Mapping[str, Any]):
    model_name, params = _normalize_model_cfg(model_cfg)
    model_cls = MODELS.get(model_name)

    filtered, ignored, missing = _filter_kwargs_for_constructor(model_cls, params)

    if missing:
        # Clear, actionable error
        missing_str = ", ".join(missing)
        raise TypeError(f"{model_name} missing required init args: {missing_str}")

    # Optional: log ignored extras (you can replace with logging)
    if ignored:
        print(f"[build_model] Ignoring extra model args for {model_name}: {list(ignored.keys())}")

    return model_cls(**filtered)
