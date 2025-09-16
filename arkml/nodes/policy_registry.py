from typing import Callable

import torch
from omegaconf import DictConfig

# Global registry for policy node builders
_POLICY_BUILDERS: dict[str, Callable[[DictConfig, torch.device, int, str], object]] = {}


def register_policy(key: str):
    """Decorator to register a policy builder under a key.

    Args:
      key: Case-insensitive lookup string used to select a policy.

    Returns:
        A decorator that registers the wrapped function as a builder.
    """

    def _wrap(fn: Callable[[DictConfig, torch.device], object]):
        _POLICY_BUILDERS[key.lower()] = fn
        return fn

    return _wrap


def _get_policy_key(cfg: DictConfig) -> str:
    """Derive a registry key from the config.

    Args:
      cfg: Hydra DictConfig-like object containing ``algo`` and ``algo.model``.

    Returns:
        Lowercased key for registry lookup.
    """
    pt = getattr(cfg.algo.model, "policy_type", None)
    if pt:
        return str(pt).lower()
    return str(getattr(cfg.algo, "name", "")).lower()


def get_policy_node(cfg: DictConfig, device: torch.device):
    """Instantiate the appropriate policy node based on configuration.

    Args:
      cfg: Configuration used to determine the policy type.
      device: Target torch device for the underlying model.

    Returns:
      Policy node.

    """
    key = _get_policy_key(cfg)
    builder = _POLICY_BUILDERS.get(key)
    if builder is None:
        raise NotImplementedError(
            f"No policy builder registered for key '{key}'. Available: {list(_POLICY_BUILDERS.keys())}"
        )
    global_config = getattr(cfg, "global_config", None)
    return builder(cfg, device, cfg.stepper_frequency, global_config)


@register_policy("pizero")
@register_policy("pi0")
def _build_pizero(
    cfg: DictConfig,
    device: torch.device,
    stepper_frequency: int,
    global_config: str | None = None,
):
    """Build and return a PiZero policy node from config.

    Args:
      cfg: Hydra configuration with VLA model fields.
      device: Target device string for the policy node.

    Returns:
      Configured PiZeroPolicyNode  instance.
    """
    from arkml.nodes.pizero_node import PiZeroPolicyNode

    return PiZeroPolicyNode(
        model_cfg=cfg.algo.model,
        device=device,
        stepper_frequency=stepper_frequency,
        global_config=global_config,
    )


@register_policy("smolvla")
def _build_smolvla(
    cfg: DictConfig,
    device: torch.device,
    stepper_frequency,
    global_config: str | None = None,
):
    """Build and return SmolVLA that reuses the PiZero builder."""
    return _build_pizero(
        model_cfg=cfg,
        device=device,
        stepper_frequency=stepper_frequency,
        global_config=global_config,
    )
