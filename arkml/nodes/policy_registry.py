from typing import Callable

import torch
from omegaconf import DictConfig
from arkml.core.factory import build_model
from arkml.nodes.act_node import ActPolicyNode
from arkml.nodes.diffusion_node import DiffusionPolicyNode
#from arkml.nodes.pizero_node import PiZeroPolicyNode
from arkml.nodes.policy_node import PolicyNode

# Global registry for policy node builders
_POLICY_BUILDERS: dict[str, Callable[[DictConfig, torch.device], object]] = {}


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
    return builder(cfg, device)


@register_policy("diffusion_policy")
def _build_diffusion(cfg: DictConfig, device: torch.device):
    """Build and return a DiffusionPolicyNode from config.

    Args:
      cfg: Configuration with diffusion model fields.
      device: Device to load the model on.

    Returns:
      Configured diffusion policy node.
    """
    model = build_model(cfg.algo.model).to(device)

    checkpoint_path = getattr(cfg, "checkpoint", None)
    if not checkpoint_path:
        raise ValueError("Model checkpoint path not provided")

    policy_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(policy_state, dict) and "state_dict" in policy_state:
        model.load_state_dict(policy_state["state_dict"])
    else:
        model.load_state_dict(policy_state)

    num_steps = cfg.algo.model.get("diffusion_steps", 100)
    return DiffusionPolicyNode(
        model=model,
        num_diffusion_iters=num_steps,
        pred_horizon=cfg.algo.model.get("pred_horizon", 16),
        action_dim=cfg.algo.model.get("action_dim", 8),
        device=str(device),
    )


@register_policy("pizero")
@register_policy("pi0")
def _build_pizero(cfg: DictConfig, device: torch.device):
    """Build and return a PiZero policy node from config.

    Args:
      cfg: Hydra configuration with VLA model fields.
      device: Target device string for the policy node.

    Returns:
      Configured PiZeroPolicyNode  instance.
    """
    pass
   # return PiZeroPolicyNode(model_cfg=cfg.algo.model, device=str(device))


@register_policy("smolvla")
def _build_smolvla(cfg: DictConfig, device: torch.device):
    """Build and return SmolVLA that reuses the PiZero builder."""
    return _build_pizero(cfg, device)

@register_policy("ACTransformer")
def _build_ACTransformer(cfg: DictConfig, device: torch.device):
    """Build and return ACTransformer"""
    return ActPolicyNode(model_cfg=cfg.algo.model, device=str(device))