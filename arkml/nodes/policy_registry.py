from collections.abc import Callable

from omegaconf import DictConfig

from arkml.core.policy import BasePolicy

# Global registry for policy node builders
_POLICY_BUILDERS: dict[str, Callable] = {}


def register_policy(key: str):
    """Decorator to register a policy builder under a key.

    Args:
      key: Case-insensitive lookup string used to select a policy.

    Returns:
        A decorator that registers the wrapped function as a builder.
    """

    def _wrap(fn: Callable[[DictConfig], object]):
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


def get_policy_node(cfg: DictConfig) -> BasePolicy:
    """Instantiate the appropriate policy node based on configuration.

    Args:
      cfg: Configuration used to determine the policy type.

    Returns:
      Policy node.

    """
    key = _get_policy_key(cfg)
    builder = _POLICY_BUILDERS.get(key)
    if builder is None:
        raise NotImplementedError(
            f"No policy builder registered for key '{key}'. Available: {list(_POLICY_BUILDERS.keys())}"
        )
    return builder()


@register_policy("pizero")
@register_policy("pi0")
def _build_pizero() -> BasePolicy:
    """Build and return a PiZero policy node from config.

    Returns:
      PiZeroPolicyNode .
    """
    from arkml.nodes.pizero_node import PiZeroPolicyNode

    return PiZeroPolicyNode

@register_policy("act")
def _build_ACT():
    """Build and return ACT"""
    from arkml.nodes.act_policy_node import ActPolicyNode

    return ActPolicyNode

@register_policy("diffusion_policy")
def _build_diffusion() -> BasePolicy:
    """Build and return a DiffusionPolicyNode from config.


    Returns:
      DiffusionPolicyNode.
    """
    from arkml.nodes.diffusion_node import DiffusionPolicyNode

    return DiffusionPolicyNode
