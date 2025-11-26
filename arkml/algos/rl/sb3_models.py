from typing import Any

from arkml.core.registry import MODELS


@MODELS.register("sb3_dummy")
class StableBaselinesDummyModel:
    """
    Placeholder model to satisfy ArkML's model factory when using
    external RL libraries such as Stable-Baselines3.

    The corresponding Algorithm class manages the actual RL policy
    internally, so this object is never used beyond construction.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Store config for debugging or future extensions, but otherwise unused.
        self.config: dict[str, Any] = dict(kwargs)

