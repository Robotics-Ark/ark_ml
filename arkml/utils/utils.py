import importlib
from typing import Any


def resolve_channel_types(mapping: dict[str, Any]) -> dict[str, type]:
    """Resolve type names from config into arktypes classes.

    Accepts either already-imported classes or string names present in the
    ``arktypes`` package. Returns a mapping of channel name to type.
    """
    if not mapping:
        return {}
    resolved: dict[str, type] = {}
    arktypes_mod = importlib.import_module("arktypes")
    for ch_name, t in mapping.items():
        if isinstance(t, str):
            resolved[ch_name] = getattr(arktypes_mod, t)
        else:
            resolved[ch_name] = t
    return resolved
