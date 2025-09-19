import ast
import importlib
from typing import Any


def _normalise_shape(shape_dim: str) -> tuple:
    """
    Parse a shape string into a normalized tuple of dimensions.
    Args:
        shape_dim: A string representation of a shape, e.g. "(3, 224, 224)" or "[64, 128]".

    Returns:
        A tuple of integers if the string could be parsed into a list or tuple.

    """
    try:
        parsed = ast.literal_eval(shape_dim)
        if isinstance(parsed, (list, tuple)):
            return tuple(parsed)
        else:
            raise ValueError(f"shape {shape_dim} failed to convert to a list/tuple")
    except (ValueError, TypeError):
        raise ValueError(f"shape {shape_dim} failed to convert to a list/tuple")


def _resolve_channel_types(mapping: dict[str, Any]) -> dict[str, type]:
    """
    Resolve a mapping of channel names to Ark types.
    Accepts either already-imported classes or string names present in the
    ``arktypes`` package. Returns a mapping of channel name to type.
    Args:
        A dictionary mapping channel names to resolved Ark type objects.

    Returns:

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
