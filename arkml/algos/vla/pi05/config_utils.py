"""Helper utilities for Pi05 configuration."""

from __future__ import annotations

import json
from typing import Iterable, Sequence


_DEFAULT_CAMERA_NAMES = ["image_top", "image_wrist"]


def resolve_visual_feature_names(value) -> list[str]:
    """Normalize a user-provided visual feature configuration into a name list."""

    if value is None:
        count = 1
        names = _DEFAULT_CAMERA_NAMES[:count]
        if len(names) < count:
            names.extend(_generate_extra_names(count - len(names)))
        return names

    if isinstance(value, int):
        if value <= 0:
            raise ValueError("visual_input_features must be a positive integer")
        names = _DEFAULT_CAMERA_NAMES[:value]
        if len(names) < value:
            names.extend(_generate_extra_names(value - len(names)))
        return names

    if isinstance(value, str):

        text = value.strip()
        if text.startswith("["):
            try:
                loaded = json.loads(text)
                return resolve_visual_feature_names(loaded)
            except json.JSONDecodeError:
                pass
        return [text]

    if isinstance(value, Sequence):
        names = [str(item) for item in value]
        if not names:
            raise ValueError("visual_input_features sequence cannot be empty")
        return names

    if isinstance(value, dict):
        if "names" in value:
            names = value["names"]
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, Iterable):
                raise TypeError("visual_input_features.names must be iterable")
            names = [str(item) for item in names]
            if not names:
                raise ValueError("visual_input_features.names cannot be empty")
            return names
        if "count" in value:
            return resolve_visual_feature_names(int(value["count"]))

    raise TypeError(
        "visual_input_features must be an int, str, sequence, or dict with 'names'."
    )


def _generate_extra_names(count: int) -> list[str]:
    return [f"image_extra_{idx}" for idx in range(count)]