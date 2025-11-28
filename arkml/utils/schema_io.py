from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from arkml.utils.utils import _resolve_channel_types
from arktypes.utils import unpack, pack

FIELD_MAP: dict[str, dict[str, int]] = {
    "joint_state": {
        "header": 0,
        "name": 1,
        "position": 2,
        "velocity": 3,
        "effort": 4,
    },
    "pose": {
        "position": 0,
        "orientation": 1,
    },
    "rigid_body_state": {
        "header": 0,
        "position": 1,
        "orientation": 2,
        "linear_velocity": 3,
        "angular_velocity": 4,
    },
    "rgbd": {
        "rgb": 0,
        "depth": 1,
    },
}


def load_yaml(config_path: str) -> dict:
    """
    Load a YAML configuration schema from a file.
    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The parsed configuration schema as a dictionary. If the file
        contains no data, an empty dictionary is returned.

    """
    cfg_path = Path(config_path)
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}
    else:
        raise FileNotFoundError(f"Config file could not found {cfg_path}")

    return cfg_dict


def get_visual_features(schema: dict) -> list[str]:
    visual_features = []
    for key, entries in schema.items():
        for item in entries:
            if item["using"] == "rgbd":
                visual_features.append(
                    item["name"] if "name" in item else item["using"]
                )
    return visual_features


def get_ark_fn_type(ark_module: unpack, name: str):
    """
    Retrieve both an unpacking function and its corresponding type from Ark module.
    Args:
        ark_module: The module (e.g., ``arktypes.utils.unpack``) containing the
        unpack functions and optional type definitions.
        name: The base name of the function/type pair to retrieve.

    Returns:
        A tuple (fn, dtype) where:
          - fn is the unpacking function corresponding to ``name``.
          - dtype is the associated type object if defined, otherwise None.

    """
    fn = getattr(ark_module, name)
    dtype = getattr(ark_module, f"{name}_t", None)
    return fn, dtype


def get_channel_types(schema: dict, channel_type: str) -> dict[str, type]:
    """
    Generate a mapping of observation channel names to Python/Ark types
    based on the observation schema.

    Args:
        schema (dict): Observation schema dictionary (from YAML or Python dict).
            Each channel entry can optionally include a 'type' key, which can be
            a string corresponding to a class in `arktypes` or a Python type.

    Returns:
        Dict[str, type]: Dictionary mapping channel name to resolved type.
    """
    channels: dict[str, Any] = {}

    obs_schema = schema.get(channel_type, {})

    for key, entries in obs_schema.items():
        for item in entries:
            ch_name = item["from"]
            using = item["using"]
            _, ch_type = get_ark_fn_type(ark_module=unpack, name=using)
            if ch_name not in channels:
                channels[ch_name] = ch_type

    # Resolve type strings to actual type objects using _resolve_channel_types
    resolved_channels = _resolve_channel_types(channels)
    return resolved_channels


def _dynamic_observation_unpacker(schema: dict) -> Callable:
    """
    Create a dynamic observation unpacker based on a schema.

    The schema should be in the format:
    observation:
      state:
        - from: channel_name
          using: callable
      image_top:
        - from: channel_name
          using: callable
          wrap: True  # optional

    Returns a function:
        _unpack(observation_dict) -> dict
    """

    obs_schema = schema.get("observation", {})

    def _unpack(observation_dict: dict[str, Any]) -> dict[str, Any]:
        if not observation_dict:
            return {}

        result: dict[str, Any] = {}

        for key, entries in obs_schema.items():
            parts = []
            for item in entries:
                ch_name = item["from"]
                wrap = item.get("wrap", False)
                using = item["using"]
                select_fields = item.get("select", None)
                msg = observation_dict.get(ch_name)
                if msg is None:
                    raise KeyError(f"Missing observation channel '{ch_name}'")

                fn, dtype = get_ark_fn_type(ark_module=unpack, name=using)
                ret = fn(msg)
                if select_fields:
                    field_map = FIELD_MAP.get(using, {})
                    selected = []
                    for f in select_fields:
                        if f in field_map:
                            selected.append(ret[field_map[f]])
                    val = selected[0] if len(selected) == 1 else selected
                else:
                    val = ret

                if wrap:
                    val = [val]  # wrap in the list for batch-like outputs

                parts.append(val)

            # Concatenate numeric arrays for state-like features
            try:
                if all(isinstance(p, np.ndarray) for p in parts):
                    result[key] = np.concatenate(parts)
                else:
                    result[key] = parts if len(parts) > 1 else parts[0]
            except (ValueError, TypeError):
                result[key] = parts
        return result

    return _unpack


def _dynamic_action_packer(schema: dict) -> Callable[..., dict[str, Any]]:
    """
    Create a dynamic action packer from schema.

    Returns a function:
        _pack(observation_dict) -> dict

    """

    act_schema = schema.get("action", {})
    entries = act_schema.get("action")
    if entries is None:
        raise ValueError("Action schema must define an 'action' entry.")

    def _pack(action: list[float] | np.ndarray) -> dict[str, Any]:
        a = np.asarray(action).tolist()
        result: dict[str, Any] = {}

        for item in entries:
            channel = item["from"]
            using = item["using"]
            select = item.get("select", {})

            # resolve packer dynamically
            fn, dtype = get_ark_fn_type(ark_module=pack, name=using)

            # build args from config
            args = []
            for field_name, idx in select.items():
                if isinstance(idx, list):
                    args.append(np.array([a[i] for i in idx]))
                elif isinstance(idx, str):
                    args.append(idx)
                else:
                    args.append(a[idx])

            msg = fn(*args)
            result[channel] = msg

        return result

    return _pack


if __name__ == "__main__":
    global_config_path = (
        "ark_ml/arkml/examples/franka_pick_place/franka_config/global_config.yaml"
    )
    global_schema = load_yaml(config_path=global_config_path)
    channel_schema = load_yaml(config_path=global_schema["channel_config"])
    obs_channels = get_channel_types(schema=channel_schema, channel_type="observation")
