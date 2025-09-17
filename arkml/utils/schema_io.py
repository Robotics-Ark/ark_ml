from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import yaml

from arktypes.utils import unpack as _unpack, pack


# -------------------------
# Schema loading utilities
# -------------------------


def load_schema(config_path: str) -> dict:
    """Load a YAML IO schema file.

    The schema defines how to unpack observations and pack actions in a
    robot-agnostic way. See the example schema under
    `ark_ml/arkml/examples/franka_pick_place/franka_config/io_schema.yaml`.
    """
    cfg_path = Path(config_path)
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}
    else:
        raise FileNotFoundError(f"Config file could not found {cfg_path}")

    return cfg_dict


# -------------------------
# Observation unpacking
# -------------------------


def _extract_from_message(msg: Any, using: str, select: str, index: int | None = None):
    """Extract a field from an arktypes message using a standard unpacker.

    Args:
      msg: Serialized LCM message object from `ObservationSpace`.
      using: Name of unpack function under `arktypes.utils.unpack`.
      select: Name of the returned field to select.
      index: Optional index for vector-like fields (e.g., a specific joint).

    Returns:
      np.ndarray or native type as extracted from the message.
    """
    fn = getattr(_unpack, using)
    out = fn(msg)

    # Map field names to tuple positions for common arktypes unpackers.
    # Users can extend this by providing schema with explicit `field_idx` later if needed.
    FIELD_MAP = {
        # rigid_body_state(): name, position, orientation, lin_vel, ang_vel
        "rigid_body_state": {
            "name": 0,
            "position": 1,
            "orientation": 2,
            "lin_vel": 3,
            "ang_vel": 4,
        },
        # joint_state(): name, effort, position, velocity, name_to_idx
        "joint_state": {
            "name": 0,
            "effort": 1,
            "position": 2,
            "velocity": 3,
            "name_to_idx": 4,
        },
        # pose(): position, quaternion
        "pose": {"position": 0, "quaternion": 1},
        # rgbd(): rgb, depth
        "rgbd": {"rgb": 0, "depth": 1},
    }

    if using not in FIELD_MAP:
        raise ValueError(
            f"Unsupported unpacker '{using}'. Extend FIELD_MAP or provide another schema."
        )

    idx = FIELD_MAP[using].get(select)
    if idx is None:
        raise ValueError(f"Unsupported select '{select}' for unpacker '{using}'.")

    value = out[idx]

    # Indexing inside vectors, e.g., a specific joint position
    if index is not None:
        value = np.asarray(value)[index]

    return value


def make_observation_unpacker(schema: dict) -> Callable[..., dict[str, Any]]:
    """Create an observation unpacker from schema.

    The resulting function has signature:
      unpacker(observation_dict: dict[str, Any], obs_keys: list[str] | None = None,
               only: list[str] | str | None = None) -> dict[str, Any]
    """

    obs_schema = (schema or {}).get("observation", {})
    outputs: dict = obs_schema.get("outputs", {})

    def _unpack(
        observation_dict: dict[str, Any],
        obs_keys: list[str] | None = None,
        only: list[str] | str | None = None,
    ) -> dict[str, Any] | None:
        if not observation_dict or any(v is None for v in observation_dict.values()):
            return None

        # Normalize filter
        if isinstance(only, str):
            requested = {only}
        elif isinstance(only, list):
            requested = set(only)
        else:
            requested = None

        result: dict[str, Any] = {}

        for out_key, spec in outputs.items():
            if requested is not None and out_key not in requested:
                continue

            if "concat" in spec:
                parts = []
                for item in spec["concat"]:
                    ch = item["from"]
                    using = item["using"]
                    select = item.get("select", "")
                    index = item.get("index", None)
                    wrap = item.get("wrap", False)

                    msg = observation_dict.get(ch)
                    if msg is None:
                        raise KeyError(
                            f"Missing observation channel '{ch}'. Check channel_config and schema."
                        )
                    val = _extract_from_message(
                        msg, using=using, select=select, index=index
                    )
                    if wrap:
                        parts.append([val])
                    else:
                        parts.append(np.asarray(val))

                # Try concatenating; if incompatible shapes, store as list
                try:
                    result[out_key] = np.concatenate(parts)
                except Exception:
                    result[out_key] = parts
            else:
                # Single source
                ch = spec["from"]
                using = spec["using"]
                select = spec.get("select", "")
                index = spec.get("index", None)
                wrap = spec.get("wrap", False)

                msg = observation_dict.get(ch)
                if msg is None:
                    raise KeyError(
                        f"Missing observation channel '{ch}'. Check channel_config and schema."
                    )
                val = _extract_from_message(
                    msg, using=using, select=select, index=index
                )

                if wrap:
                    # Some policies expect a list of images, etc.
                    result[out_key] = [val]
                else:
                    result[out_key] = val

        return result

    return _unpack


# -------------------------
# Action packing
# -------------------------


def make_action_packer(schema: dict) -> Callable[..., dict[str, Any]]:
    """Create an action packer from schema.

    The resulting function has signature:
      packer(action: list[float] | np.ndarray, action_keys: list[str] | None = None) -> dict[str, Any]
    """

    act_schema = (schema or {}).get("action", {})
    using = act_schema.get("using")
    channel = act_schema.get("channel")

    if using is None or channel is None:
        raise ValueError("Action schema must define both 'using' and 'channel'.")

    def _pack(
        action: list[float] | np.ndarray, action_keys: list[str] | None = None
    ) -> dict[str, Any]:
        a = np.asarray(action).tolist()

        if using == "task_space_command":
            # Expected fields mapping in schema
            idx = act_schema.get("indices", {})
            xyz_idx: list[int] = idx.get("xyz", [0, 1, 2])
            quat_idx: list[int] = idx.get("quat", [3, 4, 5, 6])
            grip_idx: int = idx.get("gripper", 7)

            xyz = np.array([a[i] for i in xyz_idx])
            quat = np.array([a[i] for i in quat_idx])
            grip = a[grip_idx]

            msg = pack.task_space_command("all", xyz, quat, grip)
            return {channel: msg}

        raise ValueError(
            f"Unsupported packer '{using}'. Extend implementation as needed."
        )

    return _pack

